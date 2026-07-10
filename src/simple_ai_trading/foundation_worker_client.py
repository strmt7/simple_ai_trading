"""Supervisor for the process-isolated foundation-model worker."""

from __future__ import annotations

import json
import os
import queue
import signal
import subprocess  # nosec B404 - fixed interpreter/module invocation, no shell
import sys
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


@dataclass(frozen=True)
class WorkerPrediction:
    predicted_closes: tuple[tuple[float, ...], ...]
    worker_seconds: float
    worker_pid: int


class FoundationWorkerError(RuntimeError):
    def __init__(self, message: str, *, restartable: bool) -> None:
        super().__init__(message)
        self.restartable = bool(restartable)


class FoundationWorkerSupervisor:
    """Own one worker process and enforce bounded request deadlines."""

    def __init__(
        self,
        *,
        model_size: str,
        backend: str,
        source_cache_root: str | Path | None,
        require_accelerator: bool,
        startup_timeout_seconds: float = 120.0,
        request_timeout_seconds: float = 60.0,
    ) -> None:
        self.model_size = str(model_size)
        self.backend = str(backend)
        self.source_cache_root = (
            str(Path(source_cache_root).resolve()) if source_cache_root is not None else None
        )
        self.require_accelerator = bool(require_accelerator)
        self.startup_timeout_seconds = max(1.0, float(startup_timeout_seconds))
        self.request_timeout_seconds = max(1.0, float(request_timeout_seconds))
        self.process: subprocess.Popen[str] | None = None
        self._runtime_pid: int | None = None
        self.report: dict[str, object] | None = None
        self._stdout: queue.Queue[str | None] = queue.Queue()
        self._stderr: deque[str] = deque(maxlen=100)
        self._request_id = 0

    @property
    def pid(self) -> int | None:
        if self._runtime_pid is not None:
            return self._runtime_pid
        return self.process.pid if self.process is not None else None

    @property
    def launcher_pid(self) -> int | None:
        return self.process.pid if self.process is not None else None

    @property
    def stderr_tail(self) -> tuple[str, ...]:
        return tuple(self._stderr)

    @staticmethod
    def _pump_stdout(stream: Any, target: queue.Queue[str | None]) -> None:
        try:
            for line in stream:
                target.put(line)
        except (OSError, ValueError):
            pass
        finally:
            target.put(None)

    @staticmethod
    def _pump_stderr(stream: Any, target: deque[str]) -> None:
        try:
            for line in stream:
                target.append(line.rstrip())
        except (OSError, ValueError):
            pass

    def _command(self) -> list[str]:
        command = [
            sys.executable,
            "-u",
            "-m",
            "simple_ai_trading.foundation_worker",
            "--model-size",
            self.model_size,
            "--backend",
            self.backend,
        ]
        if self.source_cache_root:
            command.extend(("--source-cache", self.source_cache_root))
        if self.require_accelerator:
            command.append("--require-accelerator")
        return command

    def start(self) -> dict[str, object]:
        self.stop()
        self._stdout = queue.Queue()
        self._stderr = deque(maxlen=100)
        creation_flags = 0
        if os.name == "nt":
            creation_flags = int(getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self.process = subprocess.Popen(  # nosec B603
            self._command(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=creation_flags,
        )
        assert self.process.stdout is not None
        assert self.process.stderr is not None
        threading.Thread(
            target=self._pump_stdout,
            args=(self.process.stdout, self._stdout),
            daemon=True,
            name="foundation-worker-stdout",
        ).start()
        threading.Thread(
            target=self._pump_stderr,
            args=(self.process.stderr, self._stderr),
            daemon=True,
            name="foundation-worker-stderr",
        ).start()
        message = self._message(self.startup_timeout_seconds, startup=True)
        if message.get("type") != "ready" or not isinstance(message.get("report"), dict):
            self.stop()
            raise FoundationWorkerError(
                f"foundation worker did not become ready: {message}",
                restartable=False,
            )
        runtime_pid = int(message.get("worker_pid", 0))
        if runtime_pid <= 0:
            self.stop()
            raise FoundationWorkerError(
                "foundation worker omitted its runtime PID",
                restartable=False,
            )
        self._runtime_pid = runtime_pid
        self.report = dict(message["report"])
        return self.report

    def _message(self, timeout_seconds: float, *, startup: bool = False) -> dict[str, Any]:
        try:
            line = self._stdout.get(timeout=max(0.1, float(timeout_seconds)))
        except queue.Empty as exc:
            tail = " | ".join(self.stderr_tail[-3:])
            raise FoundationWorkerError(
                f"foundation worker {'startup' if startup else 'request'} timed out"
                + (f": {tail}" if tail else ""),
                restartable=not startup,
            ) from exc
        if line is None:
            code = self.process.poll() if self.process is not None else None
            tail = " | ".join(self.stderr_tail[-3:])
            raise FoundationWorkerError(
                f"foundation worker exited with code {code}"
                + (f": {tail}" if tail else ""),
                restartable=not startup,
            )
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise FoundationWorkerError(
                f"foundation worker emitted invalid JSON: {line[:200]!r}",
                restartable=False,
            ) from exc
        if not isinstance(payload, dict):
            raise FoundationWorkerError(
                "foundation worker response was not an object",
                restartable=False,
            )
        return payload

    @staticmethod
    def _context_payload(context: Any) -> dict[str, object]:
        frame = context.frame
        history_ms = (
            context.history_timestamps.astype("int64").to_numpy(dtype="int64") // 1_000_000
        )
        future_ms = (
            context.future_timestamps.astype("int64").to_numpy(dtype="int64") // 1_000_000
        )
        return {
            "columns": [str(column) for column in frame.columns],
            "values": frame.to_numpy(dtype="float32").tolist(),
            "history_ms": [int(value) for value in history_ms],
            "future_ms": [int(value) for value in future_ms],
        }

    def predict(
        self,
        contexts: Sequence[Any],
        *,
        prediction_length: int,
        temperature: float,
        top_k: int,
        top_p: float,
        sample_count: int,
        seed: int,
    ) -> WorkerPrediction:
        process = self.process
        if process is None or process.poll() is not None or process.stdin is None:
            raise FoundationWorkerError("foundation worker is not running", restartable=True)
        self._request_id += 1
        request_id = self._request_id
        request = {
            "type": "predict",
            "id": request_id,
            "prediction_length": int(prediction_length),
            "temperature": float(temperature),
            "top_k": int(top_k),
            "top_p": float(top_p),
            "sample_count": int(sample_count),
            "seed": int(seed),
            "contexts": [self._context_payload(context) for context in contexts],
        }
        try:
            process.stdin.write(json.dumps(request, separators=(",", ":"), allow_nan=False) + "\n")
            process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise FoundationWorkerError(
                f"foundation worker request pipe failed: {exc}",
                restartable=True,
            ) from exc
        message = self._message(self.request_timeout_seconds)
        if message.get("type") == "error":
            raise FoundationWorkerError(
                f"foundation worker inference failed with "
                f"{message.get('exception_type', 'Error')}: {message.get('message', '')}",
                restartable=True,
            )
        if message.get("type") != "result" or int(message.get("id", -1)) != request_id:
            raise FoundationWorkerError(
                f"foundation worker protocol mismatch: {message}",
                restartable=False,
            )
        response_pid = int(message.get("worker_pid", 0))
        if response_pid <= 0 or response_pid != self._runtime_pid:
            raise FoundationWorkerError(
                "foundation worker runtime PID changed within a session",
                restartable=False,
            )
        raw_predictions = message.get("predicted_closes")
        if not isinstance(raw_predictions, list) or len(raw_predictions) != len(contexts):
            raise FoundationWorkerError(
                "foundation worker returned the wrong prediction count",
                restartable=False,
            )
        predictions = tuple(tuple(float(value) for value in values) for values in raw_predictions)
        if any(len(values) != int(prediction_length) for values in predictions):
            raise FoundationWorkerError(
                "foundation worker returned the wrong prediction horizon",
                restartable=False,
            )
        return WorkerPrediction(
            predicted_closes=predictions,
            worker_seconds=float(message.get("seconds", 0.0)),
            worker_pid=response_pid,
        )

    @staticmethod
    def _terminate_runtime_pid(runtime_pid: int | None) -> None:
        if runtime_pid is None or runtime_pid <= 0 or runtime_pid == os.getpid():
            return
        try:
            os.kill(runtime_pid, signal.SIGTERM)
        except (OSError, ProcessLookupError):
            pass

    def stop(self) -> None:
        process = self.process
        runtime_pid = self._runtime_pid
        self.process = None
        self._runtime_pid = None
        self.report = None
        if process is None:
            return
        if process.stdin is not None:
            try:
                process.stdin.close()
            except OSError:
                pass
        if process.poll() is None:
            try:
                process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                if runtime_pid != process.pid:
                    self._terminate_runtime_pid(runtime_pid)
                try:
                    process.terminate()
                    process.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    try:
                        process.kill()
                        process.wait(timeout=3.0)
                    except (OSError, subprocess.TimeoutExpired):
                        pass
                except OSError:
                    pass
            except OSError:
                if runtime_pid != process.pid:
                    self._terminate_runtime_pid(runtime_pid)
        elif runtime_pid != process.pid:
            # A Windows venv launcher can exit while its interpreter child remains.
            self._terminate_runtime_pid(runtime_pid)
        for stream in (process.stdout, process.stderr):
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass

    def __enter__(self) -> "FoundationWorkerSupervisor":
        self.start()
        return self

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        self.stop()


__all__ = [
    "FoundationWorkerError",
    "FoundationWorkerSupervisor",
    "WorkerPrediction",
]
