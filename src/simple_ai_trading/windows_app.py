"""Windows-first desktop app backed by the same command contract as the CLI."""

from __future__ import annotations

import queue
import subprocess  # nosec B404
import sys
import threading
import webbrowser
import tkinter as tk
from tkinter import messagebox, ttk

from .command_contract import CommandSpec, command_specs
from .config import load_runtime, load_strategy, save_runtime
from .compute import describe_backend, resolve_backend


WINDOWS_APP_COMMANDS = tuple(spec.name for spec in command_specs())


def _command_label(spec: CommandSpec) -> str:
    return spec.name.replace("-", " ").title()


class TradingWindowsApp(tk.Tk):
    """Small enterprise operator shell around the shared CLI."""

    def __init__(self) -> None:
        super().__init__()
        self.title("Simple AI Trading")
        self.geometry("1180x760")
        self.minsize(980, 620)
        self.specs = command_specs()
        self.output_queue: queue.Queue[str] = queue.Queue()
        self.selected_command = tk.StringVar(value=self.specs[0].name if self.specs else "")
        runtime = load_runtime()
        self.ai_enabled = tk.BooleanVar(value=runtime.ai_enabled)
        self.ai_require_gpu = tk.BooleanVar(value=runtime.ai_require_gpu)
        self.reinvest_warning_ack = tk.BooleanVar(value=False)
        self.status_text = tk.StringVar(value="Ready")
        self._build()
        self.after(250, self._startup_preflight)
        self.after(100, self._drain_output)

    def _build(self) -> None:
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        nav = ttk.Frame(self, padding=8)
        nav.grid(row=0, column=0, sticky="ns")
        ttk.Label(nav, text="Workflows").pack(anchor="w")
        self.command_list = tk.Listbox(nav, height=30, exportselection=False)
        for spec in self.specs:
            self.command_list.insert(tk.END, _command_label(spec))
        self.command_list.pack(fill="y", expand=True)
        self.command_list.bind("<<ListboxSelect>>", self._on_select)
        if self.specs:
            self.command_list.selection_set(0)

        main = ttk.Frame(self, padding=10)
        main.grid(row=0, column=1, sticky="nsew")
        main.columnconfigure(0, weight=1)
        main.rowconfigure(3, weight=1)

        ttk.Label(main, text="Command").grid(row=0, column=0, sticky="w")
        self.detail = ttk.Label(main, text="", wraplength=820, justify="left")
        self.detail.grid(row=1, column=0, sticky="ew", pady=(2, 12))

        controls = ttk.LabelFrame(main, text="Safety and AI")
        controls.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        controls.columnconfigure(4, weight=1)
        ttk.Checkbutton(controls, text="Enable AI features", variable=self.ai_enabled, command=self._save_ai).grid(row=0, column=0, sticky="w", padx=6, pady=6)
        ttk.Checkbutton(controls, text="Require GPU for AI", variable=self.ai_require_gpu, command=self._save_ai).grid(row=0, column=1, sticky="w", padx=6, pady=6)
        ttk.Button(controls, text="AI Preflight", command=lambda: self._run_command(["ai"])).grid(row=0, column=2, padx=6, pady=6)
        ttk.Button(controls, text="Risk Report", command=lambda: self._run_command(["risk", "--paper"])).grid(row=0, column=3, padx=6, pady=6)
        ttk.Button(controls, text="Backtest Graph", command=lambda: self._run_command(["backtest-chart"])).grid(row=0, column=4, padx=6, pady=6)
        ttk.Button(controls, text="Open Graph", command=self._open_default_graph).grid(row=0, column=5, padx=6, pady=6)
        ttk.Button(controls, text="Stop Autonomous", command=lambda: self._run_command(["autonomous", "stop"])).grid(row=0, column=6, sticky="e", padx=6, pady=6)

        runbar = ttk.Frame(main)
        runbar.grid(row=3, column=0, sticky="nsew")
        runbar.columnconfigure(0, weight=1)
        runbar.rowconfigure(1, weight=1)
        ttk.Button(runbar, text="Run Selected", command=self._run_selected).grid(row=0, column=0, sticky="w", pady=(0, 6))
        self.output = tk.Text(runbar, wrap="word", height=24)
        self.output.grid(row=1, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(runbar, orient="vertical", command=self.output.yview)
        scroll.grid(row=1, column=1, sticky="ns")
        self.output.configure(yscrollcommand=scroll.set)

        status = ttk.Label(self, textvariable=self.status_text, anchor="w", padding=6)
        status.grid(row=1, column=0, columnspan=2, sticky="ew")
        self._sync_detail()

    def _on_select(self, _event=None) -> None:
        selected = self.command_list.curselection()
        if selected:
            self.selected_command.set(self.specs[selected[0]].name)
            self._sync_detail()

    def _sync_detail(self) -> None:
        spec = next((item for item in self.specs if item.name == self.selected_command.get()), None)
        if spec is None:
            self.detail.configure(text="")
            return
        option_count = len(spec.options) + len(spec.positionals)
        self.detail.configure(text=f"{spec.name}: {spec.help}\nOptions in shared CLI contract: {option_count}")

    def _save_ai(self) -> None:
        runtime = load_runtime()
        runtime.ai_enabled = bool(self.ai_enabled.get())
        runtime.ai_require_gpu = bool(self.ai_require_gpu.get())
        backend = resolve_backend(runtime.compute_backend)
        if backend.kind == "cpu" and runtime.ai_enabled:
            runtime.ai_enabled = False
            self.ai_enabled.set(False)
            messagebox.showwarning(
                "CPU-only mode",
                "AI features require a GPU backend. CPU-only training and backtesting remain available, but AI has been disabled.",
            )
        save_runtime(runtime)
        self.status_text.set("AI settings saved")

    def _startup_preflight(self) -> None:
        runtime = load_runtime()
        backend = resolve_backend(runtime.compute_backend)
        if backend.kind == "cpu":
            if runtime.ai_enabled:
                runtime.ai_enabled = False
                save_runtime(runtime)
                self.ai_enabled.set(False)
            messagebox.showwarning(
                "CPU-only mode",
                "No GPU compute backend is currently active. The app will run without AI and training/backtesting will be slower.\n\n"
                f"{describe_backend(backend)}",
            )
            self.status_text.set("CPU-only mode; AI disabled")
        else:
            self.status_text.set(describe_backend(backend))

    def _run_selected(self) -> None:
        command = self.selected_command.get()
        if command == "strategy":
            strategy = load_strategy()
            if strategy.reinvest_profits and not self.reinvest_warning_ack.get():
                if not messagebox.askyesno(
                    "Profit reinvestment warning",
                    "Profit reinvestment compounds both gains and losses. Continue?",
                ):
                    return
                self.reinvest_warning_ack.set(True)
        self._run_command([command])

    def _run_command(self, args: list[str]) -> None:
        if not args or not args[0]:
            return
        self.status_text.set(f"Running {' '.join(args)}")
        self.output.insert(tk.END, f"\n> simple-ai-trading {' '.join(args)}\n")
        self.output.see(tk.END)

        def worker() -> None:
            try:
                completed = subprocess.run(  # nosec B603
                    [sys.executable, "-m", "simple_ai_trading", *args],
                    check=False,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                self.output_queue.put(completed.stdout)
                self.output_queue.put(f"\n(exit {completed.returncode})\n")
            except OSError as exc:
                self.output_queue.put(f"Failed to launch command: {exc}\n")
            self.output_queue.put("__DONE__")

        threading.Thread(target=worker, daemon=True).start()

    def _open_default_graph(self) -> None:
        path = "data/backtest_performance.svg"
        try:
            webbrowser.open(path)
        except Exception as exc:
            self.output.insert(tk.END, f"Could not open {path}: {exc}\n")

    def _drain_output(self) -> None:
        while True:
            try:
                item = self.output_queue.get_nowait()
            except queue.Empty:
                break
            if item == "__DONE__":
                self.status_text.set("Ready")
                continue
            self.output.insert(tk.END, item)
            self.output.see(tk.END)
        self.after(100, self._drain_output)


def main() -> int:
    app = TradingWindowsApp()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
