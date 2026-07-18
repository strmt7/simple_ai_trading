"""Tests for the opt-in compute backend resolver."""

from __future__ import annotations

import pytest

from simple_ai_trading.compute import (
    BackendInfo,
    SUPPORTED_COMPUTE_BACKENDS,
    backend_fallback_allowed,
    default_compute_backend,
    describe_backend,
    resolve_backend,
)


def test_resolve_backend_defaults_to_host_neutral_auto_discovery() -> None:
    info = resolve_backend(None)
    assert isinstance(info, BackendInfo)
    assert default_compute_backend() == "auto"
    assert info.requested == "auto"
    assert info.kind in {"cuda", "rocm", "xpu", "directml", "mps", "cpu"}
    assert info.request_satisfied is True


def test_resolve_backend_cpu_is_always_available() -> None:
    info = resolve_backend("cpu")
    assert info.kind == "cpu"
    assert info.requested == "cpu"


def test_resolve_backend_cuda_falls_back_with_reason_when_unavailable() -> None:
    info = resolve_backend("cuda")
    # In the unit-test environment torch is not present so we expect a graceful
    # fallback rather than an exception.
    if info.kind == "cpu":
        assert "CUDA" in info.reason
        assert info.requested == "cuda"
    else:
        assert info.kind == "cuda"
        assert info.device.startswith("cuda")


def test_resolve_backend_cuda_returns_cpu_when_torch_missing(monkeypatch) -> None:
    monkeypatch.setattr(
        "simple_ai_trading.compute._probe_torch",
        lambda: (None, "torch missing in test"),
    )

    info = resolve_backend("cuda")

    assert info.kind == "cpu"
    assert info.reason == "CUDA unavailable or its selected device is invalid"


def test_resolve_backend_rocm_falls_back_with_reason_when_unavailable() -> None:
    info = resolve_backend("rocm")
    if info.kind == "cpu":
        assert "ROCm" in info.reason
        assert info.requested == "rocm"
    else:
        assert info.kind == "rocm"


def test_resolve_backend_rocm_returns_cpu_when_torch_missing(monkeypatch) -> None:
    monkeypatch.setattr(
        "simple_ai_trading.compute._probe_torch",
        lambda: (None, "torch missing in test"),
    )

    info = resolve_backend("rocm")

    assert info.kind == "cpu"
    assert info.reason == "ROCm unavailable or its selected device is invalid"


def test_resolve_backend_directml_falls_back_with_reason_when_unavailable() -> None:
    info = resolve_backend("directml")
    if info.kind == "cpu":
        assert "DirectML" in info.reason
        assert info.requested == "directml"
    else:
        assert info.kind == "directml"


def test_resolve_backend_auto_falls_back_to_cpu_without_torch() -> None:
    info = resolve_backend("auto")
    if info.kind == "cpu":
        assert "GPU" in info.reason or "CPU" in info.reason
        assert info.requested == "auto"


def test_resolve_backend_auto_returns_cpu_when_every_probe_is_missing(monkeypatch) -> None:
    monkeypatch.setattr(
        "simple_ai_trading.compute._probe_torch",
        lambda: (None, "torch missing in test"),
    )
    monkeypatch.setattr(
        "simple_ai_trading.compute._probe_torch_directml",
        lambda: (None, "directml missing in test"),
    )

    info = resolve_backend("auto")

    assert info.kind == "cpu"
    assert info.reason == "No supported accelerator passed runtime discovery; using CPU reference"


def test_resolve_backend_unknown_value_is_cpu_with_explanation() -> None:
    info = resolve_backend("ferrari")
    assert info.kind == "cpu"
    assert "ferrari" in info.reason


def test_describe_backend_includes_components() -> None:
    info = resolve_backend("cpu")
    text = describe_backend(info)
    assert "compute=cpu" in text
    assert "device=cpu" in text
    assert "vendor=portable CPU reference" in text
    assert "selection=deterministic_cpu_reference" in text


def test_describe_backend_includes_reason_when_present() -> None:
    info = resolve_backend("cuda")
    text = describe_backend(info)
    if info.kind == "cpu":
        assert info.reason in text
    else:
        assert info.reason == ""


class _FakeCuda:
    def __init__(self, *, available: bool = True, count: int = 1, name: str = "Test GPU") -> None:
        self._available = available
        self._count = count
        self._name = name

    def is_available(self) -> bool:
        return self._available

    def device_count(self) -> int:
        return self._count

    def get_device_name(self, index: int) -> str:
        return self._name


class _FakeMpsBackend:
    def __init__(self, *, available: bool = True) -> None:
        self._available = available

    def is_available(self) -> bool:
        return self._available


class _FakeBackends:
    def __init__(self, mps: _FakeMpsBackend | None = None) -> None:
        if mps is not None:
            self.mps = mps


def _make_fake_torch(
    *,
    cuda: _FakeCuda | None = None,
    hip: str | None = None,
    mps: _FakeMpsBackend | None = None,
    include_version: bool = True,
):
    class _FakeTorch:
        pass

    fake = _FakeTorch()
    if cuda is not None:
        fake.cuda = cuda
    if include_version:
        class _Version:
            pass

        version = _Version()
        if hip is not None:
            version.hip = hip
        fake.version = version
    fake.backends = _FakeBackends(mps=mps)
    return fake


def test_resolve_backend_cuda_success_uses_torch(monkeypatch) -> None:
    fake = _make_fake_torch(cuda=_FakeCuda(name="NVIDIA Test"))
    monkeypatch.setattr(
        "simple_ai_trading.compute._probe_torch",
        lambda: (fake, ""),
    )
    info = resolve_backend("cuda")
    assert info.kind == "cuda"
    assert info.device == "cuda:0"
    assert info.vendor == "NVIDIA Test"
    assert info.reason == ""


def test_resolve_backend_cuda_returns_cpu_when_not_available(monkeypatch) -> None:
    fake = _make_fake_torch(cuda=_FakeCuda(available=False))
    monkeypatch.setattr(
        "simple_ai_trading.compute._probe_torch",
        lambda: (fake, ""),
    )
    info = resolve_backend("cuda")
    assert info.kind == "cpu"


def test_resolve_backend_cuda_returns_cpu_when_zero_devices(monkeypatch) -> None:
    fake = _make_fake_torch(cuda=_FakeCuda(count=0))
    monkeypatch.setattr(
        "simple_ai_trading.compute._probe_torch",
        lambda: (fake, ""),
    )
    info = resolve_backend("cuda")
    assert info.kind == "cpu"


def test_resolve_backend_cuda_returns_cpu_for_rocm_torch_build(monkeypatch) -> None:
    fake = _make_fake_torch(cuda=_FakeCuda(name="AMD Via CUDA Namespace"), hip="6.0")
    monkeypatch.setattr(
        "simple_ai_trading.compute._probe_torch",
        lambda: (fake, ""),
    )
    info = resolve_backend("cuda")
    assert info.kind == "cpu"


def test_resolve_backend_cuda_swallows_torch_exceptions(monkeypatch) -> None:
    class _ExplodingCuda:
        def is_available(self):
            raise RuntimeError("driver missing")

    fake = _make_fake_torch(cuda=_ExplodingCuda())
    monkeypatch.setattr(
        "simple_ai_trading.compute._probe_torch",
        lambda: (fake, ""),
    )
    info = resolve_backend("cuda")
    assert info.kind == "cpu"


def test_resolve_backend_rocm_success_via_torch_hip(monkeypatch) -> None:
    fake = _make_fake_torch(
        cuda=_FakeCuda(name="AMD Test"),
        hip="6.0",
    )
    monkeypatch.setattr(
        "simple_ai_trading.compute._probe_torch",
        lambda: (fake, ""),
    )
    info = resolve_backend("rocm")
    assert info.kind == "rocm"
    assert info.device == "cuda:0"


def test_resolve_backend_rocm_returns_cpu_when_no_hip(monkeypatch) -> None:
    fake = _make_fake_torch(cuda=_FakeCuda())
    monkeypatch.setattr(
        "simple_ai_trading.compute._probe_torch",
        lambda: (fake, ""),
    )
    info = resolve_backend("rocm")
    assert info.kind == "cpu"


def test_resolve_backend_rocm_returns_cpu_when_cuda_namespace_missing(monkeypatch) -> None:
    fake = _make_fake_torch(cuda=_FakeCuda(available=False), hip="6.0")
    monkeypatch.setattr(
        "simple_ai_trading.compute._probe_torch",
        lambda: (fake, ""),
    )
    info = resolve_backend("rocm")
    assert info.kind == "cpu"


def test_resolve_backend_rocm_returns_cpu_when_no_devices(monkeypatch) -> None:
    cuda = _FakeCuda()
    cuda._count = 0  # is_available=True, but no device count
    fake = _make_fake_torch(cuda=cuda, hip="6.0")
    monkeypatch.setattr(
        "simple_ai_trading.compute._probe_torch",
        lambda: (fake, ""),
    )
    info = resolve_backend("rocm")
    assert info.kind == "cpu"


class _FakeDirectML:
    def __init__(self, *, available: bool = True, explode: bool = False) -> None:
        self._available = available
        self._explode = explode

    def is_available(self) -> bool:
        if self._explode:
            raise RuntimeError("driver")
        return self._available

    def device(self) -> str:
        return "privateuseone:0"


def test_resolve_backend_directml_success(monkeypatch) -> None:
    fake_torch = _make_fake_torch()
    monkeypatch.setattr(
        "simple_ai_trading.compute._probe_torch",
        lambda: (fake_torch, ""),
    )
    monkeypatch.setattr(
        "simple_ai_trading.compute._probe_torch_directml",
        lambda: (_FakeDirectML(), ""),
    )
    info = resolve_backend("directml")
    assert info.kind == "directml"
    assert info.device == "privateuseone:0"
    assert info.vendor == "DirectML"


def test_resolve_backend_directml_returns_cpu_when_unavailable(monkeypatch) -> None:
    fake_torch = _make_fake_torch()
    monkeypatch.setattr(
        "simple_ai_trading.compute._probe_torch",
        lambda: (fake_torch, ""),
    )
    monkeypatch.setattr(
        "simple_ai_trading.compute._probe_torch_directml",
        lambda: (_FakeDirectML(available=False), ""),
    )
    assert resolve_backend("directml").kind == "cpu"


def test_resolve_backend_directml_swallows_exceptions(monkeypatch) -> None:
    fake_torch = _make_fake_torch()
    monkeypatch.setattr(
        "simple_ai_trading.compute._probe_torch",
        lambda: (fake_torch, ""),
    )
    monkeypatch.setattr(
        "simple_ai_trading.compute._probe_torch_directml",
        lambda: (_FakeDirectML(explode=True), ""),
    )
    assert resolve_backend("directml").kind == "cpu"


def test_resolve_backend_mps_success(monkeypatch) -> None:
    fake = _make_fake_torch(mps=_FakeMpsBackend())
    monkeypatch.setattr(
        "simple_ai_trading.compute._probe_torch",
        lambda: (fake, ""),
    )
    info = resolve_backend("mps")
    assert info.kind == "mps"
    assert info.device == "mps"


def test_resolve_backend_mps_returns_cpu_when_torch_missing(monkeypatch) -> None:
    monkeypatch.setattr(
        "simple_ai_trading.compute._probe_torch",
        lambda: (None, "torch missing in test"),
    )

    info = resolve_backend("mps")

    assert info.kind == "cpu"
    assert info.reason == "MPS unavailable or its selected device is invalid"


def test_resolve_backend_mps_returns_cpu_when_unavailable(monkeypatch) -> None:
    fake = _make_fake_torch(mps=_FakeMpsBackend(available=False))
    monkeypatch.setattr(
        "simple_ai_trading.compute._probe_torch",
        lambda: (fake, ""),
    )
    info = resolve_backend("mps")
    assert info.kind == "cpu"


def test_resolve_backend_mps_returns_cpu_when_attribute_missing(monkeypatch) -> None:
    fake = _make_fake_torch()
    fake.backends = _FakeBackends(mps=None)
    monkeypatch.setattr(
        "simple_ai_trading.compute._probe_torch",
        lambda: (fake, ""),
    )
    info = resolve_backend("mps")
    assert info.kind == "cpu"


def test_resolve_backend_mps_swallows_exceptions(monkeypatch) -> None:
    class _ExplodingMps:
        def is_available(self):
            raise RuntimeError("not on this OS")

    fake = _make_fake_torch(mps=_ExplodingMps())
    monkeypatch.setattr(
        "simple_ai_trading.compute._probe_torch",
        lambda: (fake, ""),
    )
    info = resolve_backend("mps")
    assert info.kind == "cpu"


def test_resolve_backend_auto_picks_first_available(monkeypatch) -> None:
    fake = _make_fake_torch(cuda=_FakeCuda(name="NVIDIA Auto"))
    monkeypatch.setattr(
        "simple_ai_trading.compute._probe_torch",
        lambda: (fake, ""),
    )
    monkeypatch.setattr(
        "simple_ai_trading.compute._probe_torch_directml",
        lambda: (None, "torch-directml unavailable in this test"),
    )
    info = resolve_backend("auto")
    assert info.kind == "cuda"
    assert info.requested == "auto"


def test_resolve_backend_auto_picks_mps_when_cuda_and_rocm_missing(monkeypatch) -> None:
    fake = _make_fake_torch(mps=_FakeMpsBackend())
    monkeypatch.setattr(
        "simple_ai_trading.compute._probe_torch",
        lambda: (fake, ""),
    )
    monkeypatch.setattr(
        "simple_ai_trading.compute._probe_torch_directml",
        lambda: (None, "torch-directml unavailable in this test"),
    )
    info = resolve_backend("auto")
    assert info.kind == "mps"
    assert info.requested == "auto"


def test_resolve_backend_auto_prefers_native_mps_over_legacy_directml(monkeypatch) -> None:
    fake = _make_fake_torch(mps=_FakeMpsBackend())
    monkeypatch.setattr(
        "simple_ai_trading.compute._probe_torch",
        lambda: (fake, ""),
    )
    monkeypatch.setattr(
        "simple_ai_trading.compute._probe_torch_directml",
        lambda: (_FakeDirectML(), ""),
    )
    info = resolve_backend("auto")
    assert info.kind == "mps"
    assert info.requested == "auto"


def test_probe_torch_returns_tuple_when_torch_missing(monkeypatch) -> None:
    """Direct exercise of the _probe_torch fallback path."""
    import builtins
    import importlib

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("simulated missing torch")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    # Reload the function reference
    from simple_ai_trading import compute as compute_mod

    importlib.reload(compute_mod)
    torch_obj, reason = compute_mod._probe_torch()
    assert torch_obj is None
    assert "torch" in reason.lower()
    importlib.reload(compute_mod)  # restore for other tests


def test_probe_torch_returns_torch_when_importable(monkeypatch) -> None:
    """Cover the success branch of _probe_torch via a fake sys.modules entry."""
    import sys
    import types

    fake_torch = types.SimpleNamespace()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    from simple_ai_trading import compute as compute_mod

    obj, reason = compute_mod._probe_torch()
    assert obj is fake_torch
    assert reason == ""


def test_probe_torch_directml_success_and_missing(monkeypatch) -> None:
    import importlib
    import types

    fake_directml = types.SimpleNamespace()
    original = importlib.import_module

    def fake_import(name: str):
        if name == "torch_directml":
            return fake_directml
        return original(name)

    from simple_ai_trading import compute as compute_mod

    monkeypatch.setattr(compute_mod.importlib, "import_module", fake_import)
    obj, reason = compute_mod._probe_torch_directml()
    assert obj is fake_directml
    assert reason == ""

    def missing_import(name: str):
        raise ImportError(name)

    monkeypatch.setattr(compute_mod.importlib, "import_module", missing_import)
    obj, reason = compute_mod._probe_torch_directml()
    assert obj is None
    assert "torch-directml" in reason


def test_resolve_backend_rocm_hip_attribute_falsy_returns_cpu(monkeypatch) -> None:
    """torch.version exists but torch.version.hip is missing/None â€” must fall back without raising."""
    fake = _make_fake_torch(cuda=_FakeCuda(), include_version=True, hip=None)
    monkeypatch.setattr(
        "simple_ai_trading.compute._probe_torch",
        lambda: (fake, ""),
    )
    info = resolve_backend("rocm")
    assert info.kind == "cpu"


class _FakeXpu(_FakeCuda):
    def current_device(self) -> int:
        return 0


def test_resolve_backend_xpu_success(monkeypatch) -> None:
    fake = _make_fake_torch()
    fake.xpu = _FakeXpu(name="Intel Arc Test")
    monkeypatch.setattr("simple_ai_trading.compute._probe_torch", lambda: (fake, ""))

    info = resolve_backend("xpu")

    assert info.kind == "xpu"
    assert info.device == "xpu:0"
    assert info.vendor == "Intel Arc Test"
    assert info.request_satisfied is True


def test_modern_torch_accelerator_routes_to_xpu(monkeypatch) -> None:
    class _Device:
        type = "xpu"

    class _Accelerator:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def current_accelerator(*, check_available: bool):
            assert check_available is True
            return _Device()

    fake = _make_fake_torch()
    fake.accelerator = _Accelerator()
    fake.xpu = _FakeXpu(name="Intel Generic Accelerator")
    monkeypatch.setattr("simple_ai_trading.compute._probe_torch", lambda: (fake, ""))
    monkeypatch.setattr(
        "simple_ai_trading.compute._probe_torch_directml",
        lambda: (None, "not installed"),
    )

    info = resolve_backend("auto")

    assert info.kind == "xpu"
    assert info.requested == "auto"
    assert info.vendor == "Intel Generic Accelerator"


def test_cuda_selects_device_with_most_reported_free_memory(monkeypatch) -> None:
    class _MemoryCuda(_FakeCuda):
        def __init__(self) -> None:
            super().__init__(count=2)

        @staticmethod
        def current_device() -> int:
            return 0

        @staticmethod
        def mem_get_info(index: int) -> tuple[int, int]:
            return ((2_000, 4_000) if index == 0 else (3_000, 4_000))

        @staticmethod
        def get_device_name(index: int) -> str:
            return f"NVIDIA Test {index}"

    fake = _make_fake_torch(cuda=_MemoryCuda())
    monkeypatch.setattr("simple_ai_trading.compute._probe_torch", lambda: (fake, ""))

    info = resolve_backend("cuda")

    assert info.device == "cuda:1"
    assert info.vendor == "NVIDIA Test 1"
    assert info.selection == "maximum_reported_free_memory:3000"


def test_operator_device_index_override_is_validated(monkeypatch) -> None:
    fake = _make_fake_torch(cuda=_FakeCuda(count=2, name="NVIDIA Test"))
    monkeypatch.setattr("simple_ai_trading.compute._probe_torch", lambda: (fake, ""))
    monkeypatch.setenv("SIMPLE_AI_TRADING_DEVICE_INDEX", "1")

    selected = resolve_backend("cuda")

    assert selected.device == "cuda:1"
    assert selected.selection.endswith("=1")

    monkeypatch.setenv("SIMPLE_AI_TRADING_DEVICE_INDEX", "2")
    rejected = resolve_backend("cuda")
    assert rejected.kind == "cpu"
    assert rejected.request_satisfied is False


def test_pinned_backend_can_be_required_fail_closed(monkeypatch) -> None:
    from simple_ai_trading import compute as compute_module

    monkeypatch.setattr("simple_ai_trading.compute._try_cuda", lambda: None)

    diagnostic = resolve_backend("cuda")
    assert diagnostic.fell_back is True
    assert backend_fallback_allowed(diagnostic) is False

    with pytest.raises(
        compute_module.BackendUnavailableError,
        match="requested compute backend 'cuda'",
    ):
        resolve_backend("cuda", require=True)


def test_auto_cpu_reference_is_a_satisfied_portable_resolution(monkeypatch) -> None:
    for name in (
        "_try_torch_accelerator",
        "_try_rocm",
        "_try_cuda",
        "_try_xpu",
        "_try_mps",
        "_try_directml",
    ):
        monkeypatch.setattr(f"simple_ai_trading.compute.{name}", lambda: None)

    info = resolve_backend("auto", require=True)

    assert info.kind == "cpu"
    assert info.request_satisfied is True
    assert backend_fallback_allowed(info) is True


def test_supported_backend_contract_is_explicit_and_host_independent() -> None:
    assert SUPPORTED_COMPUTE_BACKENDS == (
        "auto",
        "cpu",
        "cuda",
        "rocm",
        "xpu",
        "mps",
        "directml",
    )

