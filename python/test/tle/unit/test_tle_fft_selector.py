import importlib.util
from pathlib import Path
from types import SimpleNamespace


def _load_fft_module():
    repo_root = Path(__file__).resolve().parents[4]
    module_path = repo_root / "python" / "tutorials" / "tle" / "01-fft.py"
    spec = importlib.util.spec_from_file_location("tle_fft_tutorial", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _set_target(monkeypatch, module, backend: str, arch: str):
    target = SimpleNamespace(backend=backend, arch=arch)
    monkeypatch.setattr(
        module.triton.runtime.driver.active,
        "get_current_target",
        lambda: target,
    )


def test_fft_provider_gfx1201_boundaries(monkeypatch):
    module = _load_fft_module()
    _set_target(monkeypatch, module, "hip", "gfx1201")

    assert module._fft_provider(2048, 256) == "triton"
    assert module._fft_provider(4096, 64) == "tle"
    assert module._fft_provider(4096, 512) == "tle"
    assert module._fft_provider(4096, 1024) == "triton"
    assert module._fft_provider(8192, 512) == "tle"
    assert module._fft_provider(16384, 256) == "triton"


def test_fft_provider_other_targets_preserve_tle(monkeypatch):
    module = _load_fft_module()

    for backend, arch in (("hip", "gfx942"), ("cuda", "sm90")):
        _set_target(monkeypatch, module, backend, arch)
        assert module._fft_provider(256, 1024) == "tle"


def test_fft_num_warps_gfx1201(monkeypatch):
    module = _load_fft_module()
    _set_target(monkeypatch, module, "hip", "gfx1201")

    assert module._fft_num_warps("triton", 4096, 64) == 2
    assert module._fft_num_warps("tle", 4096, 256) == 1
    assert module._fft_num_warps("triton", 4096, 512) == 8
    assert module._fft_num_warps("tle", 4096, 1024) == 8
    assert module._fft_num_warps("tle", 4096, 32) == 4
    assert module._fft_num_warps("tle", 512, 256) == 4
