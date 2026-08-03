import importlib.util
from pathlib import Path

import pytest
import torch


def _load_fft_module():
    repo_root = Path(__file__).resolve().parents[4]
    module_path = repo_root / "python" / "tutorials" / "tle" / "01-fft.py"
    spec = importlib.util.spec_from_file_location("tle_fft_tutorial", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
@pytest.mark.parametrize("n", [64, 128, 256, 512, 1024])
def test_fft_tutorial_supported_sizes(n):
    module = _load_fft_module()
    torch.manual_seed(0)
    x = module._make_input(16, n, torch.float32, False)
    expected = torch.fft.fft(x.to(torch.complex64))

    torch.testing.assert_close(module.triton_fft(x), expected, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(module.tle_fft(x), expected, rtol=1e-3, atol=1e-3)
