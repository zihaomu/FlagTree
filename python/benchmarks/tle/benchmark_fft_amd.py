"""Reproducible AMD benchmark for the TLE FFT tutorial."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Callable

import torch

import triton

import benchmark_amd as amd_bench


REPO_ROOT = Path(__file__).resolve().parents[3]
FFT_TUTORIAL_PATH = REPO_ROOT / "python" / "tutorials" / "tle" / "01-fft.py"
BALANCED_PROVIDER_ORDERS = (
    ("tle", "triton", "torch"),
    ("triton", "torch", "tle"),
    ("torch", "tle", "triton"),
    ("torch", "triton", "tle"),
    ("triton", "tle", "torch"),
    ("tle", "torch", "triton"),
)
SHAPES = {
    "n64": {"m": 4096, "n": 64},
    "n128": {"m": 4096, "n": 128},
    "n256": {"m": 4096, "n": 256},
    "n512": {"m": 4096, "n": 512},
    "n1024": {"m": 4096, "n": 1024},
}


def _load_fft_tutorial() -> ModuleType:
    spec = importlib.util.spec_from_file_location("tle_fft_tutorial", FFT_TUTORIAL_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load FFT tutorial from {FFT_TUTORIAL_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FFT = _load_fft_tutorial()


def _torch_dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }[name]


def _comparison(
    baseline_rounds: list[dict[str, float]],
    candidate_rounds: list[dict[str, float]],
    baseline_summary: dict[str, object],
    candidate_summary: dict[str, object],
) -> dict[str, object]:
    speedup = baseline_summary["p50_ms_median"] / candidate_summary["p50_ms_median"]
    return {
        "speedup": speedup,
        "speedup_95_ci": amd_bench._bootstrap_speedup_ci(baseline_rounds, candidate_rounds),
    }


def _measure_providers(
    launches: dict[str, Callable[[], None]],
    rounds: int,
    stabilization_rounds: int,
    warmup_ms: int,
    rep_ms: int,
) -> dict[str, object]:
    samples: dict[str, list[dict[str, float]]] = {provider: [] for provider in launches}
    stabilization_provider_order: list[list[str]] = []
    provider_order_by_round: list[list[str]] = []

    for launch in launches.values():
        launch()
    torch.cuda.synchronize()

    for round_index in range(stabilization_rounds):
        order = list(BALANCED_PROVIDER_ORDERS[round_index % len(BALANCED_PROVIDER_ORDERS)])
        stabilization_provider_order.append(order)
        for provider in order:
            triton.testing.do_bench(
                launches[provider],
                warmup=warmup_ms,
                rep=rep_ms,
                quantiles=amd_bench.QUANTILES,
            )

    for round_index in range(rounds):
        order_index = stabilization_rounds + round_index
        order = list(BALANCED_PROVIDER_ORDERS[order_index % len(BALANCED_PROVIDER_ORDERS)])
        provider_order_by_round.append(order)
        for provider in order:
            p50_ms, p20_ms, p80_ms = triton.testing.do_bench(
                launches[provider],
                warmup=warmup_ms,
                rep=rep_ms,
                quantiles=amd_bench.QUANTILES,
            )
            samples[provider].append({
                "p20_ms": float(p20_ms),
                "p50_ms": float(p50_ms),
                "p80_ms": float(p80_ms),
            })

    summaries = {
        provider: amd_bench._summarize_rounds(provider_rounds)
        for provider, provider_rounds in samples.items()
    }
    return {
        "providers": summaries,
        "stabilization_provider_order": stabilization_provider_order,
        "provider_order_by_round": provider_order_by_round,
        "tle_vs_triton": _comparison(
            samples["triton"],
            samples["tle"],
            summaries["triton"],
            summaries["tle"],
        ),
        "tle_vs_torch": _comparison(
            samples["torch"],
            samples["tle"],
            summaries["torch"],
            summaries["tle"],
        ),
        "triton_vs_torch": _comparison(
            samples["torch"],
            samples["triton"],
            summaries["torch"],
            summaries["triton"],
        ),
    }


def _run_shape(
    shape_name: str,
    shape: dict[str, int],
    dtype_name: str,
    complex_input: bool,
    rounds: int,
    stabilization_rounds: int,
    warmup_ms: int,
    rep_ms: int,
) -> dict[str, object]:
    m = shape["m"]
    n = shape["n"]
    dtype = _torch_dtype(dtype_name)
    x = FFT._make_input(m, n, dtype, complex_input)
    in_real, in_imag = FFT._prepare_input(x)
    bitrev = FFT._bitrev_indices(n, x.device)
    twiddle_real, twiddle_imag = FFT._twiddle_tables(n, x.device)
    log_n = FFT._log2(n)

    triton_buffers = [
        torch.empty((m, n), device=x.device, dtype=torch.float32)
        for _ in range(4)
    ]
    tle_real = torch.empty((m, n), device=x.device, dtype=torch.float32)
    tle_imag = torch.empty_like(tle_real)
    torch_input = torch.complex(in_real, in_imag)
    torch_output = torch.empty_like(torch_input)
    triton_num_warps = FFT._fft_num_warps("triton", m, n)
    tle_num_warps = FFT._fft_num_warps("tle", m, n)
    tle_kernel = FFT.fft_kernel_tle_reg if n == FFT._FFT_REG_THRESHOLD else FFT.fft_kernel_tle

    def launch_triton() -> None:
        FFT.fft_kernel_triton[(m, )](
            in_real,
            in_imag,
            bitrev,
            twiddle_real,
            twiddle_imag,
            *triton_buffers,
            in_real.stride(0),
            triton_buffers[0].stride(0),
            m,
            N=n,
            LOG_N=log_n,
            num_warps=triton_num_warps,
            num_stages=1,
        )

    def launch_tle() -> None:
        tle_kernel[(m, )](
            in_real,
            in_imag,
            bitrev,
            twiddle_real,
            twiddle_imag,
            tle_real,
            tle_imag,
            in_real.stride(0),
            tle_real.stride(0),
            m,
            N=n,
            LOG_N=log_n,
            num_warps=tle_num_warps,
            num_stages=1,
        )

    def launch_torch() -> None:
        torch.fft.fft(torch_input, out=torch_output)

    launches = {
        "triton": launch_triton,
        "tle": launch_tle,
        "torch": launch_torch,
    }
    for launch in launches.values():
        launch()
    torch.cuda.synchronize()

    num_passes = (log_n + 1) // 2
    if num_passes % 2 == 0:
        triton_real, triton_imag = triton_buffers[0], triton_buffers[1]
    else:
        triton_real, triton_imag = triton_buffers[2], triton_buffers[3]
    torch.testing.assert_close(
        torch.complex(triton_real, triton_imag),
        torch_output,
        rtol=1e-3,
        atol=1e-3,
    )
    torch.testing.assert_close(
        torch.complex(tle_real, tle_imag),
        torch_output,
        rtol=1e-3,
        atol=1e-3,
    )

    selected_provider = FFT._fft_provider(m, n)
    measurements = _measure_providers(
        launches,
        rounds,
        stabilization_rounds,
        warmup_ms,
        rep_ms,
    )
    if selected_provider == "tle":
        selected_vs_triton = dict(measurements["tle_vs_triton"])
        selected_vs_torch = dict(measurements["tle_vs_torch"])
    else:
        selected_vs_triton = {"speedup": 1.0, "speedup_95_ci": [1.0, 1.0]}
        selected_vs_torch = dict(measurements["triton_vs_torch"])
    measurements["selected_vs_triton"] = selected_vs_triton
    measurements["selected_vs_torch"] = selected_vs_torch

    return {
        "case": "fft",
        "shape_name": shape_name,
        "selected_provider": selected_provider,
        "parameters": {
            "m": m,
            "n": n,
            "dtype": dtype_name,
            "complex_input": complex_input,
            "provider_configs": {
                "triton": {
                    "algorithm": "global_ping_pong_radix4_fft",
                    "num_warps": triton_num_warps,
                    "num_stages": 1,
                },
                "tle": {
                    "algorithm": "register_fft" if n == FFT._FFT_REG_THRESHOLD else "shared_ping_pong_radix4_fft",
                    "num_warps": tle_num_warps,
                    "num_stages": 1,
                },
                "torch": {
                    "algorithm": "torch.fft.fft",
                    "output_preallocated": True,
                },
            },
        },
        "correct": True,
        "measurements": measurements,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", choices=("all", *SHAPES), default="all")
    parser.add_argument("--dtype", choices=("float16", "float32", "bfloat16"), default="float32")
    parser.add_argument("--complex-input", action="store_true")
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--stabilization-rounds", type=int, default=1)
    parser.add_argument("--warmup-ms", type=int, default=amd_bench.BENCH_WARMUP_MS)
    parser.add_argument("--rep-ms", type=int, default=amd_bench.BENCH_REP_MS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.rounds < 1:
        parser.error("--rounds must be at least 1")
    if args.stabilization_rounds < 0:
        parser.error("--stabilization-rounds cannot be negative")
    if args.warmup_ms < 1 or args.rep_ms < 1:
        parser.error("--warmup-ms and --rep-ms must be at least 1")

    torch.manual_seed(amd_bench.RANDOM_SEED)
    selected_shapes = SHAPES.items() if args.shape == "all" else ((args.shape, SHAPES[args.shape]), )
    results = [
        _run_shape(
            shape_name,
            shape,
            args.dtype,
            args.complex_input,
            args.rounds,
            args.stabilization_rounds,
            args.warmup_ms,
            args.rep_ms,
        )
        for shape_name, shape in selected_shapes
    ]
    report = {
        "schema_version": 2,
        "environment": amd_bench._environment(),
        "measurement_config": {
            "rounds": args.rounds,
            "stabilization_rounds_discarded": args.stabilization_rounds,
            "warmup_ms": args.warmup_ms,
            "rep_ms": args.rep_ms,
            "quantiles": list(amd_bench.QUANTILES),
            "provider_order": "balanced cycle over all six provider permutations",
            "random_seed": amd_bench.RANDOM_SEED,
            "bootstrap_samples": amd_bench.BOOTSTRAP_SAMPLES,
            "confidence_level": 0.95,
            "confidence_method": "paired bootstrap of ratio of p50 medians",
            "output_allocation": "all provider outputs and scratch preallocated",
            "plan_setup": "bit reversal, twiddles, and input conversion excluded from timing",
            "cache_policy": "triton.testing.do_bench clears L2 before every timed sample",
        },
        "results": results,
    }
    serialized = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
