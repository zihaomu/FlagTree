"""Reproducible AMD benchmark for the TLE TopK tutorial."""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
from pathlib import Path
from types import ModuleType
from typing import Callable

import torch

import triton

import benchmark_amd as amd_bench


REPO_ROOT = Path(__file__).resolve().parents[3]
TOPK_TUTORIAL_PATH = REPO_ROOT / "python" / "tutorials" / "tle" / "03-topk.py"
SHAPES = {
    "short-small-k": {"m": 64, "n": 128, "k": 8, "row_class": "short", "k_class": "small"},
    "short-medium-k": {"m": 64, "n": 1024, "k": 32, "row_class": "short", "k_class": "medium"},
    "long-medium-k": {"m": 64, "n": 8192, "k": 128, "row_class": "long", "k_class": "medium"},
    "long-large-k": {"m": 128, "n": 32768, "k": 256, "row_class": "long", "k_class": "large"},
}


def _load_topk_tutorial() -> ModuleType:
    spec = importlib.util.spec_from_file_location("tle_topk_tutorial", TOPK_TUTORIAL_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load TopK tutorial from {TOPK_TUTORIAL_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TOPK = _load_topk_tutorial()


def _torch_dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }[name]


def _num_warps(block_n: int) -> int:
    if block_n <= 64:
        return 2
    if block_n <= 128:
        return 4
    return 8


def _provider_configs(n: int) -> dict[str, dict[str, object]]:
    block_n = min(max(32, triton.next_power_of_2(n)), 1024)
    return {
        "radix": {
            "algorithm": "tle_shared_memory_radix_select",
            "block_n": block_n,
            "radix_bits": 4,
            "num_warps": _num_warps(block_n),
            "num_stages": 1,
        },
        "triton": {
            "algorithm": "triton_streaming_topk",
            "block_n": block_n,
            "num_warps": _num_warps(block_n),
            "num_stages": 1,
        },
        "torch": {"algorithm": "torch.topk", "sorted": False},
    }


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
    warmup_ms: int,
    rep_ms: int,
) -> dict[str, object]:
    samples: dict[str, list[dict[str, float]]] = {provider: [] for provider in launches}
    provider_order_by_round: list[list[str]] = []

    for launch in launches.values():
        launch()
    torch.cuda.synchronize()

    rng = random.Random(amd_bench.RANDOM_SEED)
    for _ in range(rounds):
        order = list(launches)
        rng.shuffle(order)
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
        "provider_order_by_round": provider_order_by_round,
        "radix_vs_triton": _comparison(
            samples["triton"],
            samples["radix"],
            summaries["triton"],
            summaries["radix"],
        ),
        "radix_vs_torch": _comparison(
            samples["torch"],
            samples["radix"],
            summaries["torch"],
            summaries["radix"],
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
    shape: dict[str, object],
    dtype_name: str,
    rounds: int,
    warmup_ms: int,
    rep_ms: int,
) -> dict[str, object]:
    m = int(shape["m"])
    n = int(shape["n"])
    k = int(shape["k"])
    dtype = _torch_dtype(dtype_name)
    device = triton.runtime.driver.active.get_active_torch_device()
    x = torch.rand((m, n), device=device, dtype=dtype)

    radix_values = torch.empty((m, k), device=device, dtype=dtype)
    radix_indices = torch.empty((m, k), device=device, dtype=torch.int32)
    triton_values = torch.empty_like(radix_values)
    triton_indices = torch.empty_like(radix_indices)
    torch_values = torch.empty_like(radix_values)
    torch_indices = torch.empty((m, k), device=device, dtype=torch.int64)

    launches = {
        "radix": lambda: TOPK.triton_radix_topk(
            x,
            k,
            out_vals=radix_values,
            out_idx=radix_indices,
        ),
        "triton": lambda: TOPK.triton_topk(
            x,
            k,
            out_vals=triton_values,
            out_idx=triton_indices,
        ),
        "torch": lambda: torch.topk(
            x,
            k,
            dim=1,
            sorted=False,
            out=(torch_values, torch_indices),
        ),
    }

    for launch in launches.values():
        launch()
    torch.cuda.synchronize()
    expected_values = torch.sort(torch_values, dim=1, descending=True).values
    for values, indices in (
        (radix_values, radix_indices),
        (triton_values, triton_indices),
    ):
        actual_values = torch.sort(values, dim=1, descending=True).values
        torch.testing.assert_close(actual_values, expected_values, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(x.gather(1, indices.to(torch.int64)), values, rtol=1e-3, atol=1e-3)

    return {
        "case": "topk",
        "shape_name": shape_name,
        "parameters": {
            "m": m,
            "n": n,
            "k": k,
            "dtype": dtype_name,
            "row_class": shape["row_class"],
            "k_class": shape["k_class"],
            "provider_configs": _provider_configs(n),
        },
        "correct": True,
        "measurements": _measure_providers(launches, rounds, warmup_ms, rep_ms),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", choices=("all", *SHAPES), default="all")
    parser.add_argument("--dtype", choices=("float16", "float32", "bfloat16"), default="float16")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--warmup-ms", type=int, default=amd_bench.BENCH_WARMUP_MS)
    parser.add_argument("--rep-ms", type=int, default=amd_bench.BENCH_REP_MS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.rounds < 1:
        parser.error("--rounds must be at least 1")
    if args.warmup_ms < 1 or args.rep_ms < 1:
        parser.error("--warmup-ms and --rep-ms must be at least 1")

    torch.manual_seed(amd_bench.RANDOM_SEED)
    selected_shapes = SHAPES.items() if args.shape == "all" else ((args.shape, SHAPES[args.shape]), )
    results = [
        _run_shape(
            shape_name,
            shape,
            args.dtype,
            args.rounds,
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
            "warmup_ms": args.warmup_ms,
            "rep_ms": args.rep_ms,
            "quantiles": list(amd_bench.QUANTILES),
            "provider_order": "randomized per round with fixed seed",
            "random_seed": amd_bench.RANDOM_SEED,
            "bootstrap_samples": amd_bench.BOOTSTRAP_SAMPLES,
            "confidence_level": 0.95,
            "confidence_method": "paired bootstrap of ratio of p50 medians",
            "output_allocation": "preallocated for all providers",
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
