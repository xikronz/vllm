# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Phase 0a screen for Speculative Speculative Decoding (vllm-project/vllm#36037).

SSD's best case is hiding draft latency completely: step time falls from
`draft + verify` to `verify`, so the speedup upper bound is `1 + draft/verify`.
The drafter needs its own GPU(s), so throughput per GPU only improves when

    1 + draft/verify  >=  (target_gpus + draft_gpus) / target_gpus

which reduces to a threshold on the draft's share of step time:

    draft_frac  >=  draft_gpus / (target_gpus + draft_gpus)

For the paper's TP=4 target plus one draft GPU that is 20%.

The bound assumes a perfect cache (p_hit = 1) and unchanged accepted length, so
it is necessary, not sufficient: a config that fails here cannot be rescued by
tuning. A config that passes still has to survive Phase 0b.

Two input modes:

  --trace     a torch profiler trace from a real vLLM run (preferred)
  --draft-ms/--verify-ms   numbers you already have

To produce a trace:

  VLLM_CUSTOM_SCOPES_FOR_PROFILING=1 vllm serve MODEL \\
      --speculative-config '{"method":"eagle3","num_speculative_tokens":8,...}' \\
      --profiler-config.profiler=torch \\
      --profiler-config.torch_profiler_dir=/tmp/vllm_trace

The scopes `gpu_model_runner: draft` and `gpu_model_runner: forward` already
exist in vllm/v1/worker/gpu_model_runner.py; no vLLM changes are required.
"""

from __future__ import annotations

import argparse
import gzip
import json
import statistics
import sys
from bisect import bisect_left
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

DRAFT_SCOPE = "gpu_model_runner: draft"
VERIFY_SCOPE = "gpu_model_runner: forward"


@dataclass(frozen=True)
class Verdict:
    draft_frac: float
    speedup_ub: float
    per_gpu: float
    per_gpu_p: float
    threshold: float
    threshold_p: float
    label: str

    @property
    def passed(self) -> bool:
        return self.label == "PASS"


def screen(
    draft: float,
    verify: float,
    target_gpus: int = 4,
    draft_gpus: int = 1,
    p_hit: float = 0.7,
) -> Verdict:
    """Apply the break-even screen to one config.

    Reports two points. The first assumes a perfect speculation cache
    (p_hit = 1), giving the hard upper bound: below it, no amount of tuning
    helps. The second assumes only a `p_hit` fraction of steps actually hit
    the cache, so roughly `p_hit * draft` is hidden and the rest is paid at
    full cost -- a realistic figure, since the paper measures p_hit in the
    0.65-0.90 range (Fig. 3, 4).

    Args:
        draft: Time spent drafting per step, any consistent unit.
        verify: Time spent in the target forward per step, same unit.
        target_gpus: GPUs the target occupies (its TP size).
        draft_gpus: Additional GPUs the SSD drafter would occupy.
        p_hit: Pessimistic cache hit rate used for the realistic point.

    Returns:
        A Verdict carrying the draft fraction, both speedup/per-GPU figures,
        both thresholds, and a PASS/MARGINAL/FAIL label.
    """
    if draft < 0 or verify <= 0:
        raise ValueError(f"need draft >= 0 and verify > 0, got {draft} and {verify}")
    if not 0.0 < p_hit <= 1.0:
        raise ValueError(f"need 0 < p_hit <= 1, got {p_hit}")

    step = draft + verify
    draft_frac = draft / step
    hardware_ratio = (target_gpus + draft_gpus) / target_gpus

    speedup_ub = step / verify
    per_gpu = speedup_ub / hardware_ratio
    per_gpu_p = (step / (step - p_hit * draft)) / hardware_ratio

    threshold = draft_gpus / (target_gpus + draft_gpus)
    threshold_p = (hardware_ratio - 1.0) / (hardware_ratio * p_hit)

    if draft_frac < threshold:
        label = "FAIL"
    elif per_gpu_p < 1.0:
        label = "MARGINAL"
    else:
        label = "PASS"

    return Verdict(
        draft_frac, speedup_ub, per_gpu, per_gpu_p, threshold, threshold_p, label
    )


def _load_trace(path: Path) -> dict:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        return json.load(f)


def _kernel_time_by_correlation(events: list[dict]) -> dict[int, float]:
    """Map correlation id -> GPU kernel duration."""
    out: dict[int, float] = defaultdict(float)
    for e in events:
        if e.get("cat") not in ("kernel", "gpu_memcpy", "gpu_memset"):
            continue
        corr = e.get("args", {}).get("correlation")
        if corr is not None:
            out[corr] += e.get("dur", 0.0)
    return out


def _launches(events: list[dict]) -> list[tuple[float, int]]:
    """Sorted (timestamp, correlation) for host-side CUDA launches."""
    out = []
    for e in events:
        if e.get("cat") not in ("cuda_runtime", "cuda_driver"):
            continue
        corr = e.get("args", {}).get("correlation")
        if corr is not None and "ts" in e:
            out.append((e["ts"], corr))
    out.sort()
    return out


def _scopes(events: list[dict], name: str) -> list[tuple[float, float]]:
    """Sorted (start, duration) for a named user annotation."""
    out = [
        (e["ts"], e.get("dur", 0.0))
        for e in events
        if e.get("name") == name and "ts" in e and e.get("ph") == "X"
    ]
    out.sort()
    return out


def _gpu_time_in(
    window: tuple[float, float],
    launches: list[tuple[float, int]],
    kernel_time: dict[int, float],
) -> float:
    """Sum GPU time for kernels launched inside a host-side window."""
    start, dur = window
    end = start + dur
    lo = bisect_left(launches, (start,))
    total = 0.0
    for ts, corr in launches[lo:]:
        if ts > end:
            break
        total += kernel_time.get(corr, 0.0)
    return total


def analyze_trace(path: Path, use_gpu_time: bool = True) -> tuple[float, float, int]:
    """Extract mean per-iteration draft and verify time from a vLLM trace.

    Pairs each `draft` scope with the nearest preceding `forward` scope; in
    execute_model the order within one decode iteration is forward -> sample ->
    draft, so this isolates decode steps and drops prefill-only forwards.

    Returns:
        (draft_us, verify_us, num_iterations) as means over paired iterations.
    """
    trace = _load_trace(path)
    events = trace.get("traceEvents", trace if isinstance(trace, list) else [])

    drafts = _scopes(events, DRAFT_SCOPE)
    verifies = _scopes(events, VERIFY_SCOPE)
    if not drafts:
        raise SystemExit(
            f"no '{DRAFT_SCOPE}' scopes in {path}.\n"
            "Re-run vLLM with VLLM_CUSTOM_SCOPES_FOR_PROFILING=1 and a "
            "speculative-config, otherwise the scopes compile out to nullcontext."
        )
    if not verifies:
        raise SystemExit(f"no '{VERIFY_SCOPE}' scopes in {path}.")

    if use_gpu_time:
        kernel_time = _kernel_time_by_correlation(events)
        launches = _launches(events)
        if not kernel_time:
            print(
                "warning: no CUDA kernels in trace, falling back to host wall "
                "time (will understate GPU work)",
                file=sys.stderr,
            )
            use_gpu_time = False

    def measure(window: tuple[float, float]) -> float:
        if use_gpu_time:
            return _gpu_time_in(window, launches, kernel_time)
        return window[1]

    verify_starts = [v[0] for v in verifies]
    paired: list[tuple[float, float]] = []
    for d in drafts:
        idx = bisect_left(verify_starts, d[0]) - 1
        if idx < 0:
            continue
        paired.append((measure(d), measure(verifies[idx])))

    if not paired:
        raise SystemExit("found scopes but could not pair any draft with a forward")

    draft_us = statistics.median(p[0] for p in paired)
    verify_us = statistics.median(p[1] for p in paired)
    return draft_us, verify_us, len(paired)


# Reference data from EanWang211123 in vllm-project/vllm#36037 (SGLang,
# Qwen3.5-27B, H100 TP=4). Columns: label, draft_ms, step_total_ms.
THREAD_DATA = [
    ("MTP K=16, b=1", 9.56, 22.16),
    ("MTP K=12, b=1", 7.03, 19.01),
    ("MTP K=8,  b=1", 4.64, 16.18),
    ("MTP K=4,  b=1", 2.06, 13.11),
    ("DFlash K=16, b=1", 0.90, 12.36),
    ("MTP K=16, b=8", 10.61, 27.25),
    ("MTP K=8,  b=8", 5.08, 18.99),
    ("DFlash K=16, b=8", 1.12, 16.16),
    ("MTP K=16, b=32", 12.45, 48.26),
    ("MTP K=8,  b=32", 5.96, 29.03),
    ("DFlash K=16, b=32", 1.74, 33.00),
]


def _print_row(label: str, v: Verdict) -> None:
    print(
        f"  {label:<20} {v.draft_frac:>6.1%}  {v.per_gpu:>7.2f}x  "
        f"{v.per_gpu_p:>7.2f}x   {v.label}"
    )


def _print_header(target_gpus: int, draft_gpus: int, v: Verdict, p_hit: float) -> None:
    print(
        f"\n  target TP={target_gpus} + {draft_gpus} draft GPU  ->  "
        f"break-even draft share {v.threshold:.0%} at p_hit=1, "
        f"{v.threshold_p:.0%} at p_hit={p_hit:g}\n"
    )
    print(
        f"  {'config':<20} {'draft%':>6}  {'perGPU@1':>8}  "
        f"{'perGPU@' + format(p_hit, 'g'):>8}   verdict"
    )
    print(f"  {'-' * 20} {'-' * 6}  {'-' * 8}  {'-' * 8}   {'-' * 8}")


def run_reference(target_gpus: int, draft_gpus: int, p_hit: float) -> None:
    rows = [
        (label, screen(draft, total - draft, target_gpus, draft_gpus, p_hit))
        for label, draft, total in THREAD_DATA
    ]
    print("Reference data from issue #36037 (SGLang, Qwen3.5-27B, H100 TP=4)")
    _print_header(target_gpus, draft_gpus, rows[0][1], p_hit)
    for label, v in rows:
        _print_row(label, v)
    print(
        "\n  perGPU@1 is the hard upper bound; perGPU@"
        f"{p_hit:g} prices in cache misses.\n"
        "  Rows near break-even at b=32 still fail in practice -- a miss on any\n"
        "  sequence stalls the whole batch, so p_hit^b collapses (Cor. 16).\n"
    )


def main() -> int:
    p = argparse.ArgumentParser(
        description="Phase 0a break-even screen for SSD in vLLM.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--trace", type=Path, help="torch profiler trace (.json/.json.gz)")
    src.add_argument("--draft-ms", type=float, help="measured draft time per step")
    src.add_argument(
        "--reference",
        action="store_true",
        help="screen the measurements already posted in issue #36037",
    )
    p.add_argument("--verify-ms", type=float, help="target forward time per step")
    p.add_argument("--target-gpus", type=int, default=4, help="target TP size")
    p.add_argument("--draft-gpus", type=int, default=1, help="GPUs added for drafting")
    p.add_argument(
        "--host-time",
        action="store_true",
        help="use host wall time instead of attributed GPU kernel time",
    )
    p.add_argument(
        "--p-hit",
        type=float,
        default=0.7,
        help="pessimistic cache hit rate for the realistic point (default 0.7)",
    )
    p.add_argument("--label", default="measured", help="row label for trace mode")
    args = p.parse_args()

    if args.reference:
        run_reference(args.target_gpus, args.draft_gpus, args.p_hit)
        return 0

    if args.trace:
        draft, verify, n = analyze_trace(args.trace, use_gpu_time=not args.host_time)
        basis = "host wall" if args.host_time else "GPU kernel"
        print(
            f"\n{args.trace.name}: {n} decode iterations, {basis} time\n"
            f"  draft  {draft / 1000:.2f} ms\n"
            f"  verify {verify / 1000:.2f} ms"
        )
    else:
        if args.verify_ms is None:
            p.error("--draft-ms requires --verify-ms")
        draft, verify = args.draft_ms, args.verify_ms

    v = screen(draft, verify, args.target_gpus, args.draft_gpus, args.p_hit)
    _print_header(args.target_gpus, args.draft_gpus, v, args.p_hit)
    _print_row(args.label, v)

    if v.label == "FAIL":
        detail = "below the p_hit=1 bound; no tuning can repay the extra GPU"
    elif v.label == "MARGINAL":
        detail = "clears the bound but not at realistic p_hit; measure p_hit first"
    else:
        detail = "clears the screen; proceed to Phase 0b"
    print(f"\n  {v.label}: {detail}\n")
    return 0 if v.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
