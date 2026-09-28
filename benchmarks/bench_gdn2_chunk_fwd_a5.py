"""Same-card three-round sandwich for the GDN-2 chunk forward on A5.

Rounds run baseline / candidate / baseline, so drift in the machine shows up as disagreement between
the two baseline rounds rather than as a result. No speed target is set: the numbers are reported as
measured and the baseline's identity is printed beside them.

Two rules from AGENTS.md section 6 are structural here, not optional:

* One operator name, one process, one build. CANN resolves a custom operator by name and the first
  vendor tree on ASCEND_CUSTOM_OPP_PATH wins, resolved once per process, so a second build of the
  same name is ignored silently. ``prepare()`` compiles every operator set this process will touch
  before anything executes, and a block_dim sweep must use one process per block_dim.
* Profile before believing any performance inference. This times the public entry as a caller uses
  it, after warmup, and reports median with min and p90 so one outlier cannot look like a trend.

Usage, inside the task container while holding the card's lock::

    ASCEND_RT_VISIBLE_DEVICES=<card> python benchmarks/bench_gdn2_chunk_fwd_a5.py \
        --dtype float32 --seq-len 1024 --heads 16 --block-dim 8 --reps 20

The dtype is explicit and printed on every row. Since BF-05 the entry dispatches FP32 and BF16 to
two different operator sets with different performance, so an unlabelled number means nothing.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import torch

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="float32",
                        help="public q/k/v dtype; FP32 and BF16 are separate operator sets")
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--heads", type=int, default=16, choices=(1, 16),
                        help="the declared domain is H in {1, 16}")
    parser.add_argument("--block-dim", type=int, default=8, choices=(1, 2, 4, 8))
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--json", metavar="PATH", help="also write the receipt here")
    return parser.parse_args(argv)


def make_inputs(dtype, seq_len, heads, seed, device="npu"):
    """The declared domain: B=1, T=1..4096, H in {1,16}, K=V=128; gates and state stay FP32."""
    generator = torch.Generator().manual_seed(seed)
    shape = (1, seq_len, heads, 128)

    def randn(scale):
        return torch.randn(shape, generator=generator) * scale

    q, k, v = (randn(0.5).to(dtype).to(device) for _ in range(3))
    gate = (-torch.rand(shape, generator=generator) * 3.0).to(device)   # non-positive log decay
    erase_gate = torch.rand(shape, generator=generator).to(device)      # within [0, 2]
    w = torch.rand(shape, generator=generator).to(device)               # within [0, 1]
    state = (torch.randn(1, heads, 128, 128, generator=generator) * 0.1).to(device)
    return q, k, v, gate, erase_gate, w, state


def time_round(chunk_gdn2, tensors, block_dim, reps, warmup):
    q, k, v, gate, erase_gate, w, state = tensors
    for _ in range(warmup):
        chunk_gdn2(q, k, v, gate, erase_gate, w, initial_state=state, block_dim=block_dim)
    torch.npu.synchronize()
    samples = []
    for _ in range(reps):
        start = time.perf_counter()
        chunk_gdn2(q, k, v, gate, erase_gate, w, initial_state=state, block_dim=block_dim)
        torch.npu.synchronize()
        samples.append((time.perf_counter() - start) * 1e3)
    samples.sort()
    return {"median_ms": statistics.median(samples), "min_ms": samples[0],
            "p90_ms": samples[max(0, int(0.9 * len(samples)) - 1)], "reps": reps}


def main(argv=None):
    args = parse_args(argv)
    import torch_npu  # noqa: F401  registers the npu device type
    from ascend_fla.ops.gdn2_chunk_fwd import chunk_gdn2, prepare

    # Compile every operator set before the first execution: CANN reads ASCEND_CUSTOM_OPP_PATH once,
    # at first resolution, and never sees a vendor tree appended after that.
    prepare(block_dim=args.block_dim)

    dtype = DTYPES[args.dtype]
    baseline_inputs = make_inputs(torch.float32, args.seq_len, args.heads, args.seed)
    candidate_inputs = (baseline_inputs if dtype is torch.float32
                        else make_inputs(dtype, args.seq_len, args.heads, args.seed))

    plan = (("baseline", torch.float32, baseline_inputs),
            ("candidate", dtype, candidate_inputs),
            ("baseline", torch.float32, baseline_inputs))
    rounds = []
    for index, (role, round_dtype, tensors) in enumerate(plan, start=1):
        row = time_round(chunk_gdn2, tensors, args.block_dim, args.reps, args.warmup)
        row.update(round=index, role=role, dtype=str(round_dtype).replace("torch.", ""),
                   seq_len=args.seq_len, heads=args.heads, block_dim=args.block_dim)
        rounds.append(row)
        print(f"round {index} {role:9s} {row['dtype']:8s} median {row['median_ms']:8.3f} ms  "
              f"min {row['min_ms']:8.3f}  p90 {row['p90_ms']:8.3f}", flush=True)

    baselines = [row["median_ms"] for row in rounds if row["role"] == "baseline"]
    candidate = [row["median_ms"] for row in rounds if row["role"] == "candidate"][0]
    baseline = statistics.median(baselines)
    drift = abs(baselines[0] - baselines[1]) / baseline
    receipt = {"rounds": rounds,
               "baseline_identity": "the FP32 path of this same operator",
               "baseline_median_ms": baseline, "candidate_median_ms": candidate,
               "baseline_round_drift": drift,
               "ratio_candidate_over_baseline": candidate / baseline,
               "note": ("No speed target is set. If baseline_round_drift is comparable to the "
                        "candidate/baseline difference then the machine moved and the comparison "
                        "says nothing.")}
    print(f"\nbaseline median {baseline:.3f} ms (the two rounds differ by {drift * 100:.2f}%)")
    print(f"candidate median {candidate:.3f} ms   "
          f"ratio {receipt['ratio_candidate_over_baseline']:.3f}")
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(receipt, handle, indent=1, sort_keys=True)
    return receipt


if __name__ == "__main__":
    main()
