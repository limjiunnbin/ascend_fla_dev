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
    parser.add_argument("--baseline", choices=("torch_npu", "fp32_path"), default="torch_npu",
                        help=("torch_npu: the same recurrence composed from native operators on the "
                              "device, which is the comparison AGENTS.md section 6 asks for. "
                              "fp32_path: this operator's own FP32 path, useful only for comparing "
                              "two dtype paths against each other."))
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


QK_EPS = 1e-6
Q_SCALE = 128 ** -0.5


def machine_witness():
    """Whether the machine was quiet, recorded with the timings.

    Absolute milliseconds from a shared box mean nothing on their own: a constant lift affects every
    round equally and a sandwich cannot see it. This records what else was running so a reader can
    judge, and so a suspicious number can be re-examined rather than inherited.
    """
    import subprocess
    witness = {}
    try:
        out = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, timeout=30).stdout
        witness["npu_smi_available"] = bool(out.strip())
        witness["npu_smi_health_lines"] = [line.strip() for line in out.splitlines()
                                           if "Ascend" in line][:8]
    except Exception as exc:                                   # no npu-smi on some A5 hosts
        witness["npu_smi_available"] = False
        witness["npu_smi_error"] = str(exc)[:120]
    try:
        procs = subprocess.run(["ps", "-eo", "cmd", "--no-headers"],
                               capture_output=True, text=True, timeout=30).stdout.splitlines()
        busy = [p.strip()[:80] for p in procs
                if "python" in p and "bench_gdn2" not in p and "defunct" not in p]
        witness["other_python_processes"] = len(busy)
        witness["other_python_sample"] = busy[:3]
    except Exception as exc:
        witness["other_python_processes"] = None
        witness["ps_error"] = str(exc)[:120]
    return witness


def torch_npu_composed(q, k, v, gate, erase_gate, w, state):
    """The same recurrence built from native operators, run on the device.

    This is the second of the two oracles AGENTS.md section 6 requires: not our compiled kernel and
    not a CPU reference, but what a caller would write with torch_npu alone. It is the honest
    baseline for "is the kernel worth having", and it is deliberately the straightforward
    token-at-a-time form, because that is what the operator replaces.
    """
    q32, k32, v32 = q.float(), k.float(), v.float()
    qn = q32 * torch.rsqrt(q32.square().sum(-1, keepdim=True) + QK_EPS) * Q_SCALE
    kn = k32 * torch.rsqrt(k32.square().sum(-1, keepdim=True) + QK_EPS)
    current = state.clone()
    outputs = []
    for index in range(q.shape[1]):
        k_t = kn[:, index]
        current = current * torch.exp(gate[:, index]).unsqueeze(-1)
        erase = torch.matmul((erase_gate[:, index] * k_t).unsqueeze(-2), current).squeeze(-2)
        delta = w[:, index] * v32[:, index] - erase
        current = current + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        outputs.append(torch.matmul(qn[:, index].unsqueeze(-2), current).squeeze(-2))
    return torch.stack(outputs, dim=1), current


def time_baseline(tensors, reps, warmup):
    for _ in range(warmup):
        torch_npu_composed(*tensors)
    torch.npu.synchronize()
    samples = []
    for _ in range(reps):
        start = time.perf_counter()
        torch_npu_composed(*tensors)
        torch.npu.synchronize()
        samples.append((time.perf_counter() - start) * 1e3)
    samples.sort()
    return {"median_ms": statistics.median(samples), "min_ms": samples[0],
            "p90_ms": samples[max(0, int(0.9 * len(samples)) - 1)], "reps": reps}


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

    use_torch_npu = args.baseline == "torch_npu"
    identity = ("the same recurrence composed from native torch_npu operators"
                if use_torch_npu else "the FP32 path of this same operator")
    plan = (("baseline", torch.float32, baseline_inputs),
            ("candidate", dtype, candidate_inputs),
            ("baseline", torch.float32, baseline_inputs))
    rounds = []
    for index, (role, round_dtype, tensors) in enumerate(plan, start=1):
        if role == "baseline" and use_torch_npu:
            row = time_baseline(tensors, args.reps, args.warmup)
        else:
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
    receipt = {"rounds": rounds, "baseline_identity": identity,
               "baseline_median_ms": baseline, "candidate_median_ms": candidate,
               "baseline_round_drift": drift,
               "ratio_candidate_over_baseline": candidate / baseline,
               "machine_witness": machine_witness(),
               "note": ("No speed target is set. baseline_round_drift catches the machine moving "
                        "during the run; it does NOT catch a constant lift applied to every round. "
                        "A BF-05 measurement on this operator read 43 ms where a rerun of the same "
                        "script on the same card read 3.3 ms, with the ratio intact, so absolute "
                        "timings need machine_witness to be believed, while ratios survive.")}
    print(f"\nbaseline median {baseline:.3f} ms (the two rounds differ by {drift * 100:.2f}%)")
    print(f"candidate median {candidate:.3f} ms   "
          f"ratio {receipt['ratio_candidate_over_baseline']:.3f}")
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(receipt, handle, indent=1, sort_keys=True)
    return receipt


if __name__ == "__main__":
    main()
