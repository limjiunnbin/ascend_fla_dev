# GD2-02 measured results

All on one A5 host (Ascend950PR, CANN 9.2.0, op packages including ascend950, compiler
2026-08-04). The other A5 host has a different compiler build and its results are not carried here.

## Op-package survey (AGENTS.md section 5)

Both A5 hosts carry `ascend950` beside `ascend910b` and `ascend910_93`, so torch_npu's compute
operators are available and the packageless workaround does not apply.

## Contract cases on device: 12/12

Worst between-reference relative L2 is 3.349e-06 on `o` and 5.677e-06 on `final_state`, roughly
three orders inside the 1e-3 budget. The error tracks head count, not sequence length: H=1 sits near
1e-7 and H=16 near 1e-6, and T=4096 is no worse than T=64.

## block_dim 1/2/4/8: bitwise equal

| case | T | o | final_state |
|---|---|---|---|
| t64_h16_real | 64 | 8cf24e8dfee90740 | 60a928907bdbf4da |
| t65_h16_tail | 65 | eee82750891b055a | 0336c852d8354b73 |
| t192_h16_odd | 192 | 090faf0956eec21f | a1020c6bf09bc473 |

## Performance against torch_npu

T=256, H=16, bd=8. Baseline is the same recurrence composed from native torch_npu operators, which
is what a caller writes without this kernel.

| round | identity | median |
|---|---|---|
| 1 | torch_npu composition | 68.560 ms |
| 2 | compiled kernel | 1.090 ms |
| 3 | torch_npu composition | 64.526 ms |

Baseline median 66.543 ms, candidate 1.090 ms, ratio 0.016: about 61x. The baseline is the naive
token-at-a-time composition, so this is not "61x any torch implementation"; a hand-written chunked
torch version would be considerably faster than 66.5 ms.

The two baseline rounds differ by 6.06%, which a 61x gap tolerates. Absolute timings now carry a
machine witness, because a sandwich sees the machine move during a run but is blind to a constant
lift across all of it: a BF-05 measurement read 43 ms where a rerun of the same script on the same
card read 3.3 ms with the ratio intact.

## 95B checkpoint, FP32 logits: inside budget

Checkpoint sha256 4ac729c6...b2df6d at 17,401,727,659 bytes, matching models.json and GD2-01's pin.
Both backends run clean; logits dumped one backend per process and compared off-device.

| quantity | shape | between-backend relative L2 | max_abs | budget |
|---|---|---|---|---|
| logits | (1, 8, 32000) | 4.094682e-05 | 9.155273e-04 | 1e-3 |

argmax matches on all eight tokens.

## Not established

The cache comparison and the BF16 round are **not done**, so the 95B item does not pass: its budget
covers logits and cache, and only the FP32 logits half is measured. The `cache_check` field inside
each backend's own receipt is that backend's internal prefill/decode consistency check, not a
comparison between backends, and must not be read as one.

Work stopped when the host holding the checkpoint lost its NPU driver after a reboot, and the
checkpoint could not be moved to the healthy host: the upload path measured under 0.1 MB/s, the
healthy host has no internet, and the two hosts have no key between them.
