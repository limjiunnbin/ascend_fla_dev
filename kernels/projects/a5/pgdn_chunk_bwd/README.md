# PGDN FP32 chunk backward (PK-05)

**Full requalification pending:** the original dense-at-clamp failure is
retained below. The repair satisfies the three original failure cases and the
first full T4096 case; the complete original and supplemental matrices plus
new same-card measurements are still required. PR #134 remains a draft.

This standalone A5/CCE unit computes `dq, dk, dv, dg_atk, dg, dbeta_atk,
dbeta` for the loss `<do,o> + <dht,final_state> + <dA_T,final_A_state>`.
At least one cotangent must be supplied. The public adapter is
`ascend_fla.ops.pgdn_chunk_bwd.chunk_pgdn_bwd`; no autograd wiring is provided.

The frozen domain is B=1, T a positive multiple of64 no larger than4096,
K=V=128, positive H/HV with HV divisible by H, FP32 contiguous inputs,
zero initial states (only `None`), token-major layout, mandatory FP32 q/k
normalization with norm clamp1e-12, scale128**-.5, x1.5, ATK eps1e-6 and
log_atk_scale=-.2. Gates must be finite/nonpositive with finite chunk sums;
beta and beta_atk belong to[0,1]. All inputs and cotangents must be finite.
BF16 is explicitly rejected. Near-zero q/k rows are accepted.

The independent implementation has no runtime import of the forward, GDN or
GDN-2 units. Host production work is input validation, allocation and pointer
binding. All normalization, state replay, grouped reductions and derivatives
execute in the four task-owned device kernels.

| Stage | Work | UB bytes / buffers |
| --- | --- | --- |
| pgdn_bwd_atk | Normalize q/k and save every per-H ATK state | 140032 / 10 |
| pgdn_bwd_checkpoints | Save the main state before each64-token chunk | 67136 / 6 |
| pgdn_bwd_reverse | Replay one chunk, then reverse its main recurrence perHV | 135808 / 15 |
| pgdn_bwd_atk_reverse | Sum consecutive value-head contributions, reverse ATK and normalization perH | 4832 / 16 |

The main reverse uses `k_read` in the residual and read-key correction,
`k_write` in the state update and write-key derivative. The ATK terminal
cotangent is seeded once perH, after summing all its value-head contributions.
Each owner traverses the complete time axis. GM replay publication uses an
explicit MTE3-to-MTE2 event; all tape readers drain before slot reuse.
All20 named stage outputs are independently observable in the native harness.

## Numerical contract

A is literal FLA PGDN naive at pin
`e52dbc0ea19d3a40d7ab7f9eed855d2b473994d2`, differentiated by CPU FP32 autograd.
B is a separate analytical reverse with chunk checkpoints; it imports neither
FLA nor the device implementation. Both passed the six small FP64 gradchecks
away from the normalization kink. The external literal file is selected by
`FLA_PGDN_NAIVE` and hash-checked. Production execution does not depend on it.

Classification is per gradient element, generated from independent FP64
analytical truth and local cancellation scale before examining the candidate.
Algebraic zeros and elements with `abs(reference64)<=64*2^-23*local_scale`
are disclosure-only. Each ordinary subset separately requires relativeL2<=1e-4
against both A and B. The FP32 normalization clamp branch is retained when
lifting the reference at the exact boundary. Actual native branch bits are
also compared, independently of the numerical classifier.

Disclosure-only elements have no pass threshold. Complete records include
position, candidate, canonical/raw FP64 reference, local scale, threshold,
classification reason, absolute output and contribution relative to the full
FP64 gradient norm. Zero denominators remain undefined/infinite as appropriate;
no denominator epsilon is added. Complete indexed JSONL uses `start/count`
only for adjacent, bit-identical records and therefore restores every element,
including signed zeros. The concrete formulas are in
[the range study](../../../../docs/research/pgdn_chunk_bwd_gate_range.md).

**The overall numerical status is “numerical disclosure complete, nonPASS”
whenever disclosures exist.** Ordinary-budget satisfaction is reported
separately. `run.py` retains the canonical raw comparison as a diagnostic; its
reference stage only checks schema/finiteness and is not classified acceptance.

## Reproduction

Use the accepted Python environment and Ascriptor0.1.0 source selected by
`compatibility.json` (library90cfcdc / kernelsb3b3f9c). Machine selection stays
in ignored external configuration via `ASCRIPTOR_MACHINE_SPECS` and
`ASCRIPTOR_BOARDS`. Native runs require a fresh healthy/idle-device check,
the shared lock, isolated outputs, and separate processes/builds for bd1/bd2.
Run the complete T4096 workload first. Simulators are diagnostic tools only.

From this unit directory, with the selected library and repository importable:

```bash
python -m ref.gradcheck --output <fresh-gradcheck-json>
python -m ref.calibrate --output <fresh-calibration-directory> --workers 4
python run.py reference --case full_r1_m7_bd2 --output <fresh-reference-directory>
python benchmark.py --block-dim 1 --output <fresh-bd1-directory> --oracle-workers 4
python benchmark.py --block-dim 2 --output <fresh-bd2-directory> --oracle-workers 4
python compare_runs.py <bd1-qualification.json> <bd2-qualification.json> --output <comparison.json>
python measure.py --block-dim 2 --output <fresh-measurement-directory>
```

The public benchmark exercises actual `None` cotangents, poisoned outputs,
input immutability, independent leaf launches and composition, public output
bytes, internal stage values, clamp branch bits and classified A/B comparison.
The bd comparison requires the complete256-case grid and identical inputs,
20 stage hashes and seven public output hashes.

Native evidence in this task uses the `inprocess` launcher inside the accepted
Docker environment. The explicit CPU-input `aclnn` and `board` transport paths
have host validation coverage but remain untested on hardware for this task.
The contract's `board` stage means device qualification; it does not imply that
the identically named transport launcher was exercised.

Measurement uses T1024/4096, B1/H=HV8, bd2, 10warmups, 50samples and three
baseline/candidate/baseline rounds. Each timed call synchronizes before/after;
CPU references finish beforehand. The baseline reuses this candidate's saved
normalization/ATK tapes and main checkpoints, excludes stage1/2 device work,
and retains public validation and all allocations (including unused forward
outputs). It is an adjoint cost baseline, not another full backward or a
CUDA/Triton comparison. Numerical checks and disclosures bracket every round.

## Evidence status

The frozen256-case CPU calibration satisfies every ordinary A/B budget; maximum
relativeL2 is2.9902082636512436e-6. Both references passed six FP64 gradchecks.
`evidence/classification-freeze.json` pins the six reference-source hashes.
The complete512 A/B disclosure lists are plain JSONL, with content hashes.

Actual A5/CCE execution completed256 cases at each of bd1 and bd2, in separate
processes/builds. All20 returned stages and seven public gradients are byte
identical for identical runtime-generated inputs. Ordinary comparisons against
both A and B satisfy1e-4; their maximum is3.872425474189693e-6. Each matrix has
66,581,570 public disclosure-only elements, with every position/value retained.
The overall numerical status is **numerical disclosure complete, nonPASS**;
`contract.json` therefore does not mark its numerical board stage passed.

Start review with `evidence/summary-matrix-bd1-v2.json`,
`evidence/summary-matrix-bd2-v3.json` and `evidence/block-dim-comparison.json`.
Full per-case records, disclosure JSONL and sanitized execution/source receipts
are under `evidence/native/`. `evidence/source-consistency.json` links the
unchanged production and frozen-reference sources to each executed snapshot.
Raw tensors and logs containing machine bindings stay in private ignored storage.

Three same-card sandwiches completed for each of T1024/4096, with10 warmups and
50 synchronized samples per leg. The exact baseline scope is stated above;
`evidence/native/timing-bd2-v3/measurement.json` retains every sample and numerical
checks/disclosures before and after each round. The range study gives the medians.
There is no separate full-backward backend or CUDA/Triton speedup claim.

Host regression:1354 passed,10 skipped,5 existing CPU-stub warnings. The actual
Docker environment's CPU task regressions:115 passed. Matrix/PM-board checks
passed. Vendor warnings were investigated and retained without suppression in
`evidence/vendor-warning-assessment.json`. No simulator, pipesim or weight
validation is claimed. Machine configuration remains external and ignored.
