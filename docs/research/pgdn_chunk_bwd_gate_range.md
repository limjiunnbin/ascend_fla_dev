# PGDN FP32 backward: ABI, classification and range contract

PK-05 implements a standalone A5 backward entry; forward dispatch and autograd
wiring are outside this task. The source authority is FLA
`e52dbc0ea19d3a40d7ab7f9eed855d2b473994d2`,
`fla/ops/precond_gated_delta_rule/naive.py`, SHA256
`3baa67a5f35dc7230698e3f1761ec8675131318c15d4a27ed7f2fce11e84b5e8`.
The literal FP32 oracle is never modified. Native qualification is complete with numerical disclosures (nonPASS).

## Frozen public ABI

`chunk_pgdn_bwd(q,k,v,g_atk,g,beta_atk,beta,do=None,dht=None,dA_T=None,...)`
returns `(dq,dk,dv,dg_atk,dg,dbeta_atk,dbeta)`, in differentiated-input order.
Every input, cotangent and gradient is contiguous FP32 on one device. BF16 is
explicitly rejected; there is no host widening or fallback.

| Item | Supported value |
|---|---|
| Shape | B=1; positive T divisible by64, T<=4096; K=V=128; H>0, HV>0, HV%H=0 |
| q/k, v/do | [B,T,H,K], [B,T,HV,V] |
| g_atk/beta_atk, g/beta | [B,T,H], [B,T,HV] |
| dht, dA_T | [B,HV,K,V], [B,H,K] |
| Cotangents | All seven nonempty subsets of do/dht/dA_T; absent means zero |
| Initial states | Both None; zero main and ATK states; no initial-state gradients |
| Constants | scale=128**(-.5), x=1.5, eps_atk=1e-6, center=-.2 |
| Normalization | Mandatory FP32 naive x/max(norm(x),1e-12), including zero/near-zero rows |
| Layout | Token-major inputs, K-major main state; no varlen, CP or transpose mode |
| Device | A5 CCE, block_dim1/2, explicit inprocess/aclnn/board launchers |
| Values | Finite inputs/cotangents; g/g_atk<=0 and finite FP32 chunk sums; beta/beta_atk in[0,1] |

Value validation synchronizes NPU predicates and belongs to public API timing.
The host only validates, allocates, binds and launches. Normalization, state
preparation, grouping and absent-cotangent initialization belong to kernels.

## Independent adjoint and local scales

For one token and value head, let `p=scale*normalized(q)`, `u=normalized(k)`,
`w=u*M`, `a=exp(g)`, `D=a*Sprev`, `r=v-u^T D`, `z=beta*r`, `S=D+w*z^T`.
For incoming state adjoint and output cotangent, `G=dS+p*do^T`:

- `dp=S*do`, `dw=G*z`, `dz=G^T*w`, `dr=beta*dz`.
- `du_read=-D*dr`, `dv=dr`, `dbeta=dot(dz,r)`.
- `GD=G-u*dr^T`, `dg=sum(GD*D)`, `dSprev=a*GD`.

The read key `u` remains in the correction; the write key `w` remains in `dz`.
Consecutive value-head contributions are summed in ascending order before the
shared ATK adjoint. With `s=log(A+eps_atk)-center` and
`M=exp(-log(x)*s/(1+abs(s)))`, its injected adjoint is
`dM*(-log(x)*M)/((A+eps_atk)*(1+abs(s))^2)`, where `dM=u*sum_group(dw)`.
After adding it to incoming `Z`,
`dg_atk=sum(Z*exp(g_atk)*Aprev)`, `dbeta_atk=sum(Z*u^2)`,
`du=sum_group(du_read)+M*sum_group(dw)+2*beta_atk*u*Z` and
`Zprev=exp(g_atk)*Z`. Final `dA_T` is seeded once per key head.

The following local scales are evaluated in the independent FP64 analytical
reference, without any candidate output. Each sum uses the displayed terms,
not a global maximum or a bound propagated from unrelated output coordinates.

| Gradient element | Local scale L_i |
|---|---|
| dq/dk, selected norm branch | `(|h_i| + |y_i*dot(y,h)|) / norm(x)`; h is its complete grouped pre-normalization cotangent, y=x/norm(x) |
| dq/dk, below clamp | `|h_i| / eps_norm` |
| dv_j | `sum_k |beta * G_kj * w_k|` |
| dbeta | `sum_j |dz_j * r_j|` |
| dg | `sum_kj |GD_kj * D_kj|` |
| dg_atk | `sum_k |Z_k * exp(g_atk) * Aprev_k|` |
| dbeta_atk | `sum_k |Z_k * u_k^2|` |

For q, h already includes scale exactly once. For k, h includes both read/write
paths and ATK. These formulas and the ascending group order are frozen in
`ref/reference.py`; `ref/classification.py` applies the following rule.

## Element classification and evidence

The executable interpretation is approved by
[PM NOTE5842101164](https://github.com/ddddwee1/ascend_fla_dev/issues/93#issuecomment-5842101164).
Each element with algebraically zero derivative, or
`abs(reference64_i) <= 64 * 2^-23 * L_i`, is disclosure-only. All others are
ordinary. Algebraic reason codes cover normalization's axis radial nullspace,
zero initial-state gate derivatives, missing main cotangents and missing output
cotangents. Other computed zeros fall under the explicit local-threshold rule;
a floating-point zero alone is not labelled an algebraic proof.

For the classifier, the active normalization branch is selected using the
literal FP32 `norm>=1e-12`. FP64 arithmetic evaluates that selected branch,
using norm64 on the active side and the lifted FP32 epsilon on the clamped side.
It does not select the branch again in FP64. In particular the stored FP32
`1e-12` is slightly below decimal FP64 `1e-12`, but its literal FP32 branch is
active. Known algebraic zeros are canonicalized to zero with their raw FP64
residual retained. FP64 mathematical gradcheck separately uses the all-FP64
formula away from the kink; it does not prove a derivative by crossing the kink.

For each gradient, ordinary elements keep relative L2<=1e-4 against both
literal FP32 A and independent FP32 B. The same mask applies to both. Empty
ordinary subsets are identified by their zero count. Raw whole-gradient
relative L2/max_abs remain available, including their original failures.

Disclosure has no pass threshold. For every disclosed element the evidence
records position, candidate value, FP64 reference and raw reference, local scale,
threshold, reason, absolute candidate value, absolute error and
`absolute_error/norm(entire_FP64_reference_gradient)`. A zero global denominator
is recorded as undefined for zero residual or infinity for nonzero residual;
no denominator epsilon is inserted. Presence of disclosure is reported as
**numerical disclosure complete, nonPASS**, separately from the ordinary budget.

The JSONL disclosure format is lossless: each gradient header gives shape and
row-major ordering; each `start,count` record supplies the exact common payload
for every index in that interval. It merges only adjacent identical payloads,
including identical floating-point bits and reason. It is not sampled evidence.
Signed zero, complete coordinate coverage and zero-denominator handling have
explicit regression checks. Receipts carry record count, byte size and SHA256.

## Range analysis and validation boundary

Real-arithmetic normalized key norm is at most1. With zero initial ATK state,
nonpositive gates and beta_atk in[0,1], `0<=A_t<=t<=4096`. Thus A+eps_atk is
positive, M stays within `(2/3,1.5)`, and the squash derivative has no singularity
at s=0. Main/ATK reverse only multiply by nonpositive-gate exponentials; they
never invert a decay, beta or a rank-one state update. The normalization
Jacobian norm is bounded by1e12 in real arithmetic and can have an exactly zero
radial component. This bound explains large valid derivatives and cancellation;
it does not license clipping them or excluding small input norms.

These facts are not a proof that every combination of arbitrary finite FP32
magnitudes yields finite states/gradients. The complete measured range and
native evidence must be reported separately before declaring acceptance.

Current qualification: six independent A-lift/B-analytical FP64 gradchecks
passed (ratios1/2; normal, below-clamp and above-clamp). Complete256-case classified CPU calibration satisfies all ordinary A/B budgets;
maximum relativeL2 is2.9902082636512436e-6 (c3_h1_hv2_m7, dg_atk, B_against_A).
All six frozen reference/classifier source hashes are unchanged. Complete512
A/B disclosure lists are retained as plain JSONL with content hashes in the unit
evidence directory; overall numerical status remains nonPASS.

Four CCE stages emitted for each of bd1/bd2 and static balance checks report
no problems. Actual UB allocations are139808/67136/135808/4832bytes. Native results and observed whole-chain ranges are reported below, separately
from these CPU/emission results. No simulator, CUDA/Triton or weight validation
is claimed.

The separate FP32 stage reference measured seven cases: full T4096 equal-head
and ratio8, full gate-zero/weak-decay, and the three axis normalization clamp
cases, each with all three cotangents. These are CPU observations, not device
ranges or bounds for arbitrary input magnitude. Every observed intermediate
was finite; the complete per-case table is `evidence/cpu-ranges.json`.

| Observed quantity | Range across those seven cases |
|---|---|
| Raw q norm | 4.99999998e-13 to0.71367657 |
| ATK A | 0 to19.513712 |
| Multiplier M | 0.73472625 to1.45895875 |
| Decayed main state D | -0.17093094 to0.16487035 |
| Residual v-u^T D | -0.28586960 to0.27640030 |
| Main adjoint G including output cotangent | -0.44471529 to0.47014689 |
| ATK adjoint Z including write-key injection | -0.27805093 to0.36575362 |
| Query cotangent before normalization | -0.01474613 to0.01592194 |
| Key cotangent before normalization | -0.30100784 to0.23552959 |
| Reciprocal clamped q norm | 1.40119493 to9.99999995904e11 |
| Final dq absolute maximum | 6.938435584e9 |
| Final dk absolute maximum | 2.35529584640e11 |

The large finite q/k derivatives arise from the supported clamped normalization
region. They are retained without clipping; the ordinary/disclosure rule is
unchanged. Native stage ranges are reported separately from actual returned
tensors in each device case's report.

Input receipts distinguish environments: the host calibration used Torch2.10,
and actual device qualification generates its inputs and both CPU references
inside the accepted Torch2.12/torch_npu2.12 environment. For the first full case,
normal-random tensor hashes differ between the two environments while the
uniform-derived gate/beta hashes agree. A seed is therefore not treated as a
cross-environment byte identity. Every native result is compared to freshly
computed A and B for its actual inputs; bd1/bd2 comparison separately requires
identical input hashes before asserting identical output bytes.

## Completed block_dim1 and block_dim2 native matrices

The complete frozen256-case matrix executed separately at bd1 and bd2 on A5/CCE with CANN9.2.0,
Torch2.12.0 and torch_npu2.12.0. All20 stage outputs, independent leaf launches,
composition, the public wrapper, normalization branch bits and input immutability
met their required checks. This is actual vendor compilation/device execution.
Identical runtime-generated inputs produced bit-identical values for all20
returned stages and seven public gradients in all256 cases across block dimensions.
Same-card measurement also completed, as recorded below. Hardware execution used
`inprocess`; the explicit `aclnn`/`board` transport launchers remain untested.

Public ordinary-subset maximum relativeL2 across the256 cases (identical results at bd1/bd2):

| Gradient | Against literal FP32 A | Against analytical FP32 B |
|---|---:|---:|
| dq | 3.9933345e-07 | 3.9662603e-07 |
| dk | 4.2052579e-07 | 3.7563914e-07 |
| dv | 4.3865543e-07 | 4.3843385e-07 |
| dg_atk | 3.8724255e-06 | 2.0362367e-06 |
| dg | 1.3101692e-06 | 1.448601e-06 |
| dbeta_atk | 1.1249919e-06 | 7.3853813e-07 |
| dbeta | 6.3747901e-07 | 6.7390801e-07 |

The public outputs contain66,581,570 disclosure-only elements per block dimension across all cases,
including structurally absent cotangent paths. Every position and payload is
retained; no relative pass threshold is applied to these elements. Aggregate
counts by reason, maximum absolute output/error, finite global-norm contribution
and undefined/infinite denominator counts are in
`evidence/summary-matrix-bd1-v2.json`; the individual records are under
`evidence/native/matrix-bd1-v2/`. Overall numerical status remains
**numerical disclosure complete, nonPASS**.

All returned stage values in this grid were finite. Observed native ATK history
ranges from0 to19.5137138367, read/write-key cotangent parts have absolute maxima
0.109108791/0.222435489, and dq/dk absolute maxima are6.938435072e9/2.43161366528e11.
These device observations cover the complete grid, while the preceding CPU
whole-chain table covers its explicitly named seven cases; neither is a bound
for arbitrary finite input magnitudes.

## Same-card measurement

The shape is B1/H=HV8/K=V128, FP32, bd2. Each leg has10 warmups and50 samples,
with synchronization immediately before/after every timed call. Times are public-call
wall latency, including validation and allocation. A fresh healthy,
idle device check and shared lock cover the complete run; live context sampling
found no unexplained foreign occupant. CPU references finish before timing, and
all three rounds have classified numerical/disclosure checks before and after.

The baseline reuses this implementation's saved normalization/ATK tapes and main
checkpoints. It excludes stage1/2 device computation but retains public validation
and all allocations, including unused forward outputs. The candidate includes
all four device stages. This measures the cost of forward-state regeneration;
it is not a comparison against another full backward backend or a speedup claim.
Raw samples and both numerical controls are retained in
`evidence/native/timing-bd2-v3/measurement.json`.

| T | Round | Baseline before median ms | Complete backward median ms | Baseline after median ms |
|---:|---:|---:|---:|---:|
| 1024 | 1 | 152.616134 | 180.573714 | 152.637541 |
| 1024 | 2 | 152.560785 | 180.501385 | 152.604458 |
| 1024 | 3 | 152.607138 | 180.486165 | 152.541770 |
| 4096 | 1 | 610.201349 | 726.719444 | 610.426498 |
| 4096 | 2 | 610.543846 | 727.035825 | 610.347384 |
| 4096 | 3 | 610.657986 | 726.575541 | 611.274502 |
