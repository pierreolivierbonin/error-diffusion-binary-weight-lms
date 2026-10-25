# S2 Results — Scale-Predictability Evidence (Consolidated Report)

**Dates:** 2026-07-14 overnight → 2026-07-15. Raw: `results/s2*.jsonl`. Stages per `docs/overnight-plan.md`; one Windows-Update interruption, salvaged and resumed.

## S2.1 — Constant-drive transfer rule: validated

Fit from all fp-gradient ED-GPT runs, then validated by registered prediction at n_embd=512 (predicted lr* 80–90 and drive ~1.05e-3; observed optimum 80, tie at 160; drive 0.89–1.14e-3). Adopted as the flip-lr prescription. Boundary found later the same day: the rule mis-predicts through heavy-ball momentum, and (S2.6 below) transplanting rule-derived lrs across datasets without re-probing violates the protocol and skews comparisons.

## S2.2 — Stationarity and decay: linear decay adopted

50k-step ED (constant lr): loss-stationary, best 1.586 at 37k, then slow drift in flip rate (20.9k → 21.9k/step) and buffer occupancy (0.759 → 0.773). Decay pair: cosine 1.594; **linear→0.1× best at 1.583**, terminal churn cut 8× (2.8k flips/step), no late regression. Linear decay joins the frozen recipe for long horizons.

## S2.3 — Converged baselines (50k, Shakespeare, w256)

| method | best val | state |
|---|---|---|
| BitNet-latent | 1.556 | 97 b/wt |
| **ED-int8 + binact + linear decay** | **1.583** | 9 b/wt |
| FP32+Adam | 1.591 | 96 b/wt |

Both binary methods pass FP32 at convergence (regularization-by-quantization on data-limited char modeling); latent-free ED lands 1.7% from the latent reference while additionally carrying binary activations. Converged ED-int4: 1.659.

## S2.4 — Budget split (momentum vs buffer): buffer wins

At matched state (w192 binact, 5 seeds): int4+int4-momentum best split 56.3 ± 1.1 vs pure int8 buffer 56.4 ± 1.4 (tie); int6+int2 49.1; β=0.99 splits 44.3; at 5 b/wt pure int4 beats int2+int2 by 7.4. Momentum has no role in this optimizer. Failed sub-prediction disclosed: the velocity-scaled drive rule predicted lr=1.0 momentum cells would destabilize; they did not.

## S2.5 — Seed replication (transformer headline cells)

ED-int8+binact w256: 1.893 / 1.885 / 1.867 (3 seeds). ED-int4: 1.984 / 1.909. Scoped+adaptive 1-bit-gradient config: 2.004 / 2.022 / 2.025 → **2.017 ± 0.009**. All headline transformer results replicate within noise.

## S2.6 — enwik8 scaling ladder: gate undetermined (two confounds found)

Byte-level enwik8, 15k steps, seed 0, single run per cell:

| size (params) | fp32 | BitNet-latent | ED (binact, linear decay) | ED vs BitNet |
|---|---|---|---|---|
| 2L/128 (0.5M) | 1.452 | 1.850 (+27% vs fp32) | 1.958 | +5.8% |
| 4L/256 (3.3M) | 1.175 | 1.376 (+17%) | 1.505 | +9.4% |
| 6L/512 (19M) | 1.058 | 1.176 (+11%) | 1.372 | +16.7% |

Read naively, ED's gap to the latent reference widens with scale — a divergence signal that would fail the S3 gate. Two confounds prevent that conclusion:

1. **Activation asymmetry.** ED cells carry binary activations; BitNet cells do not. The widening could be the binact cost growing with scale on real data, not an integrator effect. Control: BitNet+binact at all three sizes.
2. **Flip-lr protocol violation.** The ED lrs were transplanted from Shakespeare-validated values instead of re-probed on enwik8. *(The drive figures first reported here — 5.7e-4 / 3.8e-4 / 3.1e-4 — divided buffer occupancy at step 1000 by 1000 steps; the registered protocol divides occupancy at step 250 by 250. See the correction in S2.6b.)* Control: 250-step probes on enwik8, corrected lrs, rerun ED cells.

Secondary observation: ED at 6L/512 (1.372) matches BitNet at 4L/256 (1.376) — one size up at roughly one third the total binary-layer training-state bits.

**Verdict (superseded by S2.6b): the parallel-curves gate is neither passed nor failed.** Both controls are cheap (≈6 runs); they are the first item of the next compute block. No flagship run is licensed until they land.

## S2.6b — Ladder controls: gate passed (2026-07-15)

**Drive-measurement correction.** Re-probed on enwik8 with the correct 250-step protocol, the ladder lrs were mis-set in *both* directions, not uniformly under-driven: 2L/128 at lr=100 ran over-driven (~1.7e-3 flip-units/step against the 1.0–1.3e-3 band), 4L/256 at lr=60 was already in band (1.06e-3), and 6L/512 at lr=80 was mildly under (~0.9e-3). Probe-corrected lrs: 25 / 60 (unchanged) / 180. Raw probes: `results/s2probe_enwik8.jsonl`.

**Controls** (15k steps, seed 0, both arms binact, ED at in-band lr with linear decay; raw: `results/s2controls.jsonl`):

| size (params) | BitNet-latent + binact | ED (in-band lr) | gap | gap reported in S2.6 |
|---|---|---|---|---|
| 2L/128 (0.5M) | 1.933 | 1.896 (lr 25) | **−1.9%** | +5.8% |
| 4L/256 (3.3M) | 1.469 | 1.505 (lr 60) | **+2.5%** | +9.4% |
| 6L/512 (19M) | 1.237 | 1.287 (lr 180) | **+4.0%** | +16.7% |

The apparent divergence decomposes almost entirely into the two confounds: binary activations cost the latent arm 0.06–0.09 nats at every size (the S2.6 comparison charged that cost to ED alone), and the mis-set flip-lrs cost ED 0.06 nats at 128 and 0.09 nats at 512. Like-for-like, ED *beats* the latent reference at the smallest size, and the gap increments decelerate (+4.4 points from 128→256, +1.5 from 256→512) — curves flattening toward parallel, not diverging. The one-size-up observation strengthens: ED at 6L/512 (1.287) now beats BitNet+binact at 4L/256 (1.469) and the no-binact BitNet 4L/256 (1.376) outright.

**Verdict: the parallel-curves gate passes at ladder scale.** Caveats that stay attached: single seed per cell (transformer seed spread measured at ±0.01–0.05 in S2.5), byte-level data only, and the small residual slope (−1.9 → +2.5 → +4.0%) is not yet distinguishable between a true scale cost and cell noise. The S3 flagship run is licensed once the engineering prerequisites (checkpoint/resume first) are in place.

## S2.7 — Second-order ΔΣ buffer (ed2): negative

The ED buffer is a first-order sigma-delta modulator in time; ΔΣ theory promises more low-frequency noise suppression from a second-order loop, so ADR-013 defined `ed2`: two cascaded integrators, flip decided on the second, realized movement debited from both. Tested at matched total state bits (w192, binact, 5 seeds; raw: `results/s2sigma.jsonl`):

| config | state | best acc (over lr 0.2–2.0, C 1.0/1.5) |
|---|---|---|
| pure int8 buffer | 9 b/wt | **56.4 ± 1.4** |
| ed2 int4+int4 | 9 b/wt | 29.1 ± 1.3 |
| pure int4 buffer (C=1.0) | 5 b/wt | **56.4 ± 1.3** |
| ed2 int2+int2 | 5 b/wt | 22.5 ± 4.2 |

Every cascade cell collapses into sustained thrashing: flips grow to ~10^7 per epoch (most of the 73k binary weights flipping every step) regardless of lr or clamp — lr-insensitivity that marks the instability as structural, not a tuning miss. The mechanism: the cascade adds the first integrator's full carry into the second every step (t2 = e2 + e1 + u), so after any flip the loop is immediately re-primed at the threshold; the spurious-firing regime that C=1.5@2bit produces in the first-order loop is here the *only* regime. Second-order noise shaping in ΔΣ converters survives because the modulator runs orders of magnitude faster than the signal; the training loop has no such oversampling ratio to spend. Sigma-delta loop order joins momentum (S2.4) as a confirmed non-buyer of state bits: depth of a single buffer is where bits pay. Bonus replication: the pure int4 buffer at C=1.0 (56.35 ± 1.26, 5 b/wt) ties the int8 reference (56.42 ± 1.40, 9 b/wt) at 5 seeds — the grid rule holding at matched protocol.

## S2.8 — Integer-latent control (reviewer equivalence): equivalence confirmed

The strongest anticipated objection (lit-review): "sign bit + k-bit buffer is just a (k+1)-bit integer latent read out by sign." Measured with `intlat` (saturating SR-quantized latent on a uniform grid over [−2, 2], sign readout, no carry-across-flip recentering; w192, binact, 5 seeds, best over lr {0.3, 1.0, 3.0}):

| state | integer latent | ED buffer |
|---|---|---|
| 9 b/wt | 56.2 ± 1.2 | 56.4 ± 1.4 |
| 5 b/wt | 56.1 ± 0.8 | 56.4 ± 1.3 |
| 3 b/wt | 52.1 ± 1.1 | 52.3 ± 1.2 (edB int2, C=1.0) |

**The two parameterizations tie at every matched bit width: they are one equivalence class.** The reason is geometric: the intlat grid is symmetric about zero with levels adjacent to the sign boundary, so it automatically satisfies the C=1.0 condition — SR jumps across the flip boundary are always available. The result strengthens rather than weakens the paper's framing: the buffer coordinates are the frame in which the design rules are visible (C=1.0 explains *why* both work and predicts the failure of any grid whose top level clears the threshold — the C=1.5@2bit and C<1 regimes), while the integer-latent frame fixes its geometry by construction and cannot express the design variable. Claim consequence: no accuracy advantage over a well-built integer latent is claimed; the contribution is the geometry rule, the drive invariant, and the interaction law, all formulated in flip units and applying to both members.

## S2.9 — Multi-seed ladder and the 512 ignition bistability

Three seeds per like-for-like cell (`results/s2ladderseeds.jsonl`):

| size | BitNet+binact | ED (in-band lr) | gap |
|---|---|---|---|
| 2L/128 | 1.943 ± 0.007 | 1.894 ± 0.002 | **−2.5%** |
| 4L/256 | 1.468 ± 0.006 | 1.516 ± 0.012 | **+3.3%** |
| 6L/512 | 1.239 ± 0.002 | see below | +3.9% (clean mode) |

The 128 and 256 verdicts replicate with tight spreads. At 512, five ED runs at lr=180 split bimodally: three land at 1.287–1.291 (spread 0.004), two stall at 1.433/1.486. The stall is an ignition failure, not a thrash: stalled runs show val *rising* between steps 1k and 2k with the flip rate stuck at 15–20k/step, while clean runs drop ~0.4 nats with flips ramping 13k → 31–33k. **Flip-lr warm-up is falsified as the fix** (EDWARM=1500 on the two stalled seeds: one rescued to 1.302, one worsened to 2.195 and early-stopped — warm-up perturbs the basin choice, it does not select the good one; suppressing early flips can make non-ignition permanent while Adam fits the FP remnant around a frozen core). **Operational resolution: detect and restart.** The signature separates all five runs at step 2000 (val(2k) > val(1k) ⇔ stall); detection costs ~6 min at this size, expected overhead <1 restart per run at the observed 40% stall rate. Folds into the checkpoint/resume engineering item.

## S2.10 — Deploy-then-adapt: warm-start ED beats latent continuation

The edge scenario measured end-to-end (4L/256, binact, seed 0; pretrain BitNet+binact on enwik8 15k steps → save deployed artifact (signs + scales + FP remnant; latents discarded for the ED arm) → adapt 5k steps to byte-level Shakespeare; `results/s2adapt.jsonl`):

| arm | best val | adaptation state |
|---|---|---|
| zero-shot (no adaptation) | 2.316 | — |
| ED from scratch (matched budget) | 1.951 (early-stopped 3.5k) | 9 b/wt |
| **ED warm-start, probe-set lr 60** | **1.522** | **9 b/wt** |
| BitNet-latent continuation (kept latents + fresh Adam) | 1.534 | 97 b/wt |

The warm start is worth 0.43 nats over scratch, and adapting the deployed artifact at 9 b/wt edges the latent-based continuation — the arm an edge device could not run. Checkpoint transfer is lossless (ED zero-shot exactly reproduces the pretraining val). The constant-drive band transfers to warm-started training: lr=60 (in band) beats lr=30 (just under) at the 5k horizon, despite the reverse ordering at 250 steps; over-driving (lr=120, 2.4e-3) destroys the deployed model in 250 steps. Caveats: single seed; the latent arm used the project-standard Adam lr without an adaptation-specific sweep.

## S2.11 — Grid–threshold alignment generalizes: dithered Bop rescued

Registered prediction (before running): aligning dithered Bop's EMA quantization grid so its top level sits exactly at the flip threshold τ — the C=1.0 rule transposed to a foreign accumulator — rescues the known low-bit collapse. Measured (w192, binact, 5 seeds, best over τ; `results/s2bopalign.jsonl`):

| config | state | best acc |
|---|---|---|
| Bop, FP32 EMA | 33 b/wt | 56.6 ± 1.5 |
| Bop 4-bit EMA, absmax grid | 5 b/wt | 46.5 ± 1.3 |
| **Bop 4-bit EMA, τ-aligned grid** | **5 b/wt** | **57.0 ± 1.0** |
| Bop 2-bit EMA, τ-aligned grid | 3 b/wt | 51.0 ± 1.2 (peak at τ=3e-4; falls both ways) |

Fill-in cells (2026-08-23, same protocol, `results/s2bopalign2.jsonl`, `scripts/s2bopalign2.sh`), so the paper's Table 1 has no gaps:

| config | state | absmax grid | τ-aligned grid |
|---|---|---|---|
| Bop 8-bit EMA | 9 b/wt | 55.9 ± 1.5 | 56.1 ± 0.6 |
| Bop 3-bit EMA | 4 b/wt | 45.3 ± 1.0 | 53.8 ± 0.8 |
| Bop 2-bit EMA | 3 b/wt | 38.6 ± 2.1 | 51.0 ± 1.2 |

At 8 bits the grids tie (a level already lies within one per-step drive of τ); below that the rescue is monotone: +10.5 at 4, +8.5 at 3, +12.4 at 2.

## S2.16 — Exchange-curve int4 line re-run at C=1.0

The P1 width sweep's ED-int4 series (`widthsweep_5seed.jsonl`) predates the grid rule and ran at the naive clamp C=1.5; at w384 it thrashed (1.21M flips/epoch, σ=2.1) and plateaued at 56.84, which a reviewer read as a truncated line. Re-run 2026-08-23 at C=1.0 (`scripts/widthsweep_int4c1.sh`, `results/widthsweep_int4c1.jsonl`, 5 seeds, binact):

| W | ED int4, C=1.5 (old) | **ED int4, C=1.0** | ED int8 (P1) |
|---|---|---|---|
| 64 | 39.82 | 40.30 ± 1.33 | 40.02 |
| 96 | 45.26 | 46.82 ± 1.56 | 46.78 |
| 128 | 48.63 | 50.35 ± 1.20 | 51.45 |
| 192 | 53.22 | 56.35 ± 1.26 | 56.42 |
| 256 | 56.42 | 59.76 ± 0.84 | 59.33 |
| 384 | 56.84 | 62.52 ± 0.46 | 62.53 |

Under the rule, int4 matches int8 at every width (flips_epN at w384: 3.4e4 vs 1.2e6 naive). `tools/plot_exchange.py` now plots the C=1.0 series; the paper's Figure 2 uses it.

## S2.17 — Comparability-audit fill-in cells (2026-08-23)

`scripts/audit_fix.sh` → `results/audit_fix.jsonl` (transformer) and `results/audit_fix_mlp.jsonl` (MLP). Fixes the paper's Table 2 left to one regime (FP activations, 2 seeds per cell) and grounds the interaction law like-for-like at C=1.0:

- ED int8 seed 1 at the chosen lrs: d128@lr100 2.0817 (2-seed mean with p3gpt seed0 2.0358 → **2.059**), d256@lr60 1.8789 (→ **1.861**).
- d512 at BINACT=0, 2 seeds: BitNet **1.576** (1.5764/1.5762), ED@lr80 **1.757** (1.7715/1.7421). vs-latent row becomes +5.1/+8.3/+11.5% (pre-registered +20% criterion still passes).
- 1-bit grads (sr1_rht) at C=1.0, w192, 5 seeds: 4-bit buffer **48.63 ± 1.03** (−7.8 vs 56.4), 2-bit **38.10 ± 0.90** (−14.2 vs 52.3) — the previously reported −7.7/−14.2 survive the like-for-like re-run.

Also from the int4-transformer sweep (`s2int4gpt.jsonl`): int4 2k-step cells (FP acts, best-of-sweep, 2 seeds): d128 **2.085**, d256 **1.860**, d512 **1.789** — within 0.04 of int8 at this budget. int4 adapt at probe lr30, 3 seeds: **1.610 ± 0.005**.

## S2.18 — int4 transformer at probe-set flip-lrs (final, 2026-08-23)

`scripts/s2int4gpt3.sh` → `results/s2int4gpt3.jsonl`. int4 probes (`s2int4probe.jsonl`) put in-band lrs at 12/30/45 for the enwik8 ladder (int8: 25/60/180 — 2–4× lower; at 19M the 250-step probe saturates near the band for lr ≥ 90, so the lower edge is mandatory, an int4-specific probe caveat). Ladder, 3 seeds/cell, binact, linear decay, **no stalls**:

| size | int4 (probe lrs) | int8 ref | gap |
|---|---|---|---|
| 0.5M | 2.002 ± 0.003 | 1.894 | +5.7% |
| 3.3M | 1.707 ± 0.003 | 1.521 | +12.2% |
| 19M | 1.510 ± 0.002 | 1.288 | +17.2% |

50k convergence d=256 (shakespeare-char, binact, linear decay): lr45 (drive-matched to int8's pick, registered comparable cell) **1.681**; lr30 bracket 1.675; int8 1.583 → +6.2%. The earlier reused-lr runs (S2.16 note, `s2int4gpt.jsonl` ladder rows, `s2decay` int4@60) stand as the protocol-violation record. Conclusion: int4 matches int8 at 2k steps and on the MLP exchange curve; at longer horizons the gap opens with scale even at in-band lrs. Paper §5 carries this as "int4 on the transformer".

At 4 bits the rescue is total: the aligned grid matches the FP32-EMA reference at 6.6× less state, in an optimizer with a different accumulator (gradient EMA, not residual), different firing semantics (magnitude + sign-agreement), and no design debt to ED. At 2 bits the rescue is partial (51.0 vs unquantized 56.6), landing in the same band as ED's own 2-bit cell (52.3) and the 3-bit integer latent (52.1) — matched-state parity again. The rule is thereby promoted from an ED design choice to a measured property of thresholded quantized accumulators: **a level of the state grid must sit exactly at the event threshold.** This experiment also discharges the "quantized-momentum Bop" mandatory baseline (Yayla–Chen direction) at stronger-than-required strength.

## S2.12 — Final two adversarial baselines: ECO-style leak and the sign counter

**ECO-style residual-in-momentum (edE)** — the ECO construction transposed to binary: no dedicated buffer; the flip residual is carried in a β-decayed momentum (β=1 recovers ED exactly). w192, binact, 5 seeds, lr swept ×1/(1−β) to compensate the leak:

| β | best acc (FP32 mom, 33 b/wt) | best acc (int8 mom, 9 b/wt) |
|---|---|---|
| 0.9 | 44.5 | 45.4 (47.3 at int4) |
| 0.99 | 55.8 | 55.3 |
| 1.0 (= ED) | — | **56.4** |

Monotone in the leak: a decaying carrier cannot hold sub-quantum information against a binary event quantum. ECO works at FP8/INT4 because updates there are not sub-quantum; at binary precision the construction converges to ED as β→1. This discharges the ECO comparison: the differentiation is now measured, not argued.

**Sign counter (cnt)** — Colombo-direction control (unit sign increments into a saturating counter, flip at the rail, no SR; backprop gradients, so a control, not a reimplementation of their gradient-free scheme):

| counter | state | acc |
|---|---|---|
| int8 (rail 127) | 9 b/wt | **58.4 ± 0.8** |
| int4 (rail 7) | 5 b/wt | 52.4 ± 2.5 |
| int2 (rail 1) | 3 b/wt | 10.1 ± 3.2 |

Honest surprise: at 8 bits the magnitude-free counter beats both ED-int8 (56.4) and BitNet-latent (57.5) on this task — 127-step sign consensus is extremely noise-robust, and with unit increments nothing is sub-quantum, so no dither is needed. The regime split is the finding: **at ≥8 bits of state, sign-counting suffices and SR buys nothing; at ≤5 b/wt the counter degrades then collapses (rail-1 = flip-on-every-opposing-sign thrash) while SR + grid geometry keeps training alive** (ED 56.4 at 5 b/wt, 52.3 at 3). The exchange-curve/low-bit claims are unaffected; any claim of ED optimality at 8-bit state on the MLP task is now qualified. Open: counter cell on the transformer (queued; slow-flip dynamics may behave differently over 15k steps).

## S2.13 — Tokenized-corpus ladder (wikitext-103, GPT-2 BPE): parity at two sizes, open gap at the third

117.9M tokens, block 128, 15k steps, seed 0, probe-set lrs (40/90/225; the probes needed ~1.5× the byte-level lrs — the drive rule absorbed the tokenizer change). Raw: `results/s2tokladder.jsonl`.

| size | fp32 | BitNet+binact | ED+binact | ED vs BitNet |
|---|---|---|---|---|
| 2L/128 | 4.731 | 5.113 | 5.118 | **+0.1%** |
| 4L/256 | 4.355 | 4.789 | 4.782 | **−0.1%** |
| 6L/512 | 4.042 | 4.464 | 4.920 (lr 225) / 5.038 (lr 180) | **+10.2%** |

At 0.5M and 3.3M parameters, latent-free ED ties the latent reference exactly on tokenized data — stronger parity than the byte ladder showed. At 19M the gap is real and not an lr artifact: both band-edge picks land 10–13% behind, the lr=225 run shows a thrash episode at step 3k that passed the step-2k ignition check (the detect rule needs a second checkpoint at 3k), and both ED runs were still descending at the 15k cutoff while descending faster than the BitNet arm (0.05 vs 0.02 nats/1k at the end) — an equal-budget gap with an unconverged tail. Candidate explanations, unresolved: (a) at 512 the FP remnant (embeddings+head, 51.5M Adam-trained params) outweighs the binary core (18.9M) 2.7:1 on this vocabulary, a regime the byte ladder never entered; (b) the drive band's stable width narrows at 512 (the probe curve turns steeply super-linear just above lr 225); (c) ED converges slower at this size/data and 15k steps under-serves it. **Verdict: tokenized parallelism confirmed at two of three sizes; the 19M-tokenized cell is an open stability/convergence boundary, reported as such.** Follow-ups: longer budget at 512, seed spread, and the remnant-dominance interaction.

## Status of the S3 gate checklist

- Transfer rule: ✅ validated (with known boundaries).
- Long-horizon stability: ✅ with linear decay.
- Converged-regime competitiveness: ✅ (1.7% from latent reference).
- Real-data scaling parallelism: ✅ at 3 seeds on bytes (S2.9); tokenized: ✅ at 128/256 (exact parity), ⏳ open boundary at 512 (S2.13; the flagship plan targets the mid-size regime or resolves the 512-tokenized boundary first).
- Engineering (checkpoint/resume, GPU RNG, compile): 🟡 deploy-checkpoint save/load exists (S2.10 infrastructure); full resume (buffer + Adam + RNG state) still open and gates the flagship run, now with the stall-detection rule as part of the launch protocol.

## S2.14 — Pre-submission additions (2026-07-16)

Adaptation at 3 seeds: ED warm-start 1.520 ± 0.003 vs latent continuation 1.533 ± 0.007 — the warm-start edge replicates. Counter on transformer (2 seeds, 2k budget): 1.846 ± 0.013 vs ED-int8 1.882 ± 0.011 — the 8-bit counter edges ED on both architectures; regime split unchanged. Paper gains: appendix mapping every table/figure to script+log; bitedge E3 measurement (4 MB artifact, 275 MB CPU adaptation demo) cited in the on-device paragraph. Raw: results/s2cnt.jsonl, results/s2adapt.jsonl.

## S2.15 — 512-tokenized boundary: closed as a genuine limitation (2026-07-17)

Three 30k-step probes (results/s2tok512long.jsonl): BitNet reaches 4.358 (from 4.464 at 15k). ED fails three ways: slow decay at lr 225 → 5.07 (thrash persists); slow decay at lr 180 → 5.29 (oscillating); decay-to-15k-then-floor → retraces the good trajectory, then destabilizes at the constant floor (flips 13k → 56k/step, val oscillating 5.1–6.2 for 9k steps; best 5.12). Verdict: budget and schedule are ruled out — at this scale/vocabulary ED holds no stable operating point at any sustained lr; the 15k fast-decay result (4.92) is its practical best, and the ~13–17% gap to the latent arm is real. Leading suspect by elimination: the Adam-trained FP remnant (2.7:1 vs the binary core) interacting with the moving binary layers. This sharpens the parent-paper limitation and defines the next research question (remnant/core co-stability), rather than an engineering fix.
