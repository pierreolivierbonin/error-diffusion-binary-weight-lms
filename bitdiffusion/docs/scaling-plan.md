# Scaling Plan — From Controlled Experiments to a Flagship Run

**Written 2026-07-14.** Goal (PI directive): experiments must predictably extrapolate to large compute, so a multi-day/multi-week training run can be launched with justified confidence that it converges and competes with — at matched training memory, outclasses — models trained conventionally.

## The claim being scaled

Not "binary matches FP at equal parameters" (the ~4–7% binarization tax makes that the wrong fight). The claim is the **memory exchange** (P1): at matched training-state bits, wide-binary beats narrow-FP, with the advantage growing toward 14× in the tested range. At fixed VRAM, ED trains ~10× the parameters of AdamW/latent training. The flagship bet: a 10× wider ED model beats the FP model that fits in the same memory, at matched tokens.

## Stage gates

### S1 — Recipe freeze (now)
Finish P3.5 (scoped quantization). Freeze: ED-int8, C=1.0 grid rule, binact MLP, FP attention internals; gradients = scoped sr1_clip if P3.5 lands ≤ ~2.0, else FP gradients (defer 1-bit G to research track). Every S2 experiment uses the frozen recipe.

### S2 — Predictability evidence (days of compute, gate for S3)
1. **Flip-lr transfer rule.** Fit optimal flip-lr vs width/depth/gradient-scale from existing data (MLP 64–384 sweep + GPT 128/256 + quantizer shifts — all on disk, never fit). Deliverable: a formula or normalization (µP-analog) validated by *predicting* the optimum at an untested width (e.g. GPT n_embd=512) within 2×. P3.4's RMS-normalization failure narrows the search: normalize by update coherence, not raw RMS.
2. **Stationarity run.** One 50k-step ED-GPT run: flips/step, buffer abs-mean, val loss monitored for drift or late-phase pathologies; test flip-lr decay schedule (ED analog of cosine — untested anywhere). Kill signal: flip rate or buffer norm trending at 50k steps.
3. **Scaling ladder.** Tokenized real corpus (OpenWebText-class subset, BPE ~8–16k vocab). Three sizes (~6M / 25M / 100M) × {FP-AdamW, BitNet-latent, ED frozen recipe} at compute-matched token budgets. Deliverable: loss-vs-compute curves. **Gate: ED's curve parallel to baselines (constant offset). Divergence = stop, understand, do not scale.**

### S3 — Flagship (weeks, only if S2 gates pass)
Largest ED model that fits 16 GB with margin vs the largest FP model that fits the same 16 GB, matched tokens and wall-clock class. Success = ED val loss strictly better at equal memory and tokens.

## Engineering prerequisites (fatal at scale, fine today)

- **GPU-side RNG:** replace CPU-generator draws (ADR-003) with counter-based (Philox) on-device RNG, deterministic per (seed, step, tensor). Removes the ~10× hook tax; revises ADR-003 (device-portability sacrificed knowingly at scale).
- **Checkpoint/resume:** wb, ebuf, RNG counters, optimizer state, data cursor. Bit-exact resume verified before any long run.
- **torch.compile + fused ED step:** the update is elementwise — one fused kernel; matmuls in bf16 for FP parts.
- **Data pipeline:** streaming tokenized shards; val protocol fixed in advance.
- **Monitoring:** JSONL eval records (ADR-011) + flip/buffer/loss dashboards; automatic kill criteria (thrash, dead buffers, loss NaN) — the P2/P3 diagnostics wired as guards.

## Known unknowns to carry honestly

- Attention internals and embeddings/norms/head remain FP in the frozen recipe; their Adam state is the dominant residual memory term — 8-bit Adam or binary embeddings (see reading-notes on 2509.07025) are S2-optional, S3-relevant.
- Depth scaling untested (all transformer data is 4-layer). The ladder varies depth with size; watch for depth-linked instability (binary layers × LayerNorm interactions).
- The exchange-curve saturation seen at w384 on the toy task (gap narrowing) may or may not appear on real data — the ladder answers this.

## S2.1 first result — the constant-drive transfer rule (2026-07-14)

Fit from all logged fp-gradient ED-GPT runs: raw optimal flip-lr varies 60→≥100 across widths (and ~60× vs the MLP task), but the **early buffer-fill rate at the optimum is constant: drive ≈ 1.0–1.3e-3 flip-units/step** (w256 fp-acts 9.6e-4, w256 binact 1.0e-3, w128 1.3e-3; MLP consistent at ~2× given its ~2× longer horizon). Equivalent statement: total accumulated drive ≈ 2–4 flip-units per run.

**Rule:** at any new scale, run a 250-step probe at a reference lr, measure buf_absmean(250)/250, and set lr* to hit ~1.05e-3 (scaled by horizon for longer runs). **Registered prediction:** n_embd=512 → lr* ≈ 80–90 (drive-per-lr extrapolated −20% from the 128→256 trend). Validation = bracket {40, 80, 160}; rule survives if the bracket's interior point wins and probe-implied lr* lands within 2×.

**Validation (same day):** n_embd=512 bracket {40, 80, 160} → 1.844 / **1.782** / 1.783; drive at the winning lr = 8.9e-4, at its tie 1.14e-3 — bracketing the predicted 1.05e-3; probe-implied lr* = 94, within 2× of the realized optimum. **Both registered criteria met; the constant-drive rule is adopted as the flip-lr prescription for the S2 scaling ladder.** (Context: bitnet+binact @512 = 1.630; ED +9.3%.)

## S2.2 result — 50k-step stationarity (2026-07-14)

ED-int8+binact, w256, lr 60, 50k steps: **loss-stationary** (last 10 evals 1.609 ± 0.005), best **1.586 @ 37k** — converged latent-free ED reaches the 2k-step fp32 line (1.575); the 2k-budget tables understate every method. **State drift confirmed:** buffer fill 0.759 → 0.773 and flips 20.9k → 21.9k over the final 10k steps, monotone — benign here, clamp-pinning if extrapolated to week horizons. **Gate consequence: flip-lr decay schedule is a hard S3 prerequisite** (untested). Post-37k plateau regression (~0.02) is consistent with drift cost or ~150-epoch data recycling. Raw: `results/s2stationarity.jsonl`.
