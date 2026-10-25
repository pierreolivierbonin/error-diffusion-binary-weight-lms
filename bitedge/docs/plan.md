# bitedge — On-Device Learning with Error-Diffusion Training

> Release note: this is the plan as committed on 2026-07-16, with one planned follow-up experiment (section E4) and its three mentions removed. The rest is unchanged.

**Goal.** The bitdiffusion paper claims that on-device learning "stops being blocked by training-state memory and becomes blocked by ordinary porting work." This project does the porting work and measures whether the claim survives contact with real memory budgets, real integer arithmetic, and simulated device physics. Parent repo: `D:\Work\1bit_model` (checkpoint format, drive-rule protocol, and the S2.10 deploy-then-adapt result are the starting points).

**Relationship to the paper.** Every experiment here either substantiates a sentence already in the paper's Future Directions (turning arithmetic into measurement) or seeds a follow-up paper. Nothing here blocks the arXiv submission.

**Hardware targets, in order of realism:** (1) the RTX 4080 as a *measured-memory* proxy (peak-allocation budgets enforced in code, not hand-waved); (2) CPU-only adaptation under a hard RSS cap, standing in for inference-class edge hardware.

---

## E0 — Integer-only update kernel (the "ordinary porting work" claim)

The ED update in pure integer arithmetic: buffer as literal int8 tensors, threshold comparison in integer units, stochastic rounding from a counter-based integer RNG (xorshift or Philox), no float in the update path. Verify on the MLP harness that integer-kernel training matches the float-emulation reference within seed noise (5 seeds, w192, binact — the S2 protocol).

- **Success:** accuracy within 1σ of the float-emulated edB reference at int8 and int4.
- **Kill:** if integer SR (fixed-point probability draw) systematically degrades vs float SR, the porting claim needs a caveat — report the gap and its mechanism.
- **Cost:** ~2 days implementation, minutes of GPU.

## E1 — The FP remnant (the real memory frontier)

The paper discloses that embeddings, norms, head, and their Adam state dominate what's left. Attack in three steps at 4L/256 on shakespeare_bytes (the S2.10 setting): (a) 8-bit Adam states on the remnant; (b) SGD-momentum on norms (tiny anyway); (c) binary embedding layers in the 2509.07025 direction. Report **whole-model** persistent training state in MB, not per-binary-weight bits — the honest denominator the paper's accounting deliberately scopes away.

- **Success:** end-to-end adaptation state under 3× the deployed artifact size with val loss within 2% of the FP-remnant reference.
- **Kill:** if binary embeddings cost more than 5% val loss, keep 8-bit embeddings and say so.
- **Cost:** ~3 days, small GPU runs.

## E2 — Measured memory, not accounted memory

Instrument the adaptation step: `torch.cuda.max_memory_allocated` (GPU) and RSS (CPU) for ED-adapt vs latent-continuation at matched config, plus bitpacked storage (binary weights as uint8, 8 weights/byte; sign activations stored 1-bit for the backward). The paper's 0.1 GB vs 1.2 GB arithmetic becomes a measured table, including the activation memory the paper admits is unchanged.

- **Success:** a peak-memory table where the predicted state gap appears in the measured totals; identification of what fraction activations occupy.
- **Cost:** ~2 days.

## E3 — The edge demo (deploy-then-adapt under a budget)

End-to-end scenario on CPU with an enforced memory cap chosen to mimic inference-class hardware (target: full adaptation of the 4L/256 model under 500 MB RSS). Load the deployed checkpoint from the parent repo, run the 250-step probe, adapt to shakespeare_bytes, produce the S2.10 quartet numbers on the constrained device, plus wall-clock. This is the demonstration artifact: a script anyone can run on a laptop.

- **Success:** the warm-start arm reproduces its GPU result (val ~1.52) under the cap; the latent-continuation arm *cannot run* under the same cap — the claim's content, demonstrated rather than argued.
- **Cost:** ~2 days after E0-E2.

---

## Order and gating

E0 → E2 → E1 → E3. E0 first because everything downstream inherits its kernel; E2 before E1 because the memory instrumentation defines E1's success metric. E3 is the demo; it waits for all three.

## Standards

Same as the parent repo: uv, ruff, ty, pre-commit with the no-comments hook, src layout, JSONL results with full config per record, ADR.md for design decisions, registered predictions before decisive runs, negative results documented with mechanisms.
