# Architecture Decision Records

Append-only. One entry per architectural decision: context → decision → consequences.

## ADR-001 — Refactor preserves the legacy RNG stream (2026-07-14)

**Context:** `exp2.py` numbers are the project's evidence base; a refactor that silently changes RNG consumption order makes every new number incomparable.
**Decision:** `src/bitdiffusion/` reproduces `exp2.py`'s generator consumption order exactly (RHT signs drawn before weights in `setup()`; per-epoch `randperm` then per-batch draws). Equivalence is verified seed-for-seed on CPU for FP-gradient modes.
**Consequences:** The RNG order is a frozen contract — documented in `docs/package-notes.md`; refactors must not reorder generator draws.

### Addendum (ADR-001)

`exp2.py` is excluded from ruff/ty/no-comments tooling: it is the frozen equivalence reference; any reformatting or edit would contaminate the artifact the refactor is verified against.

## ADR-002 — Fix nondeterministic 1-bit gradient sampling (2026-07-14)

**Context:** `exp2.py`'s `_q1_sr` used `torch.bernoulli` on the global RNG; Phase-3 `sr1`/`sr1_rht` cells were irreproducible.
**Decision:** Thread the run's explicit generator through the gradient quantizers (`rand < p` replaces `bernoulli`).
**Consequences:** `sr1`/`sr1_rht` cells get new (now reproducible) exact values; FP-gradient cells are unaffected. Phase-3 gradient numbers must be re-baselined (subsumed by the 5-seed rerun).

## ADR-003 — All randomness on CPU generators, tensors on device (2026-07-14)

**Context:** Local GPU (RTX 4080) is now the primary compute; CUDA generators would give different streams than CPU, splitting the evidence base by device.
**Decision:** Every stochastic draw uses a CPU `torch.Generator` and is moved to the device (`layers.crand`). `DEVICE` env selects placement only.
**Consequences:** CPU and GPU runs consume identical random streams; results differ only by float non-associativity in matmuls. Cost: one host→device transfer per draw — negligible at MLP scale, revisit for nanoGPT.

## ADR-004 — uv-managed environment, torch cu128 via explicit index (2026-07-14)

**Context:** System Python had CPU-only torch; driver is CUDA 13.0.
**Decision:** Project venv via `uv sync`; `torch` pinned to the `pytorch-cu128` explicit index in `pyproject.toml`; dev tools ruff + ty.
**Consequences:** Reproducible env from lockfile; GPU builds don't leak into system Python.

## ADR-005 — src layout, comment-free code, prose in docs/ (2026-07-14)

**Context:** PI directive: uv + src layout, no comments in code, all prose in `docs/`.
**Decision:** Package at `src/bitdiffusion/`; explanatory prose lives in `docs/package-notes.md`; enforcement via pre-commit hook rejecting comments/docstrings in `src/`.
**Consequences:** Naming and structure must carry meaning that comments otherwise would; design rationale is recorded here and in `docs/`.

## ADR-006 — Re-baseline all recorded numbers locally (2026-07-14)

**Context:** Legacy numbers came from a different torch build/container; local CPU repro of the w=96 base trio is within noise but not bit-exact.
**Decision:** All comparisons in new work use locally re-run baselines (5 seeds, GPU); legacy numbers are kept as directional reference only.
**Consequences:** One-time compute cost; every claim becomes internally consistent and auditable on this machine.

## ADR-007 — Quantized Bop EMA uses per-layer dynamic absmax scale (2026-07-14)

**Context:** P0's fair Bop comparison needs an int8-EMA variant at matched buffer bits. Bop's EMA magnitude (~1e-4, gradient-scaled) is incompatible with ED's fixed [−1.5, 1.5] flip-unit grid.
**Decision:** `bop_step(bits=...)` SR-quantizes the EMA onto 2^bits levels over a per-layer, per-step absmax range. The scale is O(1) FP metadata per layer — same acceptance class as ED's per-layer weight scale.
**Consequences:** Comparable bits/weight accounting (bop int8 = 9 b/wt vs ED int8 = 9 b/wt); the dynamic range tracks EMA shrinkage, which a fixed grid could not. Default `bits=None` leaves legacy Bop untouched and consumes no extra RNG.

## ADR-008 — P2 designs: two-color spatial error diffusion and refractory hysteresis (2026-07-14)

**Context:** The ~4-bit integrator floor (P0-5) blocks Tier C. At 2 bits the buffer quantum (1.0) equals the flip threshold, so SR thrashes and DET erases. P1-3 showed the buffer is now the binding constraint on total state.
**Decision (edS — spatial):** Diffuse flip residuals along matrix **rows** because the row is the receptive field of the matmul's summation: z_i = Σ_j w_ij·h_j, so a flip realized at a neighboring j' approximates the intended Δz_i — the same spatial-averaging that makes halftoning work, with the forward pass playing the role of the eye. True Floyd–Steinberg is a sequential raster (untenable on GPU: O(W) Python iterations per step); instead use the classic **two-color parallelization**: phase A processes even columns and sends (1−α) of their residual to odd neighbors (half left, half right, torus wrap); phase B processes odd columns with that inflow pre-threshold and sends (1−α) of theirs back to even buffers; each weight keeps α of its own residual. All inter-column traffic is transient within the step — persistent state stays one b-bit buffer per weight. α=1 reduces exactly to edB (bit-identical, used as the regression gate).
**Decision (edH — hysteresis):** One persistent refractory bit per weight (set on flip, cleared after one flip-free step); recently-flipped weights need |t| ≥ 1+δ to flip again. Directly targets the thrash mode; costs exactly 1 bit (state = 2+b bits/wt vs edS's 1+b).
**Consequences:** Two new update rules in `optim.py` sharing an extracted `_srq` quantizer (stream-identical refactor of edB's rounding). Floor must first be characterized at 3 bits (never measured). Success criterion per handoff: ≥53.4% at w192 binact with ≤2 buffer bits.

## ADR-009 — Buffer grid rule: top quantization level at the flip threshold (2026-07-14)

**Context:** P2's clamp sweep showed every sub-threshold grid (C<1) is inert at any bit width and every supra-threshold level drives SR grid-jump flips; the razor-sharp boundary (C=0.95 dead, C=1.0 works) proves ED flips fire exclusively via SR landing a buffer level with |level| ≥ threshold.
**Decision:** For low-bit buffers the grid must place its top level exactly at the flip threshold (C=1.0 in flip units). Scripts pass `clamp 1.0` explicitly for buf_bits ≤ 4; the code default stays 1.5 so every previously published number remains reproducible.
**Consequences:** The "~4-bit integrator floor" is reclassified from obstruction to design artifact: at C=1.0, 4-bit = 8-bit (56.4), 3-bit −1.1, 2-bit −4.0 (52.3, robust across 6× lr). The buffer is correctly modeled as a stochastic accumulate-and-fire unit; spurious-firing rate, not representational precision, sets the remaining 2-bit gap. Interaction caveat: 1-bit gradient noise multiplies spurious firing at the 2-bit boundary (full stack 38.1) — gradient precision and buffer bits are not independently compressible at the extreme.

## ADR-010 — P3 transformer port design (2026-07-14)

**Context:** The handoff's P3: validate ED training on a char-level transformer where attention, LayerNorm, and residual streams complicate the MLP story.
**Decision:** Minimal GPT (nanoGPT-style: 4 layers, 4 heads, LayerNorm, GELU, SDPA causal attention, no dropout/bias) at `src/bitdiffusion/gpt.py`. Binary (BitLinear) surfaces: attention qkv + output projection, MLP fc + proj — 3.15M of 3.25M params at n_embd=256. FP kept (standard exclusion, ablate later): embeddings, LayerNorms, LM head — trained by AdamW 3e-4. ED updates run from module-held proxies after backward; buffers registered as non-Parameter tensors so `model.parameters()` cleanly separates the FP remnant. C=1.0 grid rule (ADR-009) is the default in scripts. FP activations and FP gradients first; binact-MLP and sr1_rht hooks are the tightening steps after the baseline story is established.
**Consequences:** Same RNG discipline (one CPU generator: init → batch sampling → SR draws). Confirmed transformer-scale LR pathology: at edlr=1 the buffer fills ~100× too slowly (zero flips in 100 steps — decorative-binary trap); flip-lr sweep {3, 10, 30} replaces the MLP's {0.3, 1, 3}. Kill criterion unchanged: ED-int8 val loss >20% above BitNet-latent after lr sweep at two widths → pivot writeup.

## ADR-011 — All logged results use JSONL (2026-07-14)

**Context:** PI directive: machine-parseable result logs instead of .txt console captures.
**Decision:** Both harnesses natively append one JSON record per result/eval (`run.py report()` → `results/mlp.jsonl` default; `train_gpt.py` → `results/gpt.jsonl` default; per-sweep override via `LOG` env, set in every script). stdout stays human-readable for monitors. Historical .txt logs converted via `tools/convert_results.py` (parsed fields + verbatim `raw` line per record) and removed from the tree; originals remain in git history.
**Consequences:** Analysis scripts consume records instead of regex-scraping console text; every record carries its full config, so results files are self-describing.

## ADR-012 — Scoped gradient quantization (QSCOPE=w) (2026-07-14)

**Context:** P3.4a's attribution control showed quantized backward hooks cost the FP remnant 0.65 nats — most of the apparent 1-bit-gradient failure. The hook quantizes the shared backward operand, so both grad_W (binary layers) and grad_input (everything upstream) receive degraded signal.
**Decision:** `QSCOPE=w` routes through a custom autograd function: grad_W is computed from the quantized gradient (the binary layers' update path keeps its 1-bit operand), grad_input from the clean gradient. `QSCOPE=all` (default) preserves the original shared-operand semantics.
**Consequences:** Honest claim-narrowing: under scoping, only the weight-gradient matmul has a 1-bit gradient operand; the input-gradient matmul is FP-gradient × binary-weight. In exchange the FP remnant trains on clean signal. This is the standard trade in gradient-compression literature (compress what feeds the optimizer, not the backprop chain); the full shared-operand claim remains available via QSCOPE=all at its measured cost.

## ADR-013 — Second-order ΔΣ buffer (ed2) as a cascaded-integrator variant (2026-07-15)

**Context:** The ED buffer is a first-order sigma-delta (ΔΣ) modulator in time: the update stream is the input, the flip the 1-bit output, the buffer the integrator, the identity kernel the noise-shaping filter. ΔΣ theory says a second-order loop (two cascaded integrators) removes more quantization noise from the low-frequency band at the same output rate. Halftoning's spatial kernel is the same structure in space; its literal transplant (edS) failed, but the *order* of the temporal loop is untested.
**Decision:** `ed2_step`: CIFB form with unit coefficients — t1 = e1 + u; t2 = e2 + t1; flip on t2 (opposing sign, |t2| ≥ 1); realized movement subtracted from both integrators; both clamped and SR-quantized to the same bit width. State: 1 + 2b bits/wt, so comparisons run at matched total bits against a single deeper buffer (int4+int4 vs int8; int2+int2 vs int4). Clamping both integrators is the stability limiter standard in second-order ΔΣ designs.
**Consequences:** Directly tests whether noise-shaping order, not just buffer depth, matters — a different question from the edM budget split (edM's second state integrates the gradient; ed2's integrates the residual). If ed2 at 2+2 bits beats int4, the exchange curve's low-bit end improves; if not, the first-order loop joins momentum as a confirmed non-buyer of state bits.

## ADR-014 — Full checkpoint/resume and registered stall-detect in train_gpt (2026-08-10)

**Context:** The S2 gate checklist left full resume open (the deploy checkpoint saves signs + scales + FP remnant only); one overnight block was lost to a Windows Update. S2.9 registered the ignition-stall signature — val rising between steps 1k and 2k — with a ~40% stall rate at 6L/512, and S2.13 showed a thrash episode that passed the 2k check, requiring a second check at 3k. Both gate the flagship run.
**Decision:** `train_gpt.py` gains `FULLCK` (autosave at every eval: `model.state_dict()` including ED buffers, AdamW state, run-generator state, global torch/CUDA RNG states, step, best val) and `RESUME` (continue from the saved step on the saved streams). `STALLCHECK=1` applies the registered rule at steps 1000/2000/3000: if val rose over the previous checkpoint, log a `stall` record and exit with code 3; wrapper scripts retry with the seed shifted by +100, at most 3 attempts (`scripts/parity.sh`). `EVAL_EVERY` must divide 1000 for the checks to fire. `EVAL0` is skipped on resume so it cannot overwrite the resumed best.
**Consequences:** Resume is bit-identical to an uninterrupted run on CPU (verified: 40-step straight vs 20+resume-20 match exactly); on CUDA it differs only by kernel non-associativity. `ADAPT` runs lose the per-module `gnorm` EMA on resume (re-warms in a few steps; the flagship protocol uses adapt=0). The stall exit path is stub-verified at the wrapper level; the in-run branch has not been exercised by a real stall yet (no 6L/384 stall observed to date).

### Addendum (ADR-014) — divergence guard (2026-08-10)

The parity runs exercised the stall branch in production (four detections) and exposed its coverage hole: at 6L/384 on char-shakespeare, lr 250 passed the 3k check and thrashed at 4–5k. `DIVERGE` env (default 0 = off) adds a horizon-wide guard: at any eval with step ≥ 2000, val exceeding the run's best by more than `DIVERGE` nats logs a `stall` record and exits 3. `map_location` in `load_full` changed to `"cpu"`: the run generator's `set_state` requires a CPU ByteTensor, so checkpoints stage through host memory and the loaders move weights/optimizer state back to the device.

## ADR-015 — Optional dropout, off by default (2026-08-10)

**Context:** The parity benchmark targets nanoGPT's published shakespeare_char result (val 1.4697 at 6L/384, block 256), which is trained with dropout 0.2. This GPT had no dropout; on the 1.1MB char corpus that is a recipe handicap unrelated to binarization (the harness's own FP32 arm converges at 1.591, behind the same target).
**Decision:** `DROPOUT` env (default 0.0) adds dropout in nanoGPT's positions: after the token+position embedding sum, inside SDPA (`dropout_p`), and on both residual projections. Dropout masks draw from the device's global RNG stream, seeded with `SEED` — the one exception to the CPU-generator contract (ADR-003), because per-mask CPU draws plus transfer would add ~40% step time at 6L/384. Global RNG states ride in the full checkpoint (ADR-014), so dropout runs also resume exactly.
**Consequences:** Every recorded number is unaffected (default 0 adds inert `nn.Dropout` modules with no parameters and no RNG draws). Dropout runs are deterministic per device but their mask streams differ between CPU and CUDA, unlike everything else in the harness; comparisons involving `DROPOUT>0` must therefore hold the device fixed.
