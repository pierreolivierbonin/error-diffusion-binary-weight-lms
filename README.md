# Error Diffusion Replaces Latent Weights

Code, run scripts, and logs for the paper *Error Diffusion Replaces Latent Weights: Training Binary-Weight Language Models on a Few Bits of Dithered State* (Pierre-Olivier Bonin, SLM-Agents: 1st Workshop on SLMs for Agentic Systems, NeurIPS 2026; [OpenReview](https://openreview.net/forum?id=ZtrJISn7LJ)).

The error-diffusion (ED) optimizer trains binary-weight networks without a latent full-precision weight. Each binary weight keeps a 2-8-bit buffer, quantized by stochastic rounding, that accumulates the unrealized part of every update and flips the weight when the buffer crosses a threshold.

| directory | contents |
|---|---|
| `bitdiffusion/` | the ED optimizer, the MLP and transformer harnesses, every run script named in the paper's reproducibility table, and the JSON Lines (JSONL) logs behind every reported number |
| `bitedge/` | the porting experiments of the deploy-then-adapt section: the integer update kernel, the fused update-in-backward path, measured whole-process memory budgets, and the serialized deployed artifacts (`artifact_256.pt`, `artifact_512.pt`, the latter being the 4 MB artifact quoted in the paper) |

## Branches

`main` holds the code and logs of the supplementary material reviewed on OpenReview. Each later experiment has its own branch with the exact code that produced its logs.

| branch | paper section | contents |
|---|---|---|
| `main` | Sections 3-5, Appendices A-E | MLP and transformer experiments, ladders, deploy-then-adapt, Figure 2 script |
| `r5-e1-int4-drift` | Section 5.4, Appendix F | buffer histogram and re-flip diagnostics (`SNAP`) |
| `r5-e2-stall-init` | Section 5.4, Appendix F | stall diagnostics, buffer init (`EDINIT`), the per-step update cap (`UCLIP`), the avalanche figure |
| `r5-e3-fp-remnant` | Section 5.4, Appendix F | full-precision remnant controls (`FREEZE`, `ADAMSCHED`) and the capped ladders |
| `r5-e4-bop-transformer` | Section 5.4, Appendix F | Bop on the transformer (`bop` mode) |
| `r5-e5-int4-wikitext` | Section 5.4, Appendix F | 4-bit buffers on the capped wikitext-103 and enwik8 ladders |
| `r6-halftone-trace` | none (audit of Figure 2, Appendix A) | per-weight buffer trace (`TRACE`) on real training data, used to check the Figure 2 caption |

## Research records

`bitdiffusion/docs/` and `bitedge/docs/` hold the dated planning and results records that the paper's table of pre-registered predictions cites: `ADR.md`, `RESULTS_s2.md`, `scaling-plan.md`, and the companion `plan.md`. They are kept as written at the time, so later results can supersede them; the paper is the reference for the final numbers. `bitedge/docs/plan.md` omits one planned follow-up experiment, as its release note states.

## Layout of `bitdiffusion/`

| path | contents |
|---|---|
| `src/bitdiffusion/tasks.py` | teacher-student dataset (frozen seed 1234) |
| `src/bitdiffusion/layers.py` | MLP forward path: binary weights, optional sign activations with a straight-through estimator, optional 1-bit gradient quantization (`det1`, `sr1`, `sr1_rht`) |
| `src/bitdiffusion/optim.py` | update rules in flip units: `ed_step` (edA, edB, edB_det, sr), `ed2_step`, `ed_spatial_step`, `ed_hyst_step`, `edm_step`, `ede_step`, `intlat_step`, `cnt_step`, `bop_step` |
| `src/bitdiffusion/run.py` | MLP harness and command line; logs to `results/mlp.jsonl` by default |
| `src/bitdiffusion/gpt.py` | minimal GPT with `BitLinear` (modes fp32, bitnet, ed), the ED update loop, and the state accountant |
| `src/bitdiffusion/train_gpt.py` | transformer harness and command line: datasets, evaluation loop, deploy checkpoints; logs to `results/gpt.jsonl` by default |
| `scripts/*.sh` | recorded experiment sweeps; each sets its own `LOG` target under `results/` |
| `results/*.jsonl` | one JSON record per result or evaluation, each carrying its full configuration |
| `results/*.out` | captured standard output of the same runs |
| `tools/` | data fetch (`fetch_wt103.py`), figure and table generation, log conversion |

## Layout of `bitedge/`

| path | contents |
|---|---|
| `src/bitedge/kernel.py` | the ED update in integer form (`ed_step_float`, `srq_float`) with stochastic-rounding bits from a seeded `xorshift_seed` generator |
| `src/bitedge/gpt2e.py` | transformer over bit-packed binary weights (`pack_bits`, `unpack_bits`, `STEsignPacked`) |
| `src/bitedge/adam8.py` | block-quantized 8-bit Adam for the full-precision remnant |
| `src/bitedge/e0.py` to `e3.py` | experiment ladder; `e3.py` is the deploy-then-adapt run under a measured memory cap, and `FUSED=1` selects the update-in-backward path that stores no weight gradients |
| `results/e*.jsonl`, `results/*.out` | logs of the ladder |
| `tools/slim_artifact.py` | serializes a deployed artifact (signs and scales only) from a training checkpoint |

## Running

Both packages use `uv` (https://docs.astral.sh/uv/) with a locked environment (`uv.lock`); `uv run` inside a package directory resolves it. Python 3.12 or newer and PyTorch 2.9 or newer are required. Every run takes a seed. Runs with the same seed can still differ slightly on CUDA, because some GPU kernels are not deterministic.

Examples (from `bitdiffusion/`):

```
uv run python -m bitdiffusion.run one edB bits 4 binact lr 1.0     # one MLP cell
DATA=enwik8 EMBD=256 LAYERS=4 STEPS=15000 EDLR=30 BITS=4 uv run python -m bitdiffusion.train_gpt
bash scripts/s2int4gpt3.sh                                          # a full recorded sweep
```

The transformer harness reads `DATA` (`shakespeare`, `shakespeare_bytes`, `enwik8`, `wikitext103`) and the hyperparameters as environment variables; each script in `scripts/` shows the exact settings used.

`bitedge/src/bitedge/e1.py` to `e3.py` read the pretrained latent-weight checkpoint and the data from `PARENT`, which defaults to `../bitdiffusion` (the layout of this archive). The 19M-parameter checkpoint `ck_enwik8_bitnet_512.pt` is not included (92 MB). Its run is logged in `bitdiffusion/results/ck512.out` and is regenerated from `bitdiffusion/` with

```
DATA=enwik8 EMBD=512 LAYERS=6 STEPS=15000 SEED=0 SAVECK=results/ck_enwik8_bitnet_512.pt uv run python -m bitdiffusion.train_gpt bitnet
```

(`bitdiffusion/scripts/s2adapt1.sh` shows the same pattern at 3.3M parameters.)

## Datasets (not included)

Place these under `bitdiffusion/data/`:

- `shakespeare.txt`: the Tiny Shakespeare text (1.1 MB), https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
- `enwik8`: the first 10^8 bytes of English Wikipedia, http://mattmahoney.net/dc/enwik8.zip (unzip to `data/enwik8`)
- `wt103_gpt2.bin`: wikitext-103 tokenized with the GPT-2 byte-pair encoding; `uv run python tools/fetch_wt103.py` downloads the parquet files from the Hugging Face `Salesforce/wikitext` dataset and writes the binary

## Mapping to the paper

The reproducibility table in the paper's appendix maps each figure, table, and paragraph of the main text to the script in `bitdiffusion/scripts/` and the log in `bitdiffusion/results/` that produced it. The porting numbers in the deploy-then-adapt section (artifact size, the 275 MB training budget, the memory of latent-weight continuation, and the adapted validation loss) come from `bitedge/results/e3.jsonl` and `bitedge/results/e3edge275.out`, `e3fused.out`, and `e3final.out`.

## License and citation

The code is released under the MIT License (`LICENSE`). To cite the paper, use `CITATION.cff` or the "Cite this repository" button on GitHub.
