# Qwen-QK-WAFT document rectifier

This directory is an isolated implementation of
`Qwen-QK-WAFT_model_architecture_training_plan.md`.  The public model predicts
an absolute target-to-warped-source backward map.  Qwen never decodes the final
RGB; inference performs one native-resolution `grid_sample` on the original
warped image.

## Where to review the model design

The architecture is split at the same boundaries as the design document:

| Design element | Implementation |
|---|---|
| Runtime `img_shapes` token split, target Q, source K, pre/post RoPE and hidden probe | `src/qwen_qk_waft/models/qwen_qk.py` |
| Independent per-layer Q/K adapters and two official WAFT DPTHead instances | `src/qwen_qk_waft/models/dpt.py` |
| Existing deterministic Stage-A geometry plus calibrated confidence | `src/qwen_qk_waft/models/stage_a.py` |
| Non-zero warm-start around the official WAFT-A2 recurrent core, flow/info head, gate and official convex upsampling | `src/qwen_qk_waft/models/waft.py` |
| Official VisionTransformer, PatchEmbed, DepthAnythingV2 DPTHead and fusion blocks | `src/qwen_qk_waft/official_waft/` |
| Strict official checkpoint loading into ViT, DPT-Q/K and update heads | `src/qwen_qk_waft/models/waft_checkpoint.py` |
| Full forward and source-only local branch | `src/qwen_qk_waft/models/model.py` |
| Absolute-map sequence, reconstruction, edge, bending, fold, uncertainty and gate losses | `src/qwen_qk_waft/losses.py` |
| Native source-faithful one-sample inference | `src/qwen_qk_waft/infer.py` |
| FP32 coordinate/Jacobian contract | `src/qwen_qk_waft/geometry.py` |
| Annotated-line fitting and OCR character-retention evaluation | `src/qwen_qk_waft/metrics.py`, `src/qwen_qk_waft/formal_quality.py` |

The recurrent updater directly uses the modules and parameter contract of
the official [princeton-vl/WAFT](https://github.com/princeton-vl/WAFT)
`waftv2` implementation. See `OFFICIAL_WAFT_ALIGNMENT.md` for the source-level
mapping and the deliberate task adaptations. The accurate model name is
`Qwen-QK + independent DPT + Stage-A-initialized WAFT-A2 adaptation`; this is
not the unchanged official two-frame WAFT model.

The complete WAFT DAv2-A2 zero-shot checkpoint is stored at:

```text
/juicefs-algorithm/data/IPT/zhuochu_yang/DiffSynth-Studio/models/WAFT/vit_small_patch16_224_imagenet.safetensors
/juicefs-algorithm/data/IPT/zhuochu_yang/DiffSynth-Studio/models/WAFT/waft_dav2_a2_zero_shot.ckpt
```

Re-download it with `bash scripts/download_waft_weights.sh`. The Slurm entrypoint
runs `scripts/verify_official_waft.py` before any training phase and stops if a
ViT, DPTHead or update-head key does not load strictly.

## Training phases

`scripts/train_slurm.sh` first runs fail-closed preflight checks for official
WAFT loading, Stage-A full-model parity, Qwen LoRA scale semantics, 512/2K/4K
FP32 coordinate behavior, a real Qwen-to-WAFT forward, 2-GPU 512px BF16 DDP,
the GT map coordinate contract, and a real Phase-B `1 -> 3 -> 5` forward,
backward and optimizer preflight. It then runs the planned phases in order:

1. audit GT backward-map reconstruction;
2. compare base Qwen and LoRA scales `0.25/0.5/0.75/1.0`, all 60 layers, all
   four probe steps, pre-RoPE, post-RoPE, their concatenation, and hidden-state
   diagnostics over three fixed seeds; measure cycle consistency and
   text-structure/page-edge/corner/interior/background regions; select the
   formal top four Q/K layers;
3. calibrate the frozen Stage-A confidence head;
4. train Q/K adapters, independent DPT-Q/K and WAFT with a `1 -> 3 -> 5`
   iteration curriculum, with Qwen and Stage-A frozen;
5. enable the source local encoder;
6. enable and calibrate the confidence-protected residual gate.

All feature convolutions and ViT/DPT blocks may run in BF16. Stage-A map,
pixel grid, displacement, residual update, normalized sampling grid, convex
upsampling coordinates, Jacobian and bending/fold losses remain FP32. Each
gate iteration receives sequence-weighted supervision and has its own
calibration histogram. `best.pt` is written only when the configured geometry
criteria improve over Stage-A without increasing fold/invalid rates or
exceeding the high-confidence damage limit.

WAFT training uses the official OneCycle warmup and linear decay rather than
holding the peak learning rate for the whole run. Sequence and mixture-loss
weights are normalized across the `1 -> 3 -> 5` curriculum so an iteration
transition does not rescale the objective. Training logs the current step
loss, every loss term, gradient norm, learning rate and iteration count. A
non-finite loss or gradient is synchronized across ranks before the optimizer
step, written to `numerical_failure.json` with the responsible sample IDs, and
terminates the phase before parameters or Adam state can be contaminated.

The optional Qwen-unfreezing experiment is deliberately not part of the
default run.  The plan says it is only justified after the frozen descriptor
run has converged and demonstrated a Qwen feature bottleneck.

## Training entrypoint

Configure Slurm resources and the container in the outer job, then execute this
training payload inside that environment:

```bash
bash scripts/train_slurm.sh
```

The script contains no `srun`, resource request, or container configuration.
It launches the requested training phases with `torchrun` on the GPUs exposed
by the outer job. The default is eight workers; override it with
`QKWAFT_NPROC_PER_NODE` when needed. It writes:

- log: `runs/qwen_qk_waft/train.log`
- official initialization report: `runs/qwen_qk_waft/official_waft_initialization.json`
- Stage-A strict-load/parity report: `runs/qwen_qk_waft/stage_a_initialization.json`
- Qwen LoRA schema report: `runs/qwen_qk_waft/qwen_lora_initialization.json`
- 512/2K/4K precision report: `runs/qwen_qk_waft/coordinate_precision.json`
- real full-model report: `runs/qwen_qk_waft/full_model_preflight.json`
- 2-GPU BF16 DDP report: `runs/qwen_qk_waft/ddp_bfloat16_preflight.json`
- real Phase-B curriculum report: `runs/qwen_qk_waft/phase_b_stability_preflight.json`
- coordinate audit: `runs/qwen_qk_waft/phase0_audit.json`
- Q/K selection: `runs/qwen_qk_waft/phase1_probe/selection.json`
- final checkpoint: `runs/qwen_qk_waft/phase5_confidence_gate/best.pt`
- completion marker: `runs/qwen_qk_waft/_SUCCESS`

Resume an individual phase without rerunning earlier phases by setting, for
example:

```bash
QKWAFT_PHASES=d bash scripts/train_slurm.sh
```

## Inference

Inside the same container:

```bash
PYTHONPATH=src /usr/bin/python -m qwen_qk_waft.infer \
  --config configs/qwen_qk_waft.yaml \
  --checkpoint runs/qwen_qk_waft/phase5_confidence_gate/best.pt \
  --image /path/to/warped.png \
  --output-dir runs/qwen_qk_waft/inference
```

## Engineering validation

```bash
bash scripts/smoke_test.sh
```

Unit tests cover 512px BF16 coordinate uniqueness, zero identity folds,
official zero-padding, runtime Qwen token segmentation, original DiffSynth
LoRA `B@A` fusion, complete forward/loss/gradient flow, all-iteration gate
supervision, weighted bending, Phase-B/C/D continuity, Q/K margin/cycle,
official zero-flow A2 equations, and expanded metrics.

Validation manifests may provide `line_mask` and integer `line_instances` NPY
paths. Those switch line EPE from the image-gradient proxy to annotations and
enable affine text-baseline fitting. OCR output is evaluated offline without
coupling an OCR model to training. Each JSONL row contains `id`,
`reference_text`, `prediction_text`, and optionally `stage_a_text`:

```bash
PYTHONPATH=src /usr/bin/python -m qwen_qk_waft.formal_quality \
  --input runs/qwen_qk_waft/ocr_results.jsonl \
  --output runs/qwen_qk_waft/ocr_character_retention.json
```

When annotations/OCR results are absent, the report names the line metric as a
development proxy; it is not formal OCR or annotated-line evidence. Formal
convergence and visual acceptance still require the Slurm run and result
review.
