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
WAFT loading, Stage-A parity, Qwen LoRA schema, 2-GPU BF16 DDP, and the GT map
coordinate contract. It then runs the planned phases in order:

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

The optional Qwen-unfreezing experiment is deliberately not part of the
default run.  The plan says it is only justified after the frozen descriptor
run has converged and demonstrated a Qwen feature bottleneck.

## Direct Slurm launch

From this directory on a Slurm login node:

```bash
bash scripts/train_slurm.sh
```

The script requests one node with eight GPUs, enters the established
`diffsynth:v2-diffusers` container, launches eight independent frozen-Qwen
workers with `torchrun`, and writes:

- log: `runs/qwen_qk_waft/train.log`
- official initialization report: `runs/qwen_qk_waft/official_waft_initialization.json`
- Stage-A strict-load/parity report: `runs/qwen_qk_waft/stage_a_initialization.json`
- Qwen LoRA schema report: `runs/qwen_qk_waft/qwen_lora_initialization.json`
- 2-GPU BF16 DDP report: `runs/qwen_qk_waft/ddp_bfloat16_preflight.json`
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

Unit tests cover coordinate contracts, official zero-padding, runtime Qwen
token segmentation, complete forward/loss/gradient flow, Phase-B/C/D function
continuity, Q/K margin and cycle consistency, official zero-flow A2 equations,
and expanded evaluation metrics. The current line regions are deterministic
target-image structure masks because the manifests do not contain annotated
text-line masks. Formal OCR, annotated-line, and visual-quality acceptance
still require the Slurm outputs and result review.
