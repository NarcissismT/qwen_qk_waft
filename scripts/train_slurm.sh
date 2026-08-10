#!/usr/bin/env bash
# Training payload only. Slurm/container resources are configured externally.
set -euo pipefail

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
PROJECT_ROOT="$(cd "$(dirname "$SCRIPT_PATH")/.." && pwd -P)"
CONFIG="${QKWAFT_CONFIG:-$PROJECT_ROOT/configs/qwen_qk_waft.yaml}"
RUN_ROOT="${QKWAFT_RUN_ROOT:-$PROJECT_ROOT/runs/qwen_qk_waft_0810}"
PHASES="${QKWAFT_PHASES:-audit,probe,confidence,b,c,d}"
RUNTIME_PREFLIGHT="${QKWAFT_RUNTIME_PREFLIGHT:-1}"
NPROC_PER_NODE="${QKWAFT_NPROC_PER_NODE:-8}"
PREFLIGHT_NPROC="${QKWAFT_PREFLIGHT_NPROC:-2}"

cd "$PROJECT_ROOT"
mkdir -p "$RUN_ROOT"
exec > >(tee -a "$RUN_ROOT/train.log") 2>&1

export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-/juicefs-algorithm/data/IPT/yuang_feng/cache}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

PYTHON="${QKWAFT_PYTHON:-/usr/bin/python}"
TORCHRUN=("$PYTHON" -m torch.distributed.run)
IFS=',' read -r -a requested_phases <<< "$PHASES"

has_phase() {
    local wanted="$1"
    local value
    for value in "${requested_phases[@]}"; do
        [[ "$value" == "$wanted" ]] && return 0
    done
    return 1
}

echo "[run] root=$RUN_ROOT phases=$PHASES"
"$PYTHON" scripts/verify_official_waft.py \
    --config "$CONFIG" \
    --output "$RUN_ROOT/official_waft_initialization.json"
"$PYTHON" -m qwen_qk_waft.stage_a_audit --config "$CONFIG"
"$PYTHON" -m qwen_qk_waft.qwen_lora_audit --config "$CONFIG"
"$PYTHON" scripts/coordinate_precision_audit.py --config "$CONFIG"
if [[ "$RUNTIME_PREFLIGHT" == "1" ]]; then
    "$PYTHON" scripts/full_model_preflight.py --config "$CONFIG"
    "${TORCHRUN[@]}" --standalone --nproc_per_node="$PREFLIGHT_NPROC" \
        scripts/ddp_preflight.py --config "$CONFIG"
fi
if has_phase audit; then
    "$PYTHON" -m qwen_qk_waft.audit --config "$CONFIG"
fi
if has_phase probe; then
    "${TORCHRUN[@]}" --standalone --nproc_per_node="$NPROC_PER_NODE" \
        -m qwen_qk_waft.probe --config "$CONFIG"
fi
if has_phase confidence; then
    "${TORCHRUN[@]}" --standalone --nproc_per_node="$NPROC_PER_NODE" \
        -m qwen_qk_waft.train --config "$CONFIG" --phase confidence
fi
if [[ "$RUNTIME_PREFLIGHT" == "1" ]] && has_phase b; then
    "${TORCHRUN[@]}" --standalone --nproc_per_node="$PREFLIGHT_NPROC" \
        scripts/phase_b_stability_preflight.py --config "$CONFIG"
fi
for phase in b c d; do
    if has_phase "$phase"; then
        "${TORCHRUN[@]}" --standalone --nproc_per_node="$NPROC_PER_NODE" \
            -m qwen_qk_waft.train --config "$CONFIG" --phase "$phase"
    fi
done

date -Iseconds > "$RUN_ROOT/_SUCCESS"
echo "[done] $RUN_ROOT/_SUCCESS"
