#!/usr/bin/env bash
# Direct entry: bash scripts/train_slurm.sh
set -euo pipefail

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
PROJECT_ROOT="$(cd "$(dirname "$SCRIPT_PATH")/.." && pwd -P)"
CONFIG="${QKWAFT_CONFIG:-$PROJECT_ROOT/configs/qwen_qk_waft.yaml}"
CONTAINER_IMAGE="${QKWAFT_CONTAINER_IMAGE:-docker://registry.intsig.net/zhuochu_yang/diffsynth:v2-diffusers}"
RUN_ROOT="${QKWAFT_RUN_ROOT:-$PROJECT_ROOT/runs/qwen_qk_waft}"
PHASES="${QKWAFT_PHASES:-audit,probe,confidence,b,c,d}"

if [[ "${1:-}" != "--inside" ]]; then
    if ! command -v srun >/dev/null 2>&1; then
        echo "srun is unavailable on this host; run this command from a Slurm login node." >&2
        exit 69
    fi
    export QKWAFT_CONFIG="$CONFIG"
    export QKWAFT_RUN_ROOT="$RUN_ROOT"
    export QKWAFT_PHASES="$PHASES"
    export HF_HOME="${HF_HOME:-/juicefs-algorithm/data/IPT/yuang_feng/cache}"
    exec srun -K \
        --job-name=qwen-qk-waft \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task=64 \
        --gres=gpu:8 \
        --mem=0 \
        --time=14-00:00:00 \
        --container-image="$CONTAINER_IMAGE" \
        --container-mounts=/juicefs-algorithm:/juicefs-algorithm \
        --container-workdir="$PROJECT_ROOT" \
        --container-env=QKWAFT_CONFIG,QKWAFT_RUN_ROOT,QKWAFT_PHASES,HF_HOME \
        bash "$SCRIPT_PATH" --inside
fi

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
if has_phase audit; then
    "$PYTHON" -m qwen_qk_waft.audit --config "$CONFIG"
fi
if has_phase probe; then
    "${TORCHRUN[@]}" --standalone --nproc_per_node=8 -m qwen_qk_waft.probe --config "$CONFIG"
fi
if has_phase confidence; then
    "${TORCHRUN[@]}" --standalone --nproc_per_node=8 -m qwen_qk_waft.train --config "$CONFIG" --phase confidence
fi
for phase in b c d; do
    if has_phase "$phase"; then
        "${TORCHRUN[@]}" --standalone --nproc_per_node=8 -m qwen_qk_waft.train --config "$CONFIG" --phase "$phase"
    fi
done

date -Iseconds > "$RUN_ROOT/_SUCCESS"
echo "[done] $RUN_ROOT/_SUCCESS"
