#!/usr/bin/env bash
set -euo pipefail

TARGET_DIR="/juicefs-algorithm/data/IPT/zhuochu_yang/DiffSynth-Studio/models/WAFT"
TARGET="$TARGET_DIR/waft_dav2_a2_zero_shot.ckpt"
SOURCE="https://github.com/hmorimitsu/ptlflow/releases/download/weights1/waft_dav2_a2-zero_shot-4d51a008.ckpt"
DOWNLOAD_URL="${WAFT_DOWNLOAD_URL:-https://gh-proxy.com/$SOURCE}"
VIT_TARGET="$TARGET_DIR/vit_small_patch16_224_imagenet.safetensors"
VIT_URL="${WAFT_VIT_DOWNLOAD_URL:-https://hf-mirror.com/timm/vit_small_patch16_224.augreg_in21k_ft_in1k/resolve/main/model.safetensors}"

mkdir -p "$TARGET_DIR"
curl -L --retry 8 --retry-delay 3 --continue-at - -o "$TARGET" "$DOWNLOAD_URL"
curl -L --retry 8 --retry-delay 3 --continue-at - -o "$VIT_TARGET" "$VIT_URL"
chmod 0644 "$TARGET" "$VIT_TARGET"
echo "[done] $TARGET"
echo "[done] $VIT_TARGET"
