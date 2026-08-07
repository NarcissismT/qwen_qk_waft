#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

docker run --rm \
    --user "$(id -u):$(id -g)" \
    -v /juicefs-algorithm:/juicefs-algorithm \
    registry.intsig.net/zhuochu_yang/diffsynth:v2-diffusers \
    /bin/bash -lc "cd '$(pwd -P)'; PYTHONPATH=src:. /usr/bin/python scripts/run_tests.py"
