#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
source deployment/env.sh
/home/pupu/miniconda3/envs/instructpart-xdecoder/bin/python deployment/smoke_forward.py --method xdecoder --repo "$XD_ROOT" --weights "$XD_STORAGE/weights/xdecoder_focalt_last.pt" --image inference/images/fruit.jpg --text 'the larger watermelon' --short-side 512 --output "$XD_ROOT/outputs/forward"
