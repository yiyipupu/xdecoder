#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
source deployment/env.sh
"$XD_PYTHON" deployment/smoke_forward.py --method xdecoder --repo "$XD_ROOT" --weights "$XD_STORAGE/weights/xdecoder_focalt_last.pt" --image inference/images/fruit.jpg --text 'the larger watermelon' --short-side 512 --output "$XD_ROOT/outputs/forward"
