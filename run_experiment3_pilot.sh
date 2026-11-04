#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

python main_exp3.py --mode pilot
echo 'E3 pilot: training/merge complete; starting student evaluation.'
python main_exp3.py --mode evaluate
