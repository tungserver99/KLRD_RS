#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export PYTHONUNBUFFERED=1
output="${KLRD_RS_FULL_OUTPUT:-experiment3_full/run_001}"

echo 'E3 full [1/6]: fresh-adapter 10k/1k training, or resume this full run.'
python main_exp3_full.py --mode train --output "$output"
echo 'E3 full [2/6]: matched 512-token pilot/full comparison.'
python main_exp3_full.py --mode evaluate --max-new-tokens 512 --output "$output"
echo 'E3 full [3/6]: new 1024-token clean/teacher/student evaluation.'
python main_exp3_full.py --mode evaluate --max-new-tokens 1024 --output "$output"
echo 'E3 full [4/6]: ARC-Challenge on clean/teacher/pilot/full.'
python main_exp3_full.py --mode arc-c --output "$output"
echo 'E3 full [5/6]: GPT-5 high_fidelity_v1 paraphrase; up to 1200 paid requests, successes reused.'
python main_exp3_full.py --mode paraphrase --max-new-tokens 1024 --output "$output"
echo 'E3 full [6/6]: six result tables, report and private-safe review ZIP.'
python main_exp3_full.py --mode report --output "$output"
