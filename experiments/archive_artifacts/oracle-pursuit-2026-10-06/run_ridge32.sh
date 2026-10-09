#!/usr/bin/env bash
set -euo pipefail
source /home/elicer/jihoonkwon/mm_sae_grahtp_20261006/config/env.sh
for condition in cc3m-coco coco-coco; do
 "$TASK_ROOT/env/bin/python" -u experiments/oracle_sets/run_pursuit.py --config "$TASK_ROOT/config/oracle-ridge32-$condition.yaml"
done
