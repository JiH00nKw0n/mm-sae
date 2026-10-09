#!/usr/bin/env bash
set -euo pipefail
source /home/elicer/jihoonkwon/mm_sae_grahtp_20261006/config/env.sh
exec "$TASK_ROOT/env/bin/python" -u -m experiments.oracle_sets.run_pursuit --config "$TASK_ROOT/config/oracle-pursuit-80g.yaml"
