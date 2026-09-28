#!/bin/bash
#SBATCH --job-name=full56-epoch1-resume
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --partition=gpu-hp
#SBATCH --qos=ncat_h200_hp
#SBATCH --gres=gpu:h200:1
#SBATCH --time=7-00:00:00
set -euo pipefail
parent="$1"; stage="$2"; condition="$3"
export ROAD_FRAME_ROOT=/work/bbyrd1/stage56-full-20260914/frames
export HF_HOME=/work/bbyrd1/road_crop/hf_cache
export HF_HUB_OFFLINE=1
export PYTHONPATH=/work/bbyrd1/bddx-pilot-20260911/packages
export PATH=/work/bbyrd1/road_crop/py311/bin:$PATH
set +e
python - "$stage" "$condition" <<'PY'
import sys,json
from pathlib import Path
p=Path('/work/bbyrd1/stage56-full-20260914/results')/('epoch1-final-'+sys.argv[1]+'-'+sys.argv[2]+'.json')
if p.exists() and json.loads(p.read_text()).get('n_frames')==36717:print('ALREADY_COMPLETE',p);sys.exit(0)
sys.exit(10)
PY
check=$?
set -e
if [ "$check" -eq 0 ]; then exit 0; fi
if [ "$check" -ne 10 ]; then exit "$check"; fi
state=$(sacct -X -n -P -j "$parent" --format=State | head -n 1 | cut -d '|' -f 1)
case "$state" in TIMEOUT*|PREEMPTED*) ;; *) echo "Refusing automatic retry for unexplained state: $state"; exit 1;; esac
python /work/bbyrd1/stage56-full-20260914/code/guard.py
python -u /work/bbyrd1/stage56-full-20260914/code/evaluate-epoch1.py --stage "$stage" --condition "$condition"
