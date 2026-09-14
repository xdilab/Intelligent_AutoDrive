#!/bin/bash
#SBATCH --job-name=bddx-two-tower-pilot
#SBATCH --partition=gpu-hp
#SBATCH --qos=ncat_h200_hp
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/work/bbyrd1/bddx-pilot-20260911/logs/pilot-%j.log
set -euo pipefail
root=/work/bbyrd1/bddx-pilot-20260911
export HF_HOME=/work/bbyrd1/road_crop/hf_cache
export HF_HUB_OFFLINE=1
export PYTHONPATH="$root/packages"
python=/work/bbyrd1/road_crop/py311/bin/python
"$python" "$root/code/prepare.py" --videos "$root/data/BDDX_Processed/videos" --conversations "$root/data/bddx-conversation.json" --annotations "$root/data/BDD-X-Annotations_v1.csv" --train-split "$root/data/train.txt" --out "$root/manifest.json"
"$python" -u "$root/code/train.py" --manifest "$root/manifest.json" --out "$root/smoke-data" --smoke
"$python" -u "$root/code/train.py" --manifest "$root/manifest.json" --out "$root/run-seed0" --epochs 3 --batch-size 16
