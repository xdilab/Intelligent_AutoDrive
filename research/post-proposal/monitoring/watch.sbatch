#!/bin/bash
#SBATCH --job-name=research-watch
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=7-00:00:00
#SBATCH --output=/work/bbyrd1/research-monitor/watch-%j.log
set -euo pipefail
python3 -u /work/bbyrd1/research-monitor/code/watch.py
