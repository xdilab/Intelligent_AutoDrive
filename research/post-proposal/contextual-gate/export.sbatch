#!/bin/bash
#SBATCH --job-name=context-gate-export
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#SBATCH --output=/work/bbyrd1/contextual-gate-20260919/export-%j.log
set -euo pipefail
/work/bbyrd1/road_crop/py311/bin/python -u /work/bbyrd1/contextual-gate-20260919/export_gate.py
