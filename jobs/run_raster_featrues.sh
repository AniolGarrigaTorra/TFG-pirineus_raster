#!/bin/bash
#SBATCH --account=csl
#SBATCH --partition=csl
#SBATCH --job-name=raster_features
#SBATCH --output=logs/raster_features_%j.out
#SBATCH --error=logs/raster_features_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail

echo "=============================="
echo "Starting raster features job"
echo "Job ID: ${SLURM_JOB_ID:-no_slurm}"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "=============================="

PROJECT_DIR="$HOME/projects/pirineus_raster"
cd "$PROJECT_DIR"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pirineus-raster

echo "Python: $(which python)"
python --version

python -m src.features.build_raster_features \
  --project-config configs/project.yaml \
  --source-config configs/sources/worldclim_v2_1_base.yaml \
  --stage download

echo "Raster features job finished successfully"
echo "Date: $(date)"