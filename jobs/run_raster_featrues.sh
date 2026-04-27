#!/bin/bash
#SBATCH --account=csl
#SBATCH --partition=csl
#SBATCH --job-name=climate_features
#SBATCH --output=logs/climate_features_%j.out
#SBATCH --error=logs/climate_features_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

echo "=============================="
echo "Starting climate features job"
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

python -m src.features.build_climate_features

echo "Climate features finished successfully"
echo "Date: $(date)"