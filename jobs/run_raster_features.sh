#!/bin/bash
#SBATCH --account=csl
#SBATCH --partition=csl
#SBATCH --job-name=pirineus_raster
#SBATCH --output=logs/pirineus_raster_%j.out
#SBATCH --error=logs/pirineus_raster_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail

source jobs/common.sh

RUN_CONFIG="${1:-configs/runs/pallars_worldclim_100m.yaml}"

echo "=============================="
echo "Running Pirineus Raster dataset pipeline"
echo "Run config: $RUN_CONFIG"
echo "=============================="

pirineus-raster run "$RUN_CONFIG"

echo "Pirineus Raster job finished successfully"
echo "Date: $(date)"