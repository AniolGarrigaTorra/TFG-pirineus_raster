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

echo "Project directory: $PROJECT_DIR"

# Activate conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pirineus-raster

echo "Python: $(which python)"
python --version

echo "------------------------------"
echo "1. Creating base grid"
echo "------------------------------"
python -m src.make_grid

echo "------------------------------"
echo "2. Validating base grid"
echo "------------------------------"
python -m src.validation.validate_grid

echo "------------------------------"
echo "3. Building climate features"
echo "------------------------------"
python -m src.features.build_climate_features

echo "------------------------------"
echo "4. Validating final climate rasters"
echo "------------------------------"
python -m src.validation.validate_raster.py --raster data_processed/climate/climate_isothermality_may-sep_100m.tif
python -m src.validation.validate_raster.py --raster data_processed/climate/climate_temp_seasonality_may-sep_100m.tif
python -m src.validation.validate_raster.py --raster data_processed/climate/climate_precip_sum_may-sep_100m.tif

echo "=============================="
echo "Climate pipeline finished successfully"
echo "Date: $(date)"
echo "=============================="