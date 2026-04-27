#!/bin/bash
#SBATCH --account=csl
#SBATCH --partition=csl
#SBATCH --job-name=validate_rasters
#SBATCH --output=logs/validate_rasters_%j.out
#SBATCH --error=logs/validate_rasters_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

set -euo pipefail

PROJECT_DIR="$HOME/projects/pirineus_raster"
cd "$PROJECT_DIR"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pirineus-raster

RESOLUTION="${1:-100}"
RES_SUFFIX="${RESOLUTION}m"
CLIMATE_DIR="data_processed/climate/${RES_SUFFIX}"

python -m src.validation.validate_raster \
  --resolution "$RESOLUTION" \
  --raster "${CLIMATE_DIR}/climate_isothermality_may-sep_${RES_SUFFIX}.tif"

python -m src.validation.validate_raster \
  --resolution "$RESOLUTION" \
  --raster "${CLIMATE_DIR}/climate_temp_seasonality_may-sep_${RES_SUFFIX}.tif"

python -m src.validation.validate_raster \
  --resolution "$RESOLUTION" \
  --raster "${CLIMATE_DIR}/climate_precip_sum_may-sep_${RES_SUFFIX}.tif"