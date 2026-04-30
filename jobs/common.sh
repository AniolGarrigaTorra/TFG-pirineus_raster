#!/bin/bash

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/mnt/csl/work/aniol.garriga.torra/pirineus_raster}"
CONDA_ENV="${CONDA_ENV:-pirineus-raster}"

cd "$PROJECT_DIR"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

mkdir -p logs

echo "=============================="
echo "Pirineus Raster job environment"
echo "Project dir: $PROJECT_DIR"
echo "Conda env:   $CONDA_ENV"
echo "Python:      $(which python)"
python --version
echo "Date:        $(date)"
echo "=============================="