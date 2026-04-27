#!/bin/bash
#SBATCH --account=csl
#SBATCH --partition=csl
#SBATCH --job-name=make_grid
#SBATCH --output=logs/make_grid_%j.out
#SBATCH --error=logs/make_grid_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

set -euo pipefail

PROJECT_DIR="$HOME/projects/pirineus_raster"
cd "$PROJECT_DIR"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pirineus-raster

RESOLUTION="${1:-100}"

python -m src.make_grid --resolution "$RESOLUTION"
python -m src.validation.validate_grid --resolution "$RESOLUTION"