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

source jobs/common.sh

RESOLUTION="${1:-100}"
AOI_CONFIG="${2:-configs/aoi/experimental_pallars_sobira.yaml}"
PROJECT_CONFIG="${3:-configs/project.yaml}"

python -m src.make_grid \
  --project-config "$PROJECT_CONFIG" \
  --aoi-config "$AOI_CONFIG" \
  --resolution "$RESOLUTION"

python -m src.validation.validate_grid \
  --project-config "$PROJECT_CONFIG" \
  --aoi-config "$AOI_CONFIG" \
  --resolution "$RESOLUTION"