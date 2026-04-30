#!/bin/bash
#SBATCH --account=csl
#SBATCH --partition=csl
#SBATCH --job-name=pirineus_source_debug
#SBATCH --output=logs/pirineus_source_debug_%j.out
#SBATCH --error=logs/pirineus_source_debug_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail

source jobs/common.sh

SOURCE_CONFIG="${1:?Usage: bash jobs/run_source_debug.sh <source-config> [stage]}"
STAGE="${2:-build}"
PROJECT_CONFIG="${3:-configs/project.yaml}"

pirineus-raster run-source \
  --project-config "$PROJECT_CONFIG" \
  --source-config "$SOURCE_CONFIG" \
  --stage "$STAGE"