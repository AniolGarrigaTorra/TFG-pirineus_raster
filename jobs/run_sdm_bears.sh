#!/bin/bash
#SBATCH --account=csl
#SBATCH --partition=csl
#SBATCH --job-name=sdm_bears
#SBATCH --output=logs/sdm_bears_%j.out
#SBATCH --error=logs/sdm_bears_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G

set -euo pipefail
export PS1="${PS1:-}"
source jobs/common.sh

NOTEBOOK="${1:-notebooks/ursus_arctos_project/ursus_arctos_habitat_modelling.ipynb}"
OUTPUT_DIR="${2:-outputs/ursus_arctos_habitat_modelling}"
EXECUTED_NOTEBOOK="${OUTPUT_DIR}/sdm_bears_executed_$(date +%Y%m%d_%H%M%S).ipynb"

echo "=============================="
echo "Running Brown Bear SDM Pipeline"
echo "Notebook:   $NOTEBOOK"
echo "Output dir: $OUTPUT_DIR"
echo "Executed:   $EXECUTED_NOTEBOOK"
echo "=============================="

mkdir -p "$OUTPUT_DIR" logs

if command -v papermill >/dev/null 2>&1; then
  papermill "$NOTEBOOK" "$EXECUTED_NOTEBOOK" \
    -p OUTPUT_DIR "$OUTPUT_DIR" \
    -p N_OPTUNA_TRIALS 100 \
    -p N_BACKGROUND 10000 \
    -p RANDOM_SEED 42 \
    -p RUN_TUNING true \
    -p RUN_MAP_PREDICTION true \
    -p RUN_UNCERTAINTY_MAPS true \
    -p RUN_SHAP true \
    -p RUN_OBS_PARALLEL_MODELS false \
    --kernel python3 \
    --log-output
elif command -v jupyter >/dev/null 2>&1; then
  echo "papermill is not installed; falling back to jupyter nbconvert without parameter injection."
  jupyter nbconvert \
    --to notebook \
    --execute "$NOTEBOOK" \
    --output "$(basename "$EXECUTED_NOTEBOOK")" \
    --output-dir "$OUTPUT_DIR"
else
  echo "Neither papermill nor jupyter is available in the active conda environment." >&2
  echo "Install papermill for the parameterized SLURM workflow." >&2
  exit 1
fi

echo "SDM Bears job finished successfully"
echo "Date: $(date)"
