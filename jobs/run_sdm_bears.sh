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
OUTPUT_BASE="${2:-outputs/ursus_arctos_habitat_modelling}"
RUN_MODE="${3:-full}"

run_number=1
while ! mkdir "${OUTPUT_BASE}/run_${run_number}" 2>/dev/null; do
  run_number=$((run_number + 1))
done

OUTPUT_DIR="${OUTPUT_BASE}/run_${run_number}"
mkdir -p "$OUTPUT_DIR"/{tables,plots,maps,models,logs,intermediate}
EXECUTED_NOTEBOOK="${OUTPUT_DIR}/executed_notebook.ipynb"

if [[ "$RUN_MODE" == "smoke" ]]; then
  SMOKE_MODE=true
elif [[ "$RUN_MODE" == "full" ]]; then
  SMOKE_MODE=false
else
  echo "Run mode must be 'full' or 'smoke', received: $RUN_MODE" >&2
  exit 2
fi

exec > >(tee -a "$OUTPUT_DIR/logs/launcher.log") 2> >(tee -a "$OUTPUT_DIR/logs/launcher.err" >&2)

echo "=============================="
echo "Running Brown Bear SDM Pipeline"
echo "Notebook:   $NOTEBOOK"
echo "Output dir: $OUTPUT_DIR"
echo "Executed:   $EXECUTED_NOTEBOOK"
echo "Run mode:   $RUN_MODE"
echo "=============================="

mkdir -p logs

if command -v papermill >/dev/null 2>&1; then
  papermill "$NOTEBOOK" "$EXECUTED_NOTEBOOK" \
    -p OUTPUT_DIR "$OUTPUT_DIR" \
    -p SMOKE_MODE "$SMOKE_MODE" \
    -p N_BACKGROUND_TRAIN 10000 \
    -p N_BACKGROUND_TEST 2000 \
    -p RANDOM_SEED 42 \
    -p LOCAL_BUFFER_KM 25 \
    -p N_FINAL_BOOTSTRAP_REPLICATES 10 \
    -p N_EVALUATION_REPLICATES 5 \
    -p N_PAPER_VALIDATION_REPEATS 10 \
    -p RUN_TUNING true \
    -p STRICT_NESTED_TUNING true \
    -p N_RF_TUNING_ITER 12 \
    -p N_OPTUNA_TRIALS 20 \
    -p RUN_XGBOOST true \
    -p RUN_MAP_PREDICTION true \
    -p RUN_SHAP true \
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

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  for stream in out err; do
    source_log="logs/sdm_bears_${SLURM_JOB_ID}.${stream}"
    if [[ -f "$source_log" ]]; then
      cp "$source_log" "$OUTPUT_DIR/logs/slurm.${stream}"
    fi
  done
fi

echo "SDM Bears job finished successfully"
echo "Date: $(date)"
