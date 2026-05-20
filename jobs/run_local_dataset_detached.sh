#!/bin/bash

set -euo pipefail

RUN_CONFIG="${1:-configs/runs/pallars_exp1_100m.yaml}"
CONDA_ENV="${CONDA_ENV:-pirineus-raster}"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
LOG_DIR="${LOG_DIR:-$PROJECT_DIR/logs/local}"
GDAL_NUM_THREADS="${GDAL_NUM_THREADS:-1}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

mkdir -p "$LOG_DIR"

RUN_BASENAME="$(basename "$RUN_CONFIG")"
RUN_NAME="${RUN_BASENAME%.*}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/${RUN_NAME}_${STAMP}.log"
PID_FILE="$LOG_DIR/${RUN_NAME}_${STAMP}.pid"

export PROJECT_DIR
export CONDA_ENV
export RUN_CONFIG
export GDAL_NUM_THREADS
export OMP_NUM_THREADS
export OPENBLAS_NUM_THREADS
export MKL_NUM_THREADS
export NUMEXPR_NUM_THREADS

nohup bash -lc '
set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
cd "$PROJECT_DIR"

echo "=============================="
echo "Pirineus Raster local detached run"
echo "Project dir: $PROJECT_DIR"
echo "Conda env:   $CONDA_ENV"
echo "Run config:  $RUN_CONFIG"
echo "Python:      $(which python)"
python --version
echo "Threads:     GDAL=$GDAL_NUM_THREADS OMP=$OMP_NUM_THREADS OPENBLAS=$OPENBLAS_NUM_THREADS MKL=$MKL_NUM_THREADS NUMEXPR=$NUMEXPR_NUM_THREADS"
echo "Started at:  $(date)"
echo "=============================="

if command -v ionice >/dev/null 2>&1; then
  ionice -c2 -n7 nice -n 10 pirineus-raster run "$RUN_CONFIG"
else
  nice -n 10 pirineus-raster run "$RUN_CONFIG"
fi

echo "=============================="
echo "Finished at: $(date)"
echo "=============================="
' > "$LOG_FILE" 2>&1 &

PID="$!"
echo "$PID" > "$PID_FILE"
disown "$PID" 2>/dev/null || true

echo "Started detached Pirineus Raster run"
echo "PID:  $PID"
echo "Log:  $LOG_FILE"
echo "PID file: $PID_FILE"
echo
echo "Watch logs:"
echo "  tail -f \"$LOG_FILE\""
echo
echo "Check process:"
echo "  ps -p $PID -o pid,etime,pcpu,pmem,cmd"
