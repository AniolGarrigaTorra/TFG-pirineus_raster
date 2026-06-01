#!/bin/bash
#SBATCH --job-name=pyrenees_copernicus_worldclim
#SBATCH --account=csl
#SBATCH --partition=csl
#SBATCH --output=/home/usuaris/csl/aniol.garriga.torra/logs/jobs/pyrenees_copernicus_worldclim_%j.out
#SBATCH --error=/home/usuaris/csl/aniol.garriga.torra/logs/jobs/pyrenees_copernicus_worldclim_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail

REPO_DIR="/mnt/csl/work/aniol.garriga.torra/pirineus_raster"
RUN_CONFIG="configs/runs/pyrenees_full_copernicus_worldclim_100m.yaml"

echo "======================================"
echo "Job started at: $(date)"
echo "Job ID: ${SLURM_JOB_ID:-no_slurm_id}"
echo "Host: $(hostname)"
echo "Submit dir: ${SLURM_SUBMIT_DIR:-unknown}"
echo "Repo dir: $REPO_DIR"
echo "======================================"

echo "Moving to repo..."
cd "$REPO_DIR"

echo "Current directory:"
pwd

echo "Using pirineus-raster conda environment directly..."

ENV_DIR="/home/usuaris/csl/aniol.garriga.torra/.conda/envs/pirineus-raster"

if [ ! -d "$ENV_DIR" ]; then
  echo "ERROR: Conda environment directory not found: $ENV_DIR"
  exit 1
fi

export PATH="$ENV_DIR/bin:$PATH"
export PYTHONNOUSERSITE=1

echo "Environment dir: $ENV_DIR"
echo "Python executable:"
which python
python --version

echo "Pirineus raster executable:"
which pirineus-raster
pirineus-raster --help | head -40

echo "Python:"
which python
python --version

echo "Pirineus raster CLI:"
which pirineus-raster
pirineus-raster --help | head -40

echo "Checking WEkEO credentials..."
if [ ! -f "$HOME/.hdarc" ]; then
  echo "ERROR: ~/.hdarc not found. WEkEO HDA credentials are missing."
  exit 1
fi

echo "Checking run config..."
if [ ! -f "$RUN_CONFIG" ]; then
  echo "ERROR: Run config not found: $RUN_CONFIG"
  exit 1
fi

echo "Checking source configs..."
ls -lh configs/sources/copernicus/copernicus_clms_forest.yaml
ls -lh configs/sources/copernicus/copernicus_clms_grasslands.yaml
ls -lh configs/sources/copernicus/copernicus_clms_water_wetness.yaml
ls -lh configs/sources/copernicus/copernicus_clms_corine_land_cover.yaml

echo "Checking output directories..."
mkdir -p data_raw/copernicus
mkdir -p data_interim/copernicus
mkdir -p data_processed/features/copernicus
mkdir -p data_processed/datasets

echo "======================================"
echo "Launching pipeline"
echo "======================================"

pirineus-raster run "$RUN_CONFIG"

echo "======================================"
echo "Pipeline finished at: $(date)"
echo "Checking outputs..."
echo "======================================"

find data_raw/copernicus -maxdepth 5 -type f | sort || true
find data_processed/features/copernicus -maxdepth 8 -type f | sort || true
find data_processed/datasets/pyrenees_full_copernicus_worldclim_100m -maxdepth 5 -type f | sort || true

echo "======================================"
echo "Job completed successfully at: $(date)"
echo "======================================"
