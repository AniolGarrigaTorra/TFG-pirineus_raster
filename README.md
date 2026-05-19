# Pirineus Raster

Pirineus Raster is a configurable Python pipeline for building homogeneous
environmental raster datasets for the Pyrenees and smaller areas of interest.

The project is designed for researchers who need reproducible raster layers
with a common CRS, extent, resolution, transform and naming convention. Source
data can come from different providers, formats and coordinate systems, but the
final dataset is written as aligned GeoTIFF layers plus JSON metadata.

## Core Idea

Each source follows the same pipeline stages:

1. `download`: download or locate raw source data.
2. `clip`: prepare intermediate files clipped to a configured AOI.
3. `build`: align final features to the project grid and write metadata.

The target grid is explicit. Every final raster is validated against it for CRS,
shape and affine transform.

## Repository Layout

- `configs/project.yaml`: global CRS, paths, grid resolutions and naming defaults.
- `configs/aoi/`: areas of interest in project CRS.
- `configs/sources/`: provider/product source configurations.
- `configs/runs/`: complete dataset recipes combining several sources.
- `src/`: Python package and CLI implementation.
- `ui/`: React configuration workbench for building researcher-facing run YAMLs.
- `data_raw/`: original downloaded files.
- `data_interim/`: extracted, clipped or prepared intermediate data.
- `data_processed/features/`: provider-level final feature rasters.
- `data_processed/datasets/`: packaged run outputs with manifests.

## Environment

The project requires Python 3.11 or newer.

On the UPC/CSL system, the expected environment is usually:

```bash
conda activate pirineus-raster
```

For local development:

```bash
pip install -e .
```

Copernicus WEkEO downloads require HDA credentials, normally through `~/.hdarc`
or `HDA_USER` and `HDA_PASSWORD`.

## Basic Workflow

Create the reference grid for an AOI and resolution:

```bash
python -m src.make_grid \
  --project-config configs/project.yaml \
  --aoi-config configs/aoi/experimental_pallars_sobira.yaml \
  --resolution 100
```

Validate the grid:

```bash
python -m src.validation.validate_grid \
  --project-config configs/project.yaml \
  --aoi-config configs/aoi/experimental_pallars_sobira.yaml \
  --resolution 100
```

Run a complete dataset recipe:

```bash
pirineus-raster run configs/runs/pallars_worldclim_full_100m.yaml
```

Inspect the generated manifest:

```bash
pirineus-raster inspect data_processed/datasets/pallars_worldclim_full_100m
```

Validate all rasters in a generated dataset:

```bash
pirineus-raster validate-dataset data_processed/datasets/pallars_worldclim_full_100m
```

Use strict metadata validation for newly regenerated datasets:

```bash
pirineus-raster validate-dataset \
  data_processed/datasets/pallars_worldclim_full_100m \
  --strict-metadata
```

Run one source directly while developing:

```bash
pirineus-raster run-source \
  --project-config configs/project.yaml \
  --source-config configs/sources/worldclim/worldclim_v2_1_climate_normals.yaml \
  --stage build
```

Inspect source catalogs and validate a run config before processing data:

```bash
pirineus-raster catalog \
  --source-config configs/sources/worldclim/worldclim_cmip6_future.yaml

pirineus-raster validate-config \
  configs/runs/pallars_worldclim_cmip6_simplified_100m.yaml

pirineus-raster render-run \
  configs/runs/pallars_worldclim_cmip6_simplified_100m.yaml
```

## React Workbench

The React workbench is a visual editor for run configurations. It does not run
raster jobs. It reads source catalogs from the Python API, builds a YAML run
recipe, validates it, and leaves execution to the CLI or Slurm.

Start the config API:

```bash
pirineus-raster serve-config-api --host 127.0.0.1 --port 8765
```

Start the frontend in another terminal:

```bash
cd ui
npm install
npm run dev
```

The Vite dev server proxies `/api` requests to `http://127.0.0.1:8765`.

## Configuration Model

Run configs define the final dataset:

- `run.name`: dataset name.
- `run.project_config`: project YAML.
- `run.aoi_config`: final AOI/grid config.
- `run.clip_aoi_config`: optional larger clipping AOI.
- `run.resolution_m`: final output resolution.
- `sources`: source configs and stages to execute.
- `derived_features`: optional raster expressions from generated layers.
- `outputs.dataset_dir`: packaged dataset output directory.

Simplified run configs may also define source-level selections:

- `sources[].select.variables`: variables or indices to enable.
- `sources[].select.layers`: vector layers to enable.
- `sources[].select.dimensions`: selected GCMs, SSPs, periods or other dimensions.
- `sources[].select.aggregations.use`: named aggregation presets to enable.
- `sources[].select.aggregations.custom`: custom month/metric aggregations.
- `sources[].overrides.resampling`: explicit resampling overrides.
- `derived_feature_groups`: recipe-based derived features such as thermal range.

Source configs define provider-specific details:

- source identity, citation and product metadata.
- raw file structure and download mode.
- source CRS and native resolution.
- enabled variables or indices.
- resampling method per variable.
- temporal aggregations.
- output format and dtype.

Relative config paths are resolved robustly from the declaring config file, the
current working directory, and the repository root. This helps when jobs are
submitted from a different directory.

## Current Providers

- `worldclim`: climate normals, bioclimatic variables, elevation, CRU-TS
  historical monthly data and CMIP6 future projections.
- `copernicus`: CLMS static layers and temporal products downloaded through
  WEkEO HDA.
- `pdca`: Pyrenean Digital Climate Atlas topoclimate rasters.
- `igme_brgm`: transboundary Pyrenees geology vectors rasterized to the grid.

List available providers:

```bash
pirineus-raster list-sources
```

## Metadata and Validation

Final GeoTIFFs are written with embedded tags and sidecar JSON metadata. New
outputs include:

- provider, product, source ID and source config path.
- variable, units, valid range and scale factor.
- native resolution and resolution unit where known.
- output CRS, resolution, shape, transform, bounds, nodata and dtype.
- resampling method.
- grid path and AOI name.
- generation timestamp.

The dataset manifest in `data_processed/datasets/<run>/metadata/manifest.json`
indexes all generated rasters. `pirineus-raster validate-dataset` checks every
manifest raster against the declared reference grid and writes
`metadata/validation_report.json`.

## Resolution Notes

The project CRS is EPSG:3035 by default. Output resolution is configurable, but
it should not be interpreted as increased source precision.

For example, WorldClim `10m` means 10 arc-minutes, not 10 metres. The pipeline
can align such layers to a 100 m grid for modelling convenience, but metadata
must preserve the original source resolution semantics.

Continuous variables generally use `bilinear` or `average` resampling.
Categorical variables should use `nearest` or another category-preserving
method.

## HPC Usage

Reusable Slurm scripts live in `jobs/`.

Example:

```bash
sbatch jobs/run_raster_features.sh configs/runs/pallars_worldclim_full_100m.yaml
```

The job helpers switch to the repository directory, activate the conda
environment and print the Python/CLI context at the start of each job.

## Recommended Final Proof of Concept

For the thesis demonstration, a strong minimal proof of concept is:

1. Create the 100 m grid for `pyrenees_pdca_full`.
2. Run `configs/runs/pyrenees_pdca_topoclimate_100m.yaml`.
3. Validate the generated dataset.
4. Combine a smaller Pallars dataset with selected WorldClim and Copernicus
   layers for ecological modelling or habitat suitability experiments.

This shows the central goal of the project: heterogeneous sources converted
into coherent, traceable and pixel-aligned environmental raster databases.
