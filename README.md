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

The tracked `environment.yml` is intended to be portable. If a fully locked
machine-specific environment is needed, keep it as a separate lock/export file
rather than committing a local `prefix`.

Copernicus WEkEO downloads require HDA credentials, normally through `~/.hdarc`
or `HDA_USER` and `HDA_PASSWORD`.

## Basic Workflow

Create the reference grid for an AOI and resolution:

```bash
pirineus-raster make-grid \
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
pirineus-raster run configs/runs/ursus_arctos_pyrenees_100m.yaml
```

Inspect the generated manifest:

```bash
pirineus-raster inspect data_processed/datasets/ursus_arctos_pyrenees_100m
```

Validate all rasters in a generated dataset:

```bash
pirineus-raster validate-dataset data_processed/datasets/ursus_arctos_pyrenees_100m
```

Use strict metadata validation for newly regenerated datasets:

```bash
pirineus-raster validate-dataset \
  data_processed/datasets/ursus_arctos_pyrenees_100m \
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
  configs/runs/ursus_arctos_pyrenees_100m.yaml

pirineus-raster render-run \
  configs/runs/ursus_arctos_pyrenees_100m.yaml
```

`validate-config` checks the effective source selections used by the runner. It
also reports pre-run warnings such as missing target grids, so a recipe can be
structurally valid while still needing `make-grid` before `build`.

Check local WEkEO/HDA credentials before launching Copernicus downloads:

```bash
pirineus-raster check-credentials
pirineus-raster check-credentials --setup
```

## React Workbench

The React workbench is a visual editor for run configurations. It reads source
catalogs from the Python API, builds and validates YAML run recipes, can create
new AOI config files, and can create the target grid for the selected AOI,
CRS and resolution. Full raster dataset execution is still done through the
CLI, the detached local helper, or Slurm.

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

Alternatively, start both processes with one command:

```bash
pirineus-raster serve-ui
```

## Configuration Model

Run configs define the final dataset:

- `run.name`: dataset name.
- `run.project_config`: project YAML.
- `run.aoi_config`: final AOI/grid config.
- `run.clip_aoi_config`: optional larger clipping AOI.
- `run.resolution_m`: final output resolution.
- `run.crs`: optional output CRS override such as `EPSG:3035` or `EPSG:25831`.
- `features`: final dataset features requested by the user.
- `outputs.dataset_dir`: packaged dataset output directory.

The `features` list is the researcher-facing source of truth. Legacy run
configs with top-level `sources` or `derived_features` are intentionally
rejected by validation. The compiler expands final features into the internal
source requirements and derived outputs needed by the runner, then the dataset
manifest is pruned so only the final feature rasters remain visible.

Each final feature has:

- `name`: stable output name used for the GeoTIFF filename.
- `build_type`: one of `source_layer`, `recipe`, `masking`, `spatial` or
  `expression`.
- optional metadata such as `title`, `description` and `unit`. The compiler
  infers `value_semantics` and `output_dtype` from the source metadata and the
  selected operation unless an advanced override is provided.
- one or more inputs. Inputs can point to official source layers or to earlier
  final features created in the same run.

Official source inputs can define:

- `source_id` and `config`: the provider/product source config.
- `variable`, `layer` or `category_fraction`: the requested source output.
- `dimensions`: selected non-temporal dimensions such as GCM, SSP, period,
  season or product year, depending on the source.
- `temporal`: source-aware temporal choices.
- `source_resolution` and `resampling`: how the source should become the
  project grid.

Category fractions are the preferred way to turn categorical land-cover classes
into target-cell proportions. They are computed before target-grid resampling,
so `average` resampling gives a 0-1 coverage fraction at 100 m or any other
target resolution. A class mask, by contrast, tests an already aligned raster
and cannot recover sub-cell composition lost during categorical resampling.

Derived/processed features can also declare `evaluation_stage`:

- `target_grid`: the default cheap path. Source inputs are first aligned to the
  project grid, then the expression/recipe/spatial operation is evaluated.
- `native_then_resample`: inputs are reprojected to a metric intermediate grid
  using the best native resolution recorded in metadata, the operation is
  evaluated there, and the result is then aggregated/resampled to the final
  project grid. This is recommended for DEM terrain derivatives, focal windows
  and distance surfaces when the target resolution is coarser than the source.

For example, `target_grid` slope at 100 m means "slope of the DEM already
smoothed to 100 m"; `native_then_resample` slope means "native-scale slope
aggregated to the 100 m cells". The latter is usually more informative but can
be slower and depends on source builders exposing `source_clipped_path`
metadata.

`value_semantics` is Pirineus Raster metadata that describes how raster values
should be interpreted. It is not a GeoTIFF standard field, but it follows common
GIS/statistical concepts and is used for UI filtering, validation and resampling
guidance. In normal workbench use it is inferred automatically; set it manually
only when an advanced expression creates a genuinely ambiguous output:

- `categorical`: nominal class codes such as land cover or geology.
- `ordinal`: ordered class codes where rank matters but numeric spacing may not.
- `binary`: 0/1 masks for presence/absence.
- `intensive`: continuous local values such as elevation, temperature, distance
  or biomass per hectare.
- `intensive_depth`: depth-like accumulated fields such as precipitation in mm.
- `percentage`: 0-100 values such as tree cover density.
- `fraction`: 0-1 proportions such as category coverage fractions.
- `ratio`: unitless ratios that are not necessarily limited to 0-1.
- `extensive`: cell totals such as built-up square metres per cell.
- `count`: discrete counts such as population or snow days.
- `circular`: angles such as aspect where 0 and 360 degrees are neighbours.

Temporal selections are explicit because sources do not all behave the same:

- static and vector sources use `output_mode: static`.
- WorldClim monthly climatologies and CMIP6 can use `output_mode: aggregate`
  with named/custom aggregations, or `output_mode: raw_slices` to write one
  output per selected month.
- CRU-TS year-month series can aggregate with year/month ranges, two-step
  yearly summaries, or `raw_slices` for one output per selected year-month.
- PDCA uses `output_mode: supplied_layers` because annual, monthly and
  seasonal layers are supplied by the source rather than computed here.
- yearly static collections such as GHSL GHS-POP, GHS-BUILT-S, GHS-SMOD,
  ESA CCI Biomass and Copernicus HR-VPP expose base variables in the variable
  picker and let the Temporal tab select years or define year-range
  aggregations.
- HRSI snow uses `output_mode: postprocess_aggregate` because temporal outputs
  are generated during the Copernicus download/postprocess stage.

Feature-oriented configs may request multiple temporal aggregations for the
same final feature family. Multi-input derived features only combine temporal
outputs that share the same temporal label; non-temporal dimensions expand by
cartesian product.

Resampling is variable-aware. Source configs expose defaults per variable and
the UI can override them for a run. Standard raster reprojection methods include
`nearest`, `bilinear`, `cubic`, `average`, `mode`, `sum` and related GDAL/rasterio
methods. `conservative_sum` is reserved for truly extensive variables whose cell
values are totals/counts and should be redistributed by target/source pixel area.
Precipitation in `mm` is treated as `intensive_depth`, not as an extensive cell
total. Kriging-style and point-interpolation methods are not exposed as runnable
options until a geostatistical backend exists.

Source configs define provider-specific details:

- source identity, citation and product metadata.
- source domains under `domains.clip_aoi_config` and
  `domains.output_aoi_config`.
- raw file structure and download mode.
- source CRS and native resolution.
- enabled variables or indices.
- resampling method per variable.
- temporal capability and aggregation presets where the source supports them.
- output format and dtype.

Relative config paths are resolved robustly from the declaring config file, the
current working directory, and the repository root. This helps when jobs are
submitted from a different directory.

## Current Providers

- `worldclim`: climate normals, bioclimatic variables, elevation, CRU-TS
  historical monthly data and CMIP6 future projections.
- `copernicus`: CLMS static layers and temporal products downloaded through
  WEkEO HDA, including DEM, HRLs, CLC, CLC+ Backbone, HR-VPP and HRSI snow.
- `pdca`: Pyrenean Digital Climate Atlas topoclimate rasters.
- `igme_brgm`: transboundary Pyrenees geology vectors rasterized to the grid.
- `openstreetmap`: Geofabrik PBF extracts filtered into anthropic vector
  layers and rasterized as presence or distance features.
- `ghsl`: GHSL GHS-POP population, GHS-BUILT-S built-up surface and GHS-SMOD
  settlement model grids with selectable native resolution and temporal years.
- `esa_cci`: ESA CCI Biomass above-ground biomass and uncertainty rasters.
- `esa_worldcover`: ESA WorldCover 10 m global land-cover tiles mosaicked for
  the Pyrenees.

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

WorldClim source resolutions are written as `30arcs`, `2.5arcmin`, `5arcmin`
and `10arcmin` in user-facing configs and metadata. Provider download URLs
still use the original WorldClim tokens such as `30s`, `2.5m`, `5m` and `10m`.
The pipeline can align such layers to a 100 m grid for modelling convenience,
but metadata must preserve the original source resolution semantics.

Continuous variables generally use `bilinear` or `average` resampling.
Categorical variables should use `nearest` or another category-preserving
method.

## HPC Usage

Reusable Slurm scripts live in `jobs/`.

Example:

```bash
sbatch jobs/run_raster_features.sh configs/runs/ursus_arctos_pyrenees_100m.yaml
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
