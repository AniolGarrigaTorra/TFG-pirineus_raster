# Pirineus Raster

Pirineus Raster is a configuration-driven Python pipeline for creating
homogeneous environmental raster datasets for the Pyrenees. It takes source
layers from different providers, formats, coordinate systems and resolutions,
then produces aligned GeoTIFF features with JSON metadata and a dataset
manifest.

The main goal is reproducibility: every final raster in a dataset should share
the same CRS, extent, resolution, affine transform and naming convention.

The full documentation of the project is available at the following link: https://github.com/AniolGarrigaTorra/pirineus-raster/blob/main/PirineusRaster-Memoria_catala.pdf

## What The Project Does

The pipeline has three source stages:

1. `download`: download or locate raw provider data.
2. `clip`: clip or prepare source data for the selected area of interest.
3. `build`: align requested outputs to the target grid and write metadata.

Dataset runs use researcher-facing YAML files in `configs/runs/`. These files
list the final features wanted by the user. The compiler expands those final
features into the internal source layers and derived operations needed to build
the dataset.

Final outputs are written under `data_processed/datasets/<run_name>/` and
normally include:

- `rasters/`: final GeoTIFF features and sidecar JSON metadata.
- `metadata/manifest.json`: dataset index.
- `metadata/run_summary.json`: execution summary.
- `metadata/validation_report.json`: written by dataset validation.
- `config/`: copies of the configs used for the run.

## Repository Layout

- `configs/project.yaml`: global paths, CRS, grid defaults and naming settings.
- `configs/aoi/`: area-of-interest definitions.
- `configs/sources/`: provider and product source configurations.
- `configs/runs/`: complete dataset recipes.
- `src/`: Python package, CLI, pipeline, source connectors and validation code.
- `tests/`: fast Python tests for compiler, pipeline contracts and validation.
- `ui/`: React workbench for building and validating run YAML files.
- `data_raw/`: downloaded or manually supplied raw source files.
- `data_interim/`: clipped, extracted or intermediate processing files.
- `data_processed/features/`: provider-level built feature rasters.
- `data_processed/datasets/`: packaged dataset outputs.

## Installation

The project expects Python 3.11 or newer. Conda is recommended because the
geospatial stack depends on compiled libraries such as GDAL, PROJ, rasterio,
geopandas and pyogrio.

Create the environment the first time:

```bash
conda env create -f environment.yml
```

Activate it:

```bash
conda activate pirineus-raster
```

Install the package in editable mode from the repository root:

```bash
pip install -e .
```

Check that the CLI is available:

```bash
pirineus-raster --help
```

For the UI, install Node dependencies once:

```bash
cd ui
npm install
```

Then return to the repository root before running Python commands:

```bash
cd ..
```

## Credentials

Some Copernicus products use the WEkEO HDA service. Those downloads require
credentials through `~/.hdarc` or the `HDA_USER` and `HDA_PASSWORD`
environment variables.

Check the local setup:

```bash
pirineus-raster check-credentials
```

Create or refresh `~/.hdarc` interactively:

```bash
pirineus-raster check-credentials --setup
```

Runs that do not use authenticated sources do not need these credentials.

## Basic Workflow

Create the reference grid for an AOI and resolution:

```bash
pirineus-raster make-grid \
  --project-config configs/project.yaml \
  --aoi-config configs/aoi/ursus_arctos_pyrenees.yaml \
  --resolution 100
```

Validate a run config before processing data:

```bash
pirineus-raster validate-config configs/runs/ursus_arctos_pyrenees_100m.yaml
```

Run the complete dataset recipe:

```bash
pirineus-raster run configs/runs/ursus_arctos_pyrenees_100m.yaml
```

Inspect the generated dataset:

```bash
pirineus-raster inspect data_processed/datasets/ursus_arctos_pyrenees_100m
```

Validate all raster outputs against the dataset grid:

```bash
pirineus-raster validate-dataset data_processed/datasets/ursus_arctos_pyrenees_100m
```

Use strict metadata checks for regenerated datasets:

```bash
pirineus-raster validate-dataset \
  data_processed/datasets/ursus_arctos_pyrenees_100m \
  --strict-metadata
```

## CLI Commands

List available commands:

```bash
pirineus-raster --help
```

Useful commands:

- `make-grid`: create a reference raster grid for an AOI and resolution.
- `validate-config`: validate a run YAML and estimate source/derived layers.
- `render-run`: print the compiled run YAML after feature expansion.
- `run`: execute a full dataset recipe.
- `inspect`: summarize a generated dataset manifest.
- `validate-dataset`: check generated rasters against their reference grid.
- `catalog`: print source or workbench catalog information as JSON.
- `list-sources`: list registered source providers.
- `run-source`: developer command for running one source stage directly.
- `serve-config-api`: start the local API used by the React workbench.
- `serve-ui`: start the API and UI together.
- `check-credentials`: check or configure WEkEO/HDA credentials.

Examples:

```bash
pirineus-raster catalog
pirineus-raster catalog --source-config configs/sources/worldclim/worldclim_v2_1_bioclim.yaml
pirineus-raster render-run configs/runs/ursus_arctos_pyrenees_100m.yaml
pirineus-raster list-sources
```

For low-memory machines, reduce dataset-run memory pressure:

```bash
pirineus-raster run configs/runs/ursus_arctos_pyrenees_100m.yaml \
  --num-workers 1 \
  --max-rasters-in-memory 2
```

## UI Workbench

The UI is a local configuration workbench. Its purpose is to help users build,
inspect and validate run YAMLs without manually editing every source and
feature block.

The UI can:

- browse the source catalog exposed by the Python API.
- select AOIs, CRS and resolution.
- select source layers, temporal options, dimensions and resampling choices.
- define derived features and advanced expressions.
- create or validate run YAML content.
- request grid creation through the local API.

The UI does not replace the pipeline runner. Full dataset execution is still
done with the CLI command `pirineus-raster run <run_config>`.

Start the Python config API:

```bash
pirineus-raster serve-config-api --host 127.0.0.1 --port 8765
```

Start the frontend in another terminal:

```bash
cd ui
npm run dev
```

Open the Vite URL printed by the frontend, normally:

```text
http://127.0.0.1:5173
```

The Vite server proxies `/api` requests to `http://127.0.0.1:8765`.

Alternatively, start both API and UI with:

```bash
pirineus-raster serve-ui
```

Build or test the UI:

```bash
cd ui
npm run build
npm test
```

## Configuration Model

Run configs define the final dataset. The main keys are:

- `run.name`: dataset name.
- `run.project_config`: project YAML, usually `configs/project.yaml`.
- `run.aoi_config`: AOI used by the final output grid.
- `run.clip_aoi_config`: optional larger AOI used for clipping.
- `run.resolution_m`: target output resolution in metres.
- `run.crs`: optional CRS override, such as `EPSG:3035`.
- `run.stages`: stages to run, usually `all` or `build`.
- `features`: final user-facing features.
- `outputs.dataset_dir`: output dataset directory.

The `features` list is the source of truth for dataset contents. Legacy
top-level `sources` or `derived_features` blocks are internal concepts and
should not be used in researcher-facing run configs.

Each final feature usually declares:

- `name`: stable output name.
- `title` and `description`: human-readable metadata.
- `unit`: measurement unit when known, otherwise blank.
- `value_semantics`: how values should be interpreted.
- `output_dtype`: expected raster dtype.
- `build_type`: `source_layer`, `recipe`, `masking`, `spatial` or `expression`.
- `inputs`: source or feature inputs needed by the operation.
- `parameters`: operation-specific options.

Source inputs declare:

- `source_id`: stable source identifier.
- `config`: source config YAML.
- `variable`, `layer` or `category_fraction`: requested source output.
- `dimensions`: non-temporal selections such as model, scenario, period or year.
- `temporal`: temporal output mode and selected dates or aggregations.
- `source_resolution`: native/provider resolution choice where applicable.
- `resampling`: method used to align the source to the target grid.

## Derived Features

Derived features are built after source layers are prepared. Supported derived
families include expressions, terrain operations, focal/spatial operations,
masks, distances and recipes.

Expression features use a restricted numeric expression engine. Common
functions include `where`, `log`, `log1p`, `sqrt`, `minimum`, `maximum`,
`clip`, comparisons and arithmetic. Unsafe Python calls are rejected.

Spatial and terrain features can use:

- `target_grid`: inputs are aligned first, then the operation is evaluated.
- `native_then_resample`: the operation is evaluated near native resolution,
  then resampled to the final grid. This is often better for slope,
  ruggedness, focal windows and distance surfaces, but it can be slower and
  requires source metadata that points to clipped native inputs.

## Value Semantics

`value_semantics` is metadata used by the UI, validation and resampling logic.
It is not a GeoTIFF standard field, but it gives the pipeline a consistent way
to understand raster values.

Common values:

- `categorical`: nominal class codes such as land cover.
- `ordinal`: ordered class codes.
- `binary`: 0/1 presence or mask values.
- `intensive`: local continuous values such as elevation or temperature.
- `intensive_depth`: depth-like accumulated fields such as precipitation in mm.
- `percentage`: 0-100 values.
- `fraction`: 0-1 proportions.
- `ratio`: unitless ratios.
- `extensive`: cell totals such as built-up square metres per cell.
- `count`: discrete counts.
- `circular`: angles such as aspect.

## Current Providers

- `worldclim`: climate normals, bioclimatic variables, elevation, CRU-TS
  monthly data and CMIP6 future projections.
- `copernicus`: CLMS and related products, including DEM, land cover, forest,
  grasslands, imperviousness, water/wetness, HR-VPP and HRSI snow.
- `pdca`: Pyrenean Digital Climate Atlas topoclimate rasters.
- `igme_brgm`: transboundary Pyrenees geology vectors rasterized to the grid.
- `openstreetmap`: Geofabrik extracts filtered into anthropic vector layers.
- `ghsl`: population, built-up surface and settlement model grids.
- `esa_cci`: above-ground biomass and uncertainty rasters.
- `esa_worldcover`: ESA WorldCover land-cover rasters.

## Testing

Run the Python tests from the repository root:

```bash
python -m unittest discover tests
```

Run one test module:

```bash
python -m unittest tests.test_pipeline_contract
```

Run UI tests:

```bash
cd ui
npm test
```

The Python tests are designed to be fast and mostly synthetic. They validate
configuration compilation, derived expressions, source selection logic,
metadata contracts and tiny raster validation cases. They should not download
remote data.

## Validation And Metadata

Every final raster should have:

- a GeoTIFF file.
- a sidecar JSON file.
- manifest entries under `metadata/manifest.json`.
- CRS, transform, shape and resolution matching the declared grid.
- nodata, dtype, provider, product, source ID and variable metadata.
- source config and resampling metadata where applicable.
- operation and input metadata for derived layers.

`validate-dataset` checks the manifest rasters against the reference grid and
writes a validation report:

```bash
pirineus-raster validate-dataset data_processed/datasets/<dataset_name>
```

Strict mode also fails missing standardized metadata keys:

```bash
pirineus-raster validate-dataset data_processed/datasets/<dataset_name> --strict-metadata
```

## Important Warnings

- Output resolution does not increase source precision. A 100 m output built
  from a coarse source is still limited by the original data.
- Create the target grid before running build stages. Missing grids are a
  preflight warning and will become a build-time problem.
- Use category fractions for land-cover proportions. A categorical mask after
  coarse resampling cannot recover sub-cell composition.
- Use category-preserving resampling such as `nearest` or `mode` for class
  codes. Use `average`, `bilinear` or similar methods for continuous values.
- `conservative_sum` is only appropriate for true extensive cell totals.
- Authenticated Copernicus downloads need valid HDA credentials.
- Large source downloads and high-resolution native operations can require a
  lot of disk space and memory.
- Existing outputs may be skipped or overwritten depending on source builder
  logic and run settings. Validate manifests and sidecar metadata after
  important reruns.
- Relative config paths are resolved from the declaring config file, the
  current working directory and the repository root. Prefer repository-relative
  paths in committed configs.

## Recommended Development Loop

For config or pipeline changes:

```bash
python -m unittest discover tests
pirineus-raster validate-config configs/runs/ursus_arctos_pyrenees_100m.yaml
pirineus-raster render-run configs/runs/ursus_arctos_pyrenees_100m.yaml
```

For UI changes:

```bash
cd ui
npm test
npm run build
```

For regenerated datasets:

```bash
pirineus-raster inspect data_processed/datasets/<dataset_name>
pirineus-raster validate-dataset data_processed/datasets/<dataset_name> --strict-metadata
```
