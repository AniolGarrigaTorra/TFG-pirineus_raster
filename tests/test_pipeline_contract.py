import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import rasterio
from rasterio.transform import from_origin

from src.io.config import load_yaml
from src.make_grid import create_grid
from src.pipeline.dataset import prune_manifest_to_final_features, write_json
from src.validation.validate_dataset import validate_dataset_dir


REPO_ROOT = Path(__file__).resolve().parents[1]


def _walk_dicts(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_dicts(child)


def _final_feature_names(feature):
    outputs = feature.get("outputs") or []
    if outputs:
        return [str(output["name"]) for output in outputs]
    return [str(feature["name"])]


class PipelineContractTests(unittest.TestCase):
    def test_repository_run_configs_reference_existing_sources_and_unique_outputs(self):
        run_config_paths = sorted((REPO_ROOT / "configs" / "runs").glob("*.yaml"))
        self.assertTrue(run_config_paths, "No run configs found under configs/runs.")

        for run_config_path in run_config_paths:
            with self.subTest(run_config=run_config_path.name):
                cfg = load_yaml(run_config_path)

                self.assertIn("features", cfg)
                self.assertNotIn(
                    "sources",
                    cfg,
                    "Researcher-facing run configs should use features, not legacy sources.",
                )
                self.assertNotIn(
                    "derived_features",
                    cfg,
                    "Researcher-facing run configs should not expose internal derived_features.",
                )

                final_names = [
                    name
                    for feature in cfg.get("features", [])
                    for name in _final_feature_names(feature)
                ]
                self.assertEqual(
                    len(final_names),
                    len(set(final_names)),
                    "Final feature output names must be unique.",
                )

                source_configs = sorted(
                    {
                        str(item["config"])
                        for item in _walk_dicts(cfg.get("features", []))
                        if item.get("kind") == "source" and item.get("config")
                    }
                )
                self.assertTrue(source_configs, "Run config does not reference any sources.")
                for source_config in source_configs:
                    self.assertTrue(
                        (REPO_ROOT / source_config).exists(),
                        f"Missing source config: {source_config}",
                    )

    def test_feature_oriented_manifest_pruning_keeps_only_final_derived_rasters(self):
        with TemporaryDirectory() as directory:
            dataset_dir = Path(directory) / "dataset"
            rasters_dir = dataset_dir / "rasters"
            metadata_dir = dataset_dir / "metadata"
            rasters_dir.mkdir(parents=True)
            metadata_dir.mkdir(parents=True)

            source_raster = rasters_dir / "internal_source.tif"
            source_json = rasters_dir / "internal_source.json"
            derived_raster = rasters_dir / "final_feature.tif"
            derived_json = rasters_dir / "final_feature.json"
            for path in [source_raster, source_json, derived_raster, derived_json]:
                path.write_text("placeholder", encoding="utf-8")

            manifest = {
                "dataset_name": "contract",
                "sources": [{"id": "worldclim"}],
                "n_sources": 1,
                "rasters": [
                    {
                        "name": "internal_source",
                        "source_id": "worldclim",
                        "dataset_path": "rasters/internal_source.tif",
                        "sidecar_json_dataset_path": "rasters/internal_source.json",
                    },
                    {
                        "name": "final_feature",
                        "source_id": "derived",
                        "dataset_path": "rasters/final_feature.tif",
                        "sidecar_json_dataset_path": "rasters/final_feature.json",
                    },
                ],
                "n_rasters": 2,
                "layer_catalog": [
                    {"name": "internal_source", "source_id": "worldclim"},
                    {"name": "final_feature", "source_id": "derived"},
                ],
            }
            write_json(metadata_dir / "manifest.json", manifest)

            pruned = prune_manifest_to_final_features(dataset_dir)

            self.assertEqual(pruned["n_rasters"], 1)
            self.assertEqual(pruned["rasters"][0]["name"], "final_feature")
            self.assertTrue(pruned["feature_oriented_outputs_only"])
            self.assertFalse(source_raster.exists())
            self.assertFalse(source_json.exists())
            self.assertTrue(derived_raster.exists())
            self.assertTrue(derived_json.exists())

    def test_dataset_validation_accepts_aligned_raster_and_rejects_grid_mismatch(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            configs_dir = root / "configs"
            dataset_dir = root / "dataset"
            rasters_dir = dataset_dir / "rasters"
            metadata_dir = dataset_dir / "metadata"
            configs_dir.mkdir()
            rasters_dir.mkdir(parents=True)
            metadata_dir.mkdir(parents=True)

            project_config = configs_dir / "project.yaml"
            aoi_config = configs_dir / "aoi.yaml"
            project_cfg = {
                "project_name": "contract",
                "crs": "EPSG:3035",
                "nodata": -9999.0,
                "paths": {
                    "raw_dir": str(root / "data_raw"),
                    "interim_dir": str(root / "data_interim"),
                    "processed_dir": str(root / "data_processed"),
                    "logs_dir": str(root / "logs"),
                },
                "grids": {
                    "subdir": "grids",
                    "available_resolutions_m": [100],
                    "default_resolution_m": 100,
                },
            }
            aoi_cfg = {
                "name": "tiny_aoi",
                "crs": "EPSG:3035",
                "bounds": {"xmin": 0, "ymin": 0, "xmax": 200, "ymax": 200},
            }
            project_config.write_text(
                "project_name: contract\n"
                "crs: EPSG:3035\n"
                "nodata: -9999.0\n"
                "paths:\n"
                f"  raw_dir: {root / 'data_raw'}\n"
                f"  interim_dir: {root / 'data_interim'}\n"
                f"  processed_dir: {root / 'data_processed'}\n"
                f"  logs_dir: {root / 'logs'}\n"
                "grids:\n"
                "  subdir: grids\n"
                "  available_resolutions_m: [100]\n"
                "  default_resolution_m: 100\n",
                encoding="utf-8",
            )
            aoi_config.write_text(
                "name: tiny_aoi\n"
                "crs: EPSG:3035\n"
                "bounds:\n"
                "  xmin: 0\n"
                "  ymin: 0\n"
                "  xmax: 200\n"
                "  ymax: 200\n",
                encoding="utf-8",
            )
            project_cfg["_config_path"] = str(project_config)
            create_grid(project_cfg, aoi_cfg, resolution=100, overwrite=True)

            aligned = rasters_dir / "aligned.tif"
            shifted = rasters_dir / "shifted.tif"
            profile = {
                "driver": "GTiff",
                "height": 2,
                "width": 2,
                "count": 1,
                "dtype": "float32",
                "crs": "EPSG:3035",
                "transform": from_origin(0, 200, 100, 100),
                "nodata": -9999.0,
            }
            with rasterio.open(aligned, "w", **profile) as dst:
                dst.write(np.ones((2, 2), dtype=np.float32), 1)
            with rasterio.open(
                shifted,
                "w",
                **{**profile, "transform": from_origin(100, 200, 100, 100)},
            ) as dst:
                dst.write(np.ones((2, 2), dtype=np.float32), 1)

            sidecar = {
                "metadata_schema_version": "0.2",
                "generated_at": "2026-01-01T00:00:00+00:00",
                "provider": "derived",
                "product": "features",
                "source_id": "derived",
                "variable": "aligned",
                "output_crs": "EPSG:3035",
                "output_resolution_m": 100,
                "nodata": -9999.0,
                "dtype": "float32",
                "grid_path": "unused",
                "layer_type": "derived",
                "operation": "expression",
                "inputs": {"x": "source"},
            }
            (rasters_dir / "aligned.json").write_text(
                json.dumps(sidecar),
                encoding="utf-8",
            )
            (rasters_dir / "shifted.json").write_text(
                json.dumps({**sidecar, "variable": "shifted"}),
                encoding="utf-8",
            )
            write_json(
                metadata_dir / "manifest.json",
                {
                    "dataset_name": "contract",
                    "project_config": str(project_config),
                    "run_aoi_config": str(aoi_config),
                    "run_resolution_m": 100,
                    "rasters": [
                        {
                            "name": "aligned",
                            "dataset_path": "rasters/aligned.tif",
                            "sidecar_json_dataset_path": "rasters/aligned.json",
                        },
                        {
                            "name": "shifted",
                            "dataset_path": "rasters/shifted.tif",
                            "sidecar_json_dataset_path": "rasters/shifted.json",
                        },
                    ],
                },
            )

            report = validate_dataset_dir(
                dataset_dir=dataset_dir,
                strict_metadata=True,
                write_report=False,
            )

            by_name = {item["name"]: item for item in report["rasters"]}
            self.assertTrue(by_name["aligned"]["ok"], by_name["aligned"]["errors"])
            self.assertFalse(by_name["shifted"]["ok"])
            self.assertIn("Failed check: transform_match", by_name["shifted"]["errors"])
            self.assertFalse(report["ok"])


if __name__ == "__main__":
    unittest.main()
