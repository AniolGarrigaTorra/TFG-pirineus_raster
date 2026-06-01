from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from src.io.config import load_yaml
from src.make_grid import create_grid
from src.pipeline.raster_ops import build_static_feature_metadata
from src.pipeline.runner import _load_domain_configs
from src.pipeline.source_overrides import normalize_source_domains
from src.pipeline.variable_expansion import expand_source_config
from src.sources.generic_raster.naming import get_download_file_specs
from src.sources.worldclim.naming import (
    build_worldclim_download_url,
    build_worldclim_zip_name,
    get_zip_specs,
)
from src.sources.registry import list_source_connectors
from src.workbench.catalog import list_source_catalogs
from src.workbench.compiler import (
    compile_source_config_for_run,
    validate_researcher_run_config,
)


class NewSourceIntegrationTests(unittest.TestCase):
    def test_new_connectors_are_registered(self):
        connectors = set(list_source_connectors())
        self.assertIn("openstreetmap", connectors)
        self.assertIn("ghsl", connectors)
        self.assertIn("esa_cci", connectors)

    def test_new_sources_are_visible_in_catalog(self):
        catalogs = {item["id"]: item for item in list_source_catalogs()}

        self.assertEqual(
            len(catalogs["openstreetmap_geofabrik_pyrenees"].get("layers", [])),
            5,
        )
        self.assertEqual(
            len(catalogs["ghsl_ghs_pop_r2023a"].get("variables", [])),
            1,
        )
        self.assertEqual(
            [item["name"] for item in catalogs["ghsl_ghs_pop_r2023a"]["variables"]],
            ["population_count"],
        )
        self.assertIn(
            "aggregate",
            catalogs["ghsl_ghs_pop_r2023a"]["temporal"]["output_modes"],
        )
        self.assertEqual(
            catalogs["ghsl_ghs_pop_r2023a"]["temporal"]["temporal_layers"]["years"],
            [1975, 1980, 1985, 1990, 1995, 2000, 2005, 2010, 2015, 2020, 2025, 2030],
        )
        self.assertEqual(
            catalogs["ghsl_ghs_pop_r2023a"]["source_resolution_options"],
            ["100m", "1000m", "3arcs", "30arcs"],
        )
        self.assertEqual(
            len(catalogs["esa_cci_biomass_agb_100m"].get("variables", [])),
            2,
        )
        self.assertEqual(
            [item["name"] for item in catalogs["esa_cci_biomass_agb_100m"]["variables"]],
            ["agb", "agb_sd"],
        )

    def test_workbench_compiler_accepts_new_source_selections(self):
        cfg = {
            "run": {
                "name": "new_sources_smoke",
                "project_config": "configs/project.yaml",
                "crs": "EPSG:3035",
                "aoi_config": "configs/aoi/experimental_pallars_sobira.yaml",
                "resolution_m": 100,
                "stages": ["build"],
            },
            "sources": [
                {
                    "id": "osm",
                    "config": "configs/sources/openstreetmap/openstreetmap_geofabrik_pyrenees.yaml",
                    "stages": ["build"],
                    "select": {
                        "layers": [
                            "transport.secondary_roads_distance",
                            "settlements.settlements_distance",
                        ],
                    },
                },
                {
                    "id": "pop",
                    "config": "configs/sources/ghsl/ghsl_ghs_pop_r2023a.yaml",
                    "stages": ["build"],
                    "select": {
                        "variables": ["population_count"],
                        "temporal": {
                            "output_mode": "supplied_layers",
                            "layers": {"years": [2020]},
                        },
                    },
                },
                {
                    "id": "biomass",
                    "config": "configs/sources/esa_cci/esa_cci_biomass_agb_100m.yaml",
                    "stages": ["build"],
                    "select": {
                        "variables": ["agb"],
                        "temporal": {
                            "output_mode": "supplied_layers",
                            "layers": {"years": [2020]},
                        },
                    },
                },
            ],
        }

        report = validate_researcher_run_config(cfg)
        self.assertTrue(report["ok"])
        self.assertEqual(report["estimated_layers"], 4)

    def test_yearly_static_collections_support_base_variable_aggregations(self):
        cfg = {
            "run": {
                "name": "annual_aggregation_smoke",
                "project_config": "configs/project.yaml",
                "crs": "EPSG:3035",
                "aoi_config": "configs/aoi/experimental_pallars_sobira.yaml",
                "resolution_m": 100,
                "stages": ["build"],
            },
            "sources": [
                {
                    "id": "biomass",
                    "config": "configs/sources/esa_cci/esa_cci_biomass_agb_100m.yaml",
                    "stages": ["build"],
                    "select": {
                        "variables": ["agb"],
                        "temporal": {
                            "output_mode": "aggregate",
                            "aggregations": {
                                "custom": [
                                    {
                                        "name": "mean_2015_2024",
                                        "form": "year_range_metric",
                                        "years": [2015, 2024],
                                        "metric": "mean",
                                        "variables": ["agb"],
                                    }
                                ]
                            },
                        },
                    },
                },
            ],
        }

        report = validate_researcher_run_config(cfg)
        self.assertTrue(report["ok"], report["errors"])
        self.assertEqual(report["estimated_layers"], 1)

        source_cfg = expand_source_config(
            load_yaml("configs/sources/esa_cci/esa_cci_biomass_agb_100m.yaml")
        )
        compiled = compile_source_config_for_run(source_cfg, cfg["sources"][0])
        enabled = [
            name
            for name, item in compiled["variables"].items()
            if item.get("enabled")
        ]
        self.assertEqual(enabled, [f"agb_{year}" for year in range(2015, 2025)])

    def test_ghsl_resolution_labels_are_mapped_to_provider_tokens(self):
        cfg = expand_source_config(
            load_yaml("configs/sources/ghsl/ghsl_ghs_pop_r2023a.yaml")
        )
        compiled = compile_source_config_for_run(
            cfg,
            {
                "overrides": {"processing": {"source_resolution": "3arcs"}},
                "select": {
                    "variables": ["population_count"],
                    "temporal": {
                        "output_mode": "supplied_layers",
                        "layers": {"years": [2020]},
                    },
                },
            },
        )

        spec = get_download_file_specs(compiled)[0]
        metadata = build_static_feature_metadata(
            compiled,
            "population_count_2020",
            compiled["variables"]["population_count_2020"],
            "pyrenees_full",
            "experimental_pallars_sobira",
            100,
            "conservative_sum",
        )

        self.assertIn("_3arcs_", spec["filename"])
        self.assertIn("4326_3ss", spec["urls"][0])
        self.assertIn("4326_3ss", spec["zip_member_pattern"])
        self.assertEqual(metadata["source_resolution"], "3arcs")
        self.assertEqual(metadata["source_resolution_unit"], "arcs")
        self.assertEqual(metadata["source_crs"], "EPSG:4326")

    def test_worldclim_resolution_labels_use_provider_tokens_for_downloads(self):
        cfg = load_yaml("configs/sources/worldclim/worldclim_v2_1_climate_normals.yaml")
        compiled = compile_source_config_for_run(
            cfg,
            {"overrides": {"processing": {"source_resolution": "10arcmin"}}},
        )
        zip_spec = get_zip_specs(compiled)[0]
        metadata = build_static_feature_metadata(
            compiled,
            "tmin",
            compiled["variables"]["tmin"],
            "pyrenees_full",
            "experimental_pallars_sobira",
            100,
            "bilinear",
        )

        self.assertEqual(
            build_worldclim_zip_name(compiled, zip_spec),
            "wc2.1_10m_tmin.zip",
        )
        self.assertIn(
            "/wc2.1_10m_tmin.zip",
            build_worldclim_download_url(compiled, zip_spec),
        )
        self.assertEqual(metadata["source_resolution"], "10arcmin")
        self.assertEqual(metadata["source_resolution_unit"], "arcmin")

    def test_osm_domains_follow_runner_contract(self):
        cfg = load_yaml("configs/sources/openstreetmap/openstreetmap_geofabrik_pyrenees.yaml")
        cfg["_config_path"] = "configs/sources/openstreetmap/openstreetmap_geofabrik_pyrenees.yaml"
        self.assertIn("domains", cfg)
        clip_aoi, output_aoi = _load_domain_configs(cfg)
        self.assertEqual(clip_aoi["name"], "pyrenees_full")
        self.assertEqual(output_aoi["name"], "experimental_pallars_sobira")

    def test_legacy_dataset_domains_are_normalized(self):
        cfg = {
            "dataset": {
                "clip_aoi_config": "configs/aoi/pyrenees_full.yaml",
                "output_aoi_config": "configs/aoi/experimental_pallars_sobira.yaml",
            }
        }
        normalized = normalize_source_domains(cfg)
        self.assertEqual(
            normalized["domains"]["clip_aoi_config"],
            "configs/aoi/pyrenees_full.yaml",
        )
        self.assertNotIn("clip_aoi_config", normalized["dataset"])

    def test_make_grid_accepts_geographic_aoi_bounds(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            project_cfg = {
                "_config_path": str(tmp_path / "configs" / "project.yaml"),
                "crs": "EPSG:3035",
                "nodata": -9999.0,
                "paths": {"interim_dir": str(tmp_path / "interim")},
                "grids": {
                    "subdir": "grids",
                    "available_resolutions_m": [100],
                    "default_resolution_m": 100,
                },
            }
            aoi_cfg = {
                "name": "wgs84_test_aoi",
                "crs": "EPSG:4326",
                "bounds": {
                    "xmin": 0.6,
                    "xmax": 0.7,
                    "ymin": 42.4,
                    "ymax": 42.5,
                },
            }
            with contextlib.redirect_stdout(io.StringIO()):
                path = create_grid(project_cfg, aoi_cfg, resolution=100, overwrite=True)
            self.assertTrue(path.exists())
            self.assertIn("wgs84_test_aoi", path.name)

    def test_derived_features_require_packaged_manifest_outputs(self):
        cfg = {
            "run": {
                "name": "bad_derived_contract",
                "project_config": "configs/project.yaml",
                "aoi_config": "configs/aoi/experimental_pallars_sobira.yaml",
                "resolution_m": 100,
                "stages": ["build"],
            },
            "sources": [
                {
                    "id": "worldclim",
                    "config": "configs/sources/worldclim/worldclim_v2_1_elevation.yaml",
                    "select": {"variables": ["elev"]},
                }
            ],
            "derived_features": [
                {
                    "name": "slope",
                    "operation": "terrain",
                    "method": "slope",
                    "inputs": {
                        "dem": {"source_id": "worldclim", "variable": "elev"}
                    },
                }
            ],
            "outputs": {
                "copy_rasters": False,
                "write_manifest": True,
            },
        }
        report = validate_researcher_run_config(cfg)
        self.assertFalse(report["ok"])
        self.assertIn(
            "derived_features require outputs.copy_rasters=true.",
            report["errors"],
        )


if __name__ == "__main__":
    unittest.main()
