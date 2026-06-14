from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import geopandas as gpd

from src.io.config import load_yaml
from src.make_grid import create_grid
from src.pipeline.raster_ops import build_static_feature_metadata
from src.pipeline.resampling import VALUE_SEMANTICS
from src.pipeline.runner import _load_domain_configs
from src.pipeline.source_overrides import normalize_source_domains
from src.pipeline.variable_expansion import expand_source_config
from src.sources.generic_raster.naming import get_download_file_specs
from src.sources.copernicus.naming import (
    get_download_file_specs as get_copernicus_download_file_specs,
)
from src.sources.worldclim.naming import (
    build_worldclim_cmip6_download_url,
    build_worldclim_download_url,
    build_worldclim_zip_name,
    get_file_specs,
    get_zip_specs,
)
from src.sources.registry import list_source_connectors
from src.workbench.catalog import list_source_catalogs
from src.workbench.compiler import (
    compile_run_config,
    compile_source_config_for_run,
    render_run_config_yaml,
    validate_researcher_run_config,
)
from src.pipeline.config import validate_run_config
from src.sources.openstreetmap.build import _build_layer_array


class _DummyGrid:
    shape = (4, 4)
    transform = None
    resolution_m = 100


class NewSourceIntegrationTests(unittest.TestCase):
    def test_feature_oriented_source_layer_compiles_to_identity_output(self):
        cfg = {
            "run": {
                "name": "feature_contract_smoke",
                "project_config": "configs/project.yaml",
                "crs": "EPSG:3035",
                "aoi_config": "configs/aoi/experimental_pallars_sobira.yaml",
                "resolution_m": 100,
                "stages": ["build"],
            },
            "features": [
                {
                    "name": "bio1",
                    "title": "Annual mean temperature",
                    "build_type": "source_layer",
                    "source": {
                        "kind": "source",
                        "source_id": "worldclim",
                        "config": "configs/sources/worldclim/worldclim_v2_1_bioclim.yaml",
                        "variable": "bio1",
                    },
                },
                {
                    "name": "bio2",
                    "title": "Mean diurnal range",
                    "build_type": "source_layer",
                    "source": {
                        "kind": "source",
                        "source_id": "worldclim",
                        "config": "configs/sources/worldclim/worldclim_v2_1_bioclim.yaml",
                        "variable": "bio2",
                    },
                },
            ],
            "outputs": {"dataset_dir": "data/processed/feature_contract_smoke"},
        }

        report = validate_researcher_run_config(cfg)
        self.assertTrue(report["ok"], report["errors"])
        compiled = compile_run_config(cfg)
        self.assertTrue(compiled["_compiled_from_features"])
        self.assertEqual(len(compiled["sources"]), 1)
        self.assertEqual(
            sorted(compiled["sources"][0]["select"]["variables"]),
            ["bio1", "bio2"],
        )
        self.assertEqual(len(compiled["derived_features"]), 2)
        self.assertEqual(compiled["derived_features"][0]["operation"], "expression")
        self.assertEqual(compiled["derived_features"][0]["expression"], "x")

    def test_render_run_config_yaml_does_not_emit_yaml_aliases(self):
        shared_temporal = {
            "output_mode": "supplied_layers",
            "layers": {"years": [2018]},
        }
        cfg = {
            "run": {
                "name": "no_yaml_aliases",
                "project_config": "configs/project.yaml",
                "aoi_config": "configs/aoi/experimental_pallars_sobira.yaml",
                "resolution_m": 100,
            },
            "features": [
                {
                    "name": "two_outputs",
                    "build_type": "source_layer",
                    "outputs": [
                        {
                            "name": "a",
                            "source": {
                                "kind": "source",
                                "source_id": "copernicus_clms_clcplus_backbone",
                                "config": "configs/sources/copernicus/copernicus_clms_clcplus_backbone.yaml",
                                "variable": "clcplus_backbone",
                                "temporal": shared_temporal,
                            },
                        },
                        {
                            "name": "b",
                            "source": {
                                "kind": "source",
                                "source_id": "copernicus_clms_clcplus_backbone",
                                "config": "configs/sources/copernicus/copernicus_clms_clcplus_backbone.yaml",
                                "variable": "clcplus_backbone",
                                "temporal": shared_temporal,
                            },
                        },
                    ],
                }
            ],
            "outputs": {"dataset_dir": "data/processed/no_yaml_aliases"},
        }

        rendered = render_run_config_yaml(cfg)
        self.assertNotIn("&id", rendered)
        self.assertNotIn("*id", rendered)
        self.assertEqual(rendered.count("output_mode: supplied_layers"), 2)

    def test_feature_oriented_yearly_dimension_source_layer_keeps_temporal_query(self):
        cfg = {
            "run": {
                "name": "hrvpp_feature_source_layer",
                "project_config": "configs/project.yaml",
                "crs": "EPSG:3035",
                "aoi_config": "configs/aoi/experimental_pallars_sobira.yaml",
                "resolution_m": 100,
                "stages": ["build"],
            },
            "features": [
                {
                    "name": "amplitude_s1_2020",
                    "title": "Seasonal amplitude",
                    "build_type": "source_layer",
                    "source": {
                        "kind": "source",
                        "source_id": "copernicus_clms_hrvpp_vpp_laea",
                        "config": "configs/sources/copernicus/copernicus_clms_hrvpp_vpp_laea.yaml",
                        "variable": "amplitude",
                        "dimensions": {"growth_season": ["s1"]},
                        "temporal": {
                            "output_mode": "supplied_layers",
                            "layers": {"years": [2020]},
                        },
                        "query": {
                            "source_id": "copernicus_clms_hrvpp_vpp_laea",
                            "variable": "amplitude_s1_2020",
                            "season": "s1",
                        },
                    },
                },
            ],
            "outputs": {"dataset_dir": "data/processed/hrvpp_feature_source_layer"},
        }

        report = validate_researcher_run_config(cfg)
        self.assertTrue(report["ok"], report["errors"])
        compiled = compile_run_config(cfg)
        self.assertEqual(
            compiled["sources"][0]["select"]["temporal"]["output_mode"],
            "supplied_layers",
        )
        self.assertEqual(
            compiled["sources"][0]["select"]["dimensions"],
            {"growth_season": ["s1"]},
        )
        self.assertEqual(
            compiled["derived_features"][0]["inputs"]["x"]["variable"],
            "amplitude_s1_2020",
        )
        self.assertNotEqual(
            compiled["sources"][0]["select"]["temporal"]["output_mode"],
            "static",
        )

    def test_feature_compiler_splits_conflicting_custom_aggregation_names(self):
        cfg = {
            "run": {
                "name": "conflicting_aggregations",
                "project_config": "configs/project.yaml",
                "crs": "EPSG:3035",
                "aoi_config": "configs/aoi/experimental_pallars_sobira.yaml",
                "resolution_m": 100,
                "stages": ["build"],
            },
            "features": [
                {
                    "name": "built_surface_mean_2010_2020",
                    "build_type": "source_layer",
                    "source": {
                        "kind": "source",
                        "source_id": "ghsl_ghs_built_s_r2023a",
                        "config": "configs/sources/ghsl/ghsl_ghs_built_s_r2023a.yaml",
                        "variable": "built_surface",
                        "query": {
                            "variable": "built_surface",
                            "aggregation_name": "built_mean",
                        },
                        "temporal": {
                            "output_mode": "aggregate",
                            "aggregations": {
                                "custom": [
                                    {
                                        "name": "built_mean",
                                        "form": "year_range_metric",
                                        "variables": ["built_surface"],
                                        "years": [2010, 2020],
                                        "metric": "mean",
                                    }
                                ]
                            },
                        },
                    },
                },
                {
                    "name": "built_surface_mean_2015_2020",
                    "build_type": "source_layer",
                    "source": {
                        "kind": "source",
                        "source_id": "ghsl_ghs_built_s_r2023a",
                        "config": "configs/sources/ghsl/ghsl_ghs_built_s_r2023a.yaml",
                        "variable": "built_surface",
                        "query": {
                            "variable": "built_surface",
                            "aggregation_name": "built_mean",
                        },
                        "temporal": {
                            "output_mode": "aggregate",
                            "aggregations": {
                                "custom": [
                                    {
                                        "name": "built_mean",
                                        "form": "year_range_metric",
                                        "variables": ["built_surface"],
                                        "years": [2015, 2020],
                                        "metric": "mean",
                                    }
                                ]
                            },
                        },
                    },
                },
            ],
            "outputs": {"dataset_dir": "data/processed/conflicting_aggregations"},
        }

        compiled = compile_run_config(cfg)
        self.assertEqual(len(compiled["sources"]), 2)
        self.assertNotEqual(compiled["sources"][0]["id"], compiled["sources"][1]["id"])
        self.assertEqual(len(compiled["derived_features"]), 2)

    def test_legacy_run_configs_without_features_fail_clearly(self):
        cfg = {
            "run": {
                "name": "legacy_sources",
                "project_config": "configs/project.yaml",
                "aoi_config": "configs/aoi/experimental_pallars_sobira.yaml",
                "resolution_m": 100,
            },
            "sources": [
                {
                    "id": "dem",
                    "config": "configs/sources/copernicus/copernicus_dem_glo30.yaml",
                    "select": {"variables": ["elevation"]},
                }
            ],
        }

        with self.assertRaisesRegex(ValueError, "top-level 'sources'.*feature-oriented 'features'"):
            validate_run_config(cfg)

    def test_new_connectors_are_registered(self):
        connectors = set(list_source_connectors())
        self.assertIn("openstreetmap", connectors)
        self.assertIn("ghsl", connectors)
        self.assertIn("esa_cci", connectors)
        self.assertIn("esa_worldcover", connectors)

    def test_value_semantics_include_binary_and_circular(self):
        self.assertIn("binary", VALUE_SEMANTICS)
        self.assertIn("circular", VALUE_SEMANTICS)
        self.assertIn("ratio", VALUE_SEMANTICS)

    def test_feature_compiler_infers_value_semantics(self):
        cfg = {
            "run": {
                "name": "inferred_semantics",
                "project_config": "configs/project.yaml",
                "crs": "EPSG:3035",
                "aoi_config": "configs/aoi/experimental_pallars_sobira.yaml",
                "resolution_m": 100,
                "stages": ["build"],
            },
            "features": [
                {
                    "name": "tree_cover_density",
                    "build_type": "source_layer",
                    "source": {
                        "kind": "source",
                        "source_id": "copernicus_clms_forest",
                        "config": "configs/sources/copernicus/copernicus_clms_forest.yaml",
                        "variable": "tree_cover_density",
                    },
                },
                {
                    "name": "broadleaved_fraction",
                    "build_type": "source_layer",
                    "source": {
                        "kind": "source",
                        "source_id": "copernicus_clms_forest",
                        "config": "configs/sources/copernicus/copernicus_clms_forest.yaml",
                        "variable": "dominant_leaf_type",
                        "query": {"variable": "broadleaved_fraction"},
                        "category_fraction": {
                            "variable": "dominant_leaf_type",
                            "name": "broadleaved_fraction",
                            "class_values": [1],
                            "resampling": "average",
                        },
                        "resampling": "average",
                    },
                },
                {
                    "name": "broadleaved_mask",
                    "build_type": "masking",
                    "recipe": "class_mask",
                    "parameters": {"class_value": 1},
                    "inputs": {
                        "x": {
                            "kind": "source",
                            "source_id": "copernicus_clms_forest",
                            "config": "configs/sources/copernicus/copernicus_clms_forest.yaml",
                            "variable": "dominant_leaf_type",
                        }
                    },
                },
            ],
            "outputs": {"dataset_dir": "data/processed/inferred_semantics"},
        }

        compiled = compile_run_config(cfg)
        by_name = {item["name"]: item for item in compiled["derived_features"]}
        self.assertEqual(by_name["tree_cover_density"]["value_semantics"], "percentage")
        self.assertEqual(by_name["tree_cover_density"]["output_dtype"], "float32")
        self.assertEqual(by_name["broadleaved_fraction"]["value_semantics"], "fraction")
        self.assertEqual(by_name["broadleaved_fraction"]["output_dtype"], "float32")
        self.assertEqual(by_name["broadleaved_mask"]["value_semantics"], "binary")
        self.assertEqual(by_name["broadleaved_mask"]["output_dtype"], "uint8")

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
        self.assertEqual(
            catalogs["ghsl_ghs_pop_r2023a"]["variables"][0]["description"],
            "Resident population count",
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
        self.assertEqual(
            [item["name"] for item in catalogs["copernicus_hrsi_snow"]["variables"]],
            ["snow_fraction"],
        )
        self.assertEqual(
            [item["name"] for item in catalogs["ghsl_ghs_built_s_r2023a"]["variables"]],
            ["built_surface", "built_surface_non_residential"],
        )
        self.assertEqual(
            catalogs["ghsl_ghs_built_s_r2023a"]["source_resolution_options"],
            ["100m", "1000m", "3arcs", "30arcs"],
        )
        self.assertEqual(
            [item["name"] for item in catalogs["ghsl_ghs_smod_r2023a"]["variables"]],
            ["settlement_model"],
        )
        self.assertEqual(
            catalogs["ghsl_ghs_smod_r2023a"]["variables"][0]["description"],
            "GHSL settlement model class",
        )
        self.assertEqual(
            [item["name"] for item in catalogs["esa_worldcover_land_cover_10m"]["variables"]],
            ["worldcover"],
        )
        worldcover_classes = {
            item["name"]
            for item in catalogs["esa_worldcover_land_cover_10m"]["variables"][0]["category_classes"]
        }
        self.assertIn("tree_cover", worldcover_classes)
        self.assertIn("open_low_vegetation", worldcover_classes)
        self.assertEqual(
            catalogs["esa_worldcover_land_cover_10m"]["temporal"]["temporal_layers"]["years"],
            [2020, 2021],
        )
        self.assertEqual(
            catalogs["esa_worldcover_land_cover_10m"]["temporal"]["output_modes"],
            ["supplied_layers"],
        )
        self.assertEqual(
            catalogs["copernicus_clms_clcplus_backbone"]["temporal"]["temporal_layers"]["years"],
            [2018, 2021, 2023],
        )
        self.assertEqual(
            catalogs["copernicus_clms_clcplus_backbone"]["temporal"]["output_modes"],
            ["supplied_layers"],
        )
        self.assertEqual(
            catalogs["ghsl_ghs_smod_r2023a"]["temporal"]["output_modes"],
            ["supplied_layers"],
        )
        hrvpp_variables = {
            item["name"]
            for item in catalogs["copernicus_clms_hrvpp_vpp_laea"]["variables"]
        }
        self.assertIn("total_productivity", hrvpp_variables)
        self.assertIn("start_of_season_day", hrvpp_variables)
        hrvpp_descriptions = {
            item["name"]: item["description"]
            for item in catalogs["copernicus_clms_hrvpp_vpp_laea"]["variables"]
        }
        self.assertEqual(hrvpp_descriptions["amplitude"], "Seasonal amplitude")
        self.assertNotIn("season 1", hrvpp_descriptions["amplitude"])
        self.assertNotIn("2017", hrvpp_descriptions["amplitude"])
        self.assertEqual(
            catalogs["copernicus_clms_hrvpp_vpp_laea"]["dimensions"],
            {"growth_season": ["s1", "s2"]},
        )
        self.assertIn(
            "aggregate",
            catalogs["copernicus_clms_hrvpp_vpp_laea"]["temporal"]["output_modes"],
        )
        forest_variables = {
            item["name"]: item
            for item in catalogs["copernicus_clms_forest"]["variables"]
        }
        self.assertEqual(
            [item["name"] for item in forest_variables["dominant_leaf_type"]["category_classes"]],
            ["broadleaved_forest", "coniferous_forest"],
        )
        self.assertTrue(
            catalogs["copernicus_hrsi_snow"]["temporal"]["supports_custom_aggregations"]
        )
        self.assertEqual(
            catalogs["copernicus_hrsi_snow"]["temporal"]["aggregation_stage"],
            "download_postprocess",
        )
        worldclim_cmip6_dimensions = catalogs["worldclim_cmip6_future"]["dimensions"]
        self.assertEqual(len(worldclim_cmip6_dimensions["gcms"]), 25)
        self.assertEqual(
            worldclim_cmip6_dimensions["ssps"],
            ["ssp126", "ssp245", "ssp370", "ssp585"],
        )
        self.assertEqual(
            worldclim_cmip6_dimensions["periods"],
            ["2021-2040", "2041-2060", "2061-2080", "2081-2100"],
        )

    def test_workbench_compiler_accepts_new_source_selections(self):
        cfg = {
            "_compiled_from_features": True,
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

    def test_worldcover_category_fraction_selection(self):
        source_cfg = expand_source_config(
            load_yaml("configs/sources/esa_worldcover/esa_worldcover_land_cover_10m.yaml")
        )
        compiled = compile_source_config_for_run(
            source_cfg,
            {
                "select": {
                    "variables": [],
                    "temporal": {
                        "output_mode": "supplied_layers",
                        "layers": {"years": [2021]},
                    },
                    "category_fractions": [
                        {
                            "variable": "worldcover",
                            "name": "worldcover_tree_cover_fraction",
                            "class_values": [10],
                        },
                        {
                            "variable": "worldcover",
                            "name": "worldcover_open_low_vegetation_fraction",
                            "class_values": [20, 30],
                        },
                    ],
                },
            },
        )

        self.assertFalse(compiled["variables"]["worldcover_2021"]["build_output_enabled"])
        self.assertTrue(compiled["variables"]["worldcover_2021"]["enabled"])
        self.assertEqual(
            [item["name"] for item in compiled["category_fractions"]],
            [
                "worldcover_tree_cover_fraction",
                "worldcover_open_low_vegetation_fraction",
            ],
        )

        specs = get_download_file_specs(compiled)
        self.assertEqual(len(specs), 1)
        self.assertEqual(len(specs[0]["urls"]), 6)
        self.assertEqual(specs[0]["postprocess"], "mosaic_mixed_geotiff")
        self.assertIn("v200", specs[0]["urls"][0])

    def test_hrvpp_yearly_templates_and_aggregations(self):
        source_cfg = expand_source_config(
            load_yaml("configs/sources/copernicus/copernicus_clms_hrvpp_vpp_laea.yaml")
        )
        compiled = compile_source_config_for_run(
            source_cfg,
            {
                "select": {
                    "variables": ["total_productivity"],
                    "dimensions": {"growth_season": ["s1"]},
                    "temporal": {
                        "output_mode": "supplied_layers",
                        "layers": {"years": [2020]},
                    },
                },
            },
        )
        enabled = [
            name
            for name, item in compiled["variables"].items()
            if item.get("enabled")
        ]
        self.assertEqual(enabled, ["total_productivity_s1_2020"])

        spec = get_copernicus_download_file_specs(compiled)[0]
        self.assertIn("total_productivity_s1_2020_10m", spec["filename"])
        self.assertEqual(spec["file_pattern"], "*_s1_TPROD.tif")
        self.assertEqual(spec["hda_query"]["productType"], "TPROD")
        self.assertEqual(spec["hda_query"]["resolution"], "10")
        self.assertEqual(spec["hda_query"]["start"], "2020-01-01T00:00:00.000Z")

        aggregate_compiled = compile_source_config_for_run(
            source_cfg,
            {
                "select": {
                    "variables": ["total_productivity"],
                    "dimensions": {"growth_season": ["s1"]},
                    "temporal": {
                        "output_mode": "aggregate",
                        "aggregations": {
                            "custom": [
                                {
                                    "name": "mean",
                                    "form": "year_range_metric",
                                    "years": [2017, 2024],
                                    "metric": "mean",
                                    "variables": ["total_productivity"],
                                }
                            ]
                        },
                    },
                },
            },
        )
        aggregate_compiled = expand_source_config(aggregate_compiled)
        enabled = [
            name
            for name, item in aggregate_compiled["variables"].items()
            if item.get("enabled")
        ]
        self.assertEqual(
            enabled,
            [f"total_productivity_s1_{year}" for year in range(2017, 2025)],
        )

        cfg = {
            "_compiled_from_features": True,
            "run": {
                "name": "hrvpp_aggregation_smoke",
                "project_config": "configs/project.yaml",
                "crs": "EPSG:3035",
                "aoi_config": "configs/aoi/experimental_pallars_sobira.yaml",
                "resolution_m": 100,
                "stages": ["build"],
            },
            "sources": [
                {
                    "id": "hrvpp",
                    "config": "configs/sources/copernicus/copernicus_clms_hrvpp_vpp_laea.yaml",
                    "stages": ["build"],
                    "select": {
                        "variables": ["total_productivity"],
                        "dimensions": {"growth_season": ["s1", "s2"]},
                        "temporal": {
                            "output_mode": "aggregate",
                            "aggregations": {
                                "custom": [
                                    {
                                        "name": "mean_2020_2022",
                                        "form": "year_range_metric",
                                        "years": [2020, 2022],
                                        "metric": "mean",
                                        "variables": ["total_productivity"],
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
        self.assertEqual(report["estimated_layers"], 2)

    def test_yearly_category_fractions_follow_temporal_years(self):
        source_cfg = expand_source_config(
            load_yaml("configs/sources/ghsl/ghsl_ghs_smod_r2023a.yaml")
        )
        compiled = compile_source_config_for_run(
            source_cfg,
            {
                "select": {
                    "variables": [],
                    "temporal": {
                        "output_mode": "supplied_layers",
                        "layers": {"years": [2020]},
                    },
                    "category_fractions": [
                        {
                            "variable": "settlement_model",
                            "name": "urban_or_suburban_fraction",
                            "class_values": [21, 22, 23, 30],
                        },
                    ],
                },
            },
        )

        self.assertEqual(
            [item["name"] for item in compiled["category_fractions"]],
            ["urban_or_suburban_fraction"],
        )
        self.assertEqual(
            compiled["category_fractions"][0]["variable"],
            "settlement_model_2020",
        )
        self.assertTrue(compiled["variables"]["settlement_model_2020"]["enabled"])
        self.assertFalse(
            compiled["variables"]["settlement_model_2020"]["build_output_enabled"]
        )

    def test_yearly_category_fraction_accepts_catalog_generated_from_reference(self):
        cases = [
            (
                "configs/sources/copernicus/copernicus_clms_clcplus_backbone.yaml",
                "clcplus_backbone",
                2018,
                [8, 9],
                "clcplus_backbone_2018",
            ),
            (
                "configs/sources/esa_worldcover/esa_worldcover_land_cover_10m.yaml",
                "worldcover",
                2021,
                [20, 30],
                "worldcover_2021",
            ),
            (
                "configs/sources/ghsl/ghsl_ghs_smod_r2023a.yaml",
                "settlement_model",
                2020,
                [21, 22, 23, 30],
                "settlement_model_2020",
            ),
        ]

        for config_path, group_name, year, class_values, expected_variable in cases:
            with self.subTest(config_path=config_path):
                source_cfg = expand_source_config(load_yaml(config_path))
                compiled = compile_source_config_for_run(
                    source_cfg,
                    {
                        "select": {
                            "variables": [],
                            "temporal": {
                                "output_mode": "supplied_layers",
                                "layers": {"years": [year]},
                            },
                            "category_fractions": [
                                {
                                    "variable": f"variable_groups.{group_name}",
                                    "name": "category_fraction",
                                    "class_values": class_values,
                                    "resampling": "average",
                                },
                            ],
                        },
                    },
                )

                self.assertEqual(
                    compiled["category_fractions"][0]["variable"],
                    expected_variable,
                )
                self.assertTrue(compiled["variables"][expected_variable]["enabled"])
                self.assertFalse(
                    compiled["variables"][expected_variable]["build_output_enabled"]
                )

    def test_validation_errors_include_final_feature_context(self):
        cfg = {
            "run": {
                "name": "bad_fraction_context",
                "project_config": "configs/project.yaml",
                "crs": "EPSG:3035",
                "aoi_config": "configs/aoi/experimental_pallars_sobira.yaml",
                "resolution_m": 100,
                "stages": ["build"],
            },
            "features": [
                {
                    "name": "habitat_bad_fraction",
                    "build_type": "source_layer",
                    "source": {
                        "kind": "source",
                        "source_id": "copernicus_clms_clcplus_backbone",
                        "config": "configs/sources/copernicus/copernicus_clms_clcplus_backbone.yaml",
                        "variable": "clcplus_backbone",
                        "query": {"variable": "bad_fraction"},
                        "temporal": {
                            "output_mode": "supplied_layers",
                            "layers": {"years": [2018]},
                        },
                        "category_fraction": {
                            "variable": "variable_groups.not_a_real_group",
                            "name": "bad_fraction",
                            "class_values": [8, 9],
                            "resampling": "average",
                        },
                    },
                }
            ],
            "outputs": {"dataset_dir": "data/processed/bad_fraction_context"},
        }

        report = validate_researcher_run_config(cfg)
        self.assertFalse(report["ok"])
        self.assertIn("Source 'copernicus_clms_clcplus_backbone' failed validation.", report["errors"][0])
        self.assertIn("Used by final feature(s): habitat_bad_fraction.", report["errors"][0])
        self.assertIn("Problem:", report["errors"][0])

    def test_categorical_yearly_collections_reject_numeric_aggregations(self):
        cfg = {
            "_compiled_from_features": True,
            "run": {
                "name": "bad_smod_aggregation",
                "project_config": "configs/project.yaml",
                "crs": "EPSG:3035",
                "aoi_config": "configs/aoi/experimental_pallars_sobira.yaml",
                "resolution_m": 100,
                "stages": ["build"],
            },
            "sources": [
                {
                    "id": "smod",
                    "config": "configs/sources/ghsl/ghsl_ghs_smod_r2023a.yaml",
                    "stages": ["build"],
                    "select": {
                        "variables": ["settlement_model"],
                        "temporal": {
                            "output_mode": "aggregate",
                            "aggregations": {
                                "custom": [
                                    {
                                        "name": "mean_2020_2025",
                                        "form": "year_range_metric",
                                        "years": [2020, 2025],
                                        "metric": "mean",
                                        "variables": ["settlement_model"],
                                    }
                                ]
                            },
                        },
                    },
                },
            ],
        }
        report = validate_researcher_run_config(cfg)
        self.assertFalse(report["ok"])
        self.assertIn("Temporal output_mode 'aggregate'", report["errors"][0])

    def test_ghsl_built_surface_resolution_labels_are_mapped_to_provider_tokens(self):
        cfg = expand_source_config(
            load_yaml("configs/sources/ghsl/ghsl_ghs_built_s_r2023a.yaml")
        )
        compiled = compile_source_config_for_run(
            cfg,
            {
                "overrides": {"processing": {"source_resolution": "30arcs"}},
                "select": {
                    "variables": ["built_surface"],
                    "temporal": {
                        "output_mode": "supplied_layers",
                        "layers": {"years": [2020]},
                    },
                },
            },
        )
        spec = get_download_file_specs(compiled)[0]
        self.assertIn("_30arcs_", spec["filename"])
        self.assertIn("4326_30ss", spec["urls"][0])
        self.assertIn("4326_30ss", spec["zip_member_pattern"])

    def test_yearly_static_collections_support_base_variable_aggregations(self):
        cfg = {
            "_compiled_from_features": True,
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

    def test_yearly_static_aggregations_reject_unavailable_endpoint_years(self):
        cfg = {
            "_compiled_from_features": True,
            "run": {
                "name": "bad_yearly_endpoint",
                "project_config": "configs/project.yaml",
                "crs": "EPSG:3035",
                "aoi_config": "configs/aoi/experimental_pallars_sobira.yaml",
                "resolution_m": 100,
                "stages": ["build"],
            },
            "sources": [
                {
                    "id": "population",
                    "config": "configs/sources/ghsl/ghsl_ghs_pop_r2023a.yaml",
                    "stages": ["build"],
                    "select": {
                        "variables": ["population_count"],
                        "temporal": {
                            "output_mode": "aggregate",
                            "aggregations": {
                                "custom": [
                                    {
                                        "name": "mean_2006_2009",
                                        "form": "year_range_metric",
                                        "years": [2006, 2009],
                                        "metric": "mean",
                                        "variables": ["population_count"],
                                    }
                                ]
                            },
                        },
                    },
                },
            ],
        }

        report = validate_researcher_run_config(cfg)
        self.assertFalse(report["ok"])
        self.assertIn("available source years", report["errors"][0])

    def test_copernicus_snow_postprocess_aggregations_are_run_configured(self):
        source_cfg = expand_source_config(
            load_yaml("configs/sources/copernicus/copernicus_hrsi_snow.yaml")
        )
        compiled = compile_source_config_for_run(
            source_cfg,
            {
                "select": {
                    "variables": ["snow_fraction"],
                    "temporal": {
                        "output_mode": "postprocess_aggregate",
                        "aggregations": {
                            "use": ["snow_fraction_mean_winter"],
                            "custom": [
                                {
                                    "name": "snow_days_jan_mar",
                                    "metric": "count_threshold",
                                    "months": [1, 3],
                                    "years": [2022, 2022],
                                    "threshold": 50,
                                    "comparison": ">=",
                                    "variables": ["snow_fraction"],
                                }
                            ],
                        },
                    },
                },
            },
        )
        compiled = expand_source_config(compiled)

        output_variables = compiled["temporal_postprocess"]["output_variables"]
        self.assertEqual(
            sorted(output_variables),
            ["snow_days_jan_mar", "snow_fraction_mean_winter"],
        )
        self.assertEqual(output_variables["snow_days_jan_mar"]["method"], "count_threshold")
        self.assertEqual(output_variables["snow_days_jan_mar"]["months"], [1, 2, 3])
        self.assertEqual(output_variables["snow_days_jan_mar"]["source_variable"], "snow_fraction")
        self.assertEqual(output_variables["snow_days_jan_mar"]["variables"], ["snow_fraction"])

        enabled = [
            name
            for name, item in compiled["variables"].items()
            if item.get("enabled")
        ]
        self.assertEqual(
            sorted(enabled),
            ["snow_days_jan_mar", "snow_fraction_mean_winter"],
        )
        self.assertFalse(compiled["variables"]["snow_fraction"].get("enabled"))

        hda_query = compiled["download"]["files"]["snow_scenes"]["hda_query"]
        self.assertEqual(hda_query["startdate"], "2022-01-01T00:00:00.000Z")
        self.assertEqual(hda_query["enddate"], "2022-12-31T23:59:59.999Z")

        exact_only = compile_source_config_for_run(
            source_cfg,
            {
                "select": {
                    "variables": ["snow_fraction"],
                    "temporal": {
                        "output_mode": "postprocess_aggregate",
                        "aggregations": {
                            "custom": [
                                {
                                    "name": "snow_days_feb_first_half",
                                    "metric": "count_threshold",
                                    "start_date": "2022-02-01",
                                    "end_date": "2022-02-15",
                                    "threshold": 50,
                                    "comparison": ">=",
                                    "variables": ["snow_fraction"],
                                }
                            ],
                        },
                    },
                },
            },
        )
        output_cfg = exact_only["temporal_postprocess"]["output_variables"]["snow_days_feb_first_half"]
        self.assertEqual(output_cfg["months"], [2])
        self.assertEqual(output_cfg["start_date"], "2022-02-01")
        self.assertEqual(output_cfg["end_date"], "2022-02-15")
        exact_query = exact_only["download"]["files"]["snow_scenes"]["hda_query"]
        self.assertEqual(exact_query["startdate"], "2022-02-01T00:00:00.000Z")
        self.assertEqual(exact_query["enddate"], "2022-02-15T23:59:59.999Z")

    def test_copernicus_snow_postprocess_rejects_unknown_source_variable(self):
        source_cfg = expand_source_config(
            load_yaml("configs/sources/copernicus/copernicus_hrsi_snow.yaml")
        )

        with self.assertRaisesRegex(ValueError, "Unknown postprocess source variable"):
            compile_source_config_for_run(
                source_cfg,
                {
                    "select": {
                        "variables": ["snow_fraction"],
                        "temporal": {
                            "output_mode": "postprocess_aggregate",
                            "aggregations": {
                                "custom": [
                                    {
                                        "name": "bad_source_variable",
                                        "metric": "mean",
                                        "months": [1, 3],
                                        "years": [2022, 2022],
                                        "variables": ["not_snow_fraction"],
                                    }
                                ],
                            },
                        },
                    },
                },
            )

    def test_categorical_category_fractions_can_replace_source_variable(self):
        source_cfg = expand_source_config(
            load_yaml("configs/sources/copernicus/copernicus_clms_forest.yaml")
        )
        compiled = compile_source_config_for_run(
            source_cfg,
            {
                "overrides": {
                    "resampling": {
                        "by_variable": {
                            "dominant_leaf_type_fraction_broadleaved": "nearest"
                        }
                    }
                },
                "select": {
                    "variables": [],
                    "category_fractions": [
                        {
                            "variable": "dominant_leaf_type",
                            "name": "dominant_leaf_type_fraction_broadleaved",
                            "class_values": [1],
                            "label": "Broadleaved forest fraction",
                        },
                        {
                            "variable": "dominant_leaf_type",
                            "name": "dominant_leaf_type_fraction_coniferous",
                            "class_values": [2],
                            "label": "Coniferous forest fraction",
                        },
                    ],
                },
            },
        )

        self.assertEqual(
            [item["name"] for item in compiled["category_fractions"]],
            [
                "dominant_leaf_type_fraction_broadleaved",
                "dominant_leaf_type_fraction_coniferous",
            ],
        )
        self.assertTrue(compiled["variables"]["dominant_leaf_type"]["enabled"])
        self.assertFalse(compiled["variables"]["dominant_leaf_type"]["build_output_enabled"])
        self.assertEqual(compiled["category_fractions"][0]["resampling"], "nearest")

        cfg = {
            "_compiled_from_features": True,
            "run": {
                "name": "category_fraction_smoke",
                "project_config": "configs/project.yaml",
                "crs": "EPSG:3035",
                "aoi_config": "configs/aoi/experimental_pallars_sobira.yaml",
                "resolution_m": 100,
                "stages": ["build"],
            },
            "sources": [
                {
                    "id": "forest",
                    "config": "configs/sources/copernicus/copernicus_clms_forest.yaml",
                    "stages": ["build"],
                    "select": {
                        "variables": [],
                        "category_fractions": [
                            {
                                "variable": "dominant_leaf_type",
                                "name": "dominant_leaf_type_fraction_broadleaved",
                                "class_values": [1],
                            },
                            {
                                "variable": "dominant_leaf_type",
                                "name": "dominant_leaf_type_fraction_coniferous",
                                "class_values": [2],
                            },
                        ],
                    },
                }
            ],
        }
        report = validate_researcher_run_config(cfg)
        self.assertTrue(report["ok"], report["errors"])
        self.assertEqual(report["estimated_layers"], 2)

    def test_feature_compiler_merges_category_fraction_source_requirements(self):
        cfg = {
            "run": {
                "name": "merged_fraction_requirements",
                "project_config": "configs/project.yaml",
                "crs": "EPSG:3035",
                "aoi_config": "configs/aoi/experimental_pallars_sobira.yaml",
                "resolution_m": 100,
                "stages": ["build"],
            },
            "features": [
                {
                    "name": "leaf_type_fractions",
                    "build_type": "source_layer",
                    "outputs": [
                        {
                            "name": "broadleaved_fraction",
                            "source": {
                                "kind": "source",
                                "source_id": "copernicus_clms_forest",
                                "config": "configs/sources/copernicus/copernicus_clms_forest.yaml",
                                "variable": "dominant_leaf_type",
                                "query": {"variable": "broadleaved_fraction"},
                                "category_fraction": {
                                    "variable": "dominant_leaf_type",
                                    "name": "broadleaved_fraction",
                                    "class_values": [1],
                                    "resampling": "average",
                                },
                                "resampling": "average",
                            },
                        },
                        {
                            "name": "coniferous_fraction",
                            "source": {
                                "kind": "source",
                                "source_id": "copernicus_clms_forest",
                                "config": "configs/sources/copernicus/copernicus_clms_forest.yaml",
                                "variable": "dominant_leaf_type",
                                "query": {"variable": "coniferous_fraction"},
                                "category_fraction": {
                                    "variable": "dominant_leaf_type",
                                    "name": "coniferous_fraction",
                                    "class_values": [2],
                                    "resampling": "average",
                                },
                                "resampling": "average",
                            },
                        },
                    ],
                }
            ],
            "outputs": {"dataset_dir": "data/processed/merged_fraction_requirements"},
        }

        compiled = compile_run_config(cfg)
        self.assertEqual(len(compiled["sources"]), 1)
        source_entry = compiled["sources"][0]
        self.assertEqual(source_entry["select"]["variables"], [])
        self.assertEqual(len(source_entry["select"]["category_fractions"]), 2)

        source_cfg = expand_source_config(load_yaml(source_entry["config"]))
        compiled_source = compile_source_config_for_run(source_cfg, source_entry)
        self.assertTrue(compiled_source["variables"]["dominant_leaf_type"]["enabled"])
        self.assertFalse(
            compiled_source["variables"]["dominant_leaf_type"]["build_output_enabled"]
        )

    def test_feature_compiler_merges_compatible_temporal_aggregations(self):
        cfg = {
            "run": {
                "name": "merged_temporal_requirements",
                "project_config": "configs/project.yaml",
                "crs": "EPSG:3035",
                "aoi_config": "configs/aoi/experimental_pallars_sobira.yaml",
                "resolution_m": 100,
                "stages": ["build"],
            },
            "features": [
                {
                    "name": "built_surface_mean",
                    "build_type": "source_layer",
                    "source": {
                        "kind": "source",
                        "source_id": "ghsl_ghs_built_s_r2023a",
                        "config": "configs/sources/ghsl/ghsl_ghs_built_s_r2023a.yaml",
                        "variable": "built_surface",
                        "query": {
                            "variable": "built_surface",
                            "aggregation_name": "built_surface_mean",
                        },
                        "temporal": {
                            "output_mode": "aggregate",
                            "aggregations": {
                                "custom": [
                                    {
                                        "name": "built_surface_mean",
                                        "form": "year_range_metric",
                                        "variables": ["built_surface"],
                                        "years": [2010, 2020],
                                        "metric": "mean",
                                    }
                                ]
                            },
                        },
                        "source_resolution": "100m",
                        "resampling": "conservative_sum",
                    },
                },
                {
                    "name": "built_surface_non_residential_mean",
                    "build_type": "source_layer",
                    "source": {
                        "kind": "source",
                        "source_id": "ghsl_ghs_built_s_r2023a",
                        "config": "configs/sources/ghsl/ghsl_ghs_built_s_r2023a.yaml",
                        "variable": "built_surface_non_residential",
                        "query": {
                            "variable": "built_surface_non_residential",
                            "aggregation_name": "built_surface_non_residential_mean",
                        },
                        "temporal": {
                            "output_mode": "aggregate",
                            "aggregations": {
                                "custom": [
                                    {
                                        "name": "built_surface_non_residential_mean",
                                        "form": "year_range_metric",
                                        "variables": ["built_surface_non_residential"],
                                        "years": [2010, 2020],
                                        "metric": "mean",
                                    }
                                ]
                            },
                        },
                        "source_resolution": "100m",
                        "resampling": "conservative_sum",
                    },
                },
            ],
            "outputs": {"dataset_dir": "data/processed/merged_temporal_requirements"},
        }

        compiled = compile_run_config(cfg)
        self.assertEqual(len(compiled["sources"]), 1)
        source_entry = compiled["sources"][0]
        self.assertEqual(
            sorted(source_entry["select"]["variables"]),
            ["built_surface", "built_surface_non_residential"],
        )
        custom = source_entry["select"]["temporal"]["aggregations"]["custom"]
        self.assertEqual(len(custom), 2)
        self.assertEqual(
            sorted(source_entry["overrides"]["resampling"]["by_variable"]),
            ["built_surface", "built_surface_non_residential"],
        )

    def test_variable_group_expansion_preserves_output_suppression(self):
        source_cfg = expand_source_config(
            load_yaml("configs/sources/copernicus/copernicus_clms_clcplus_backbone.yaml")
        )
        compiled = compile_source_config_for_run(
            source_cfg,
            {
                "select": {
                    "variables": [],
                    "temporal": {
                        "output_mode": "supplied_layers",
                        "layers": {"years": [2018]},
                    },
                    "category_fractions": [
                        {
                            "variable": "clcplus_backbone",
                            "name": "low_woody_fraction",
                            "class_values": [5],
                        }
                    ],
                }
            },
        )
        expanded_again = expand_source_config(compiled)

        self.assertTrue(expanded_again["variables"]["clcplus_backbone_2018"]["enabled"])
        self.assertFalse(
            expanded_again["variables"]["clcplus_backbone_2018"]["build_output_enabled"]
        )
        self.assertEqual(len(expanded_again["category_fractions"]), 1)

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

    def test_worldclim_cmip6_availability_filters_partial_combinations(self):
        cfg = load_yaml("configs/sources/worldclim/worldclim_cmip6_future.yaml")
        specs = get_file_specs(cfg)

        self.assertEqual(len(specs), 1156)
        self.assertFalse(
            any(
                item["gcm"] == "FIO-ESM-2-0" and item["ssp"] == "ssp585"
                for item in specs
            )
        )
        self.assertFalse(
            any(
                item["gcm"] == "HadGEM3-GC31-LL" and item["ssp"] == "ssp585"
                for item in specs
            )
        )
        self.assertFalse(
            any(
                item["gcm"] == "GFDL-ESM4" and item["ssp"] == "ssp585"
                for item in specs
            )
        )
        self.assertFalse(
            any(
                item["gcm"] == "GFDL-ESM4"
                and item["ssp"] == "ssp370"
                and item["variable"] in {"tmin", "tmax"}
                for item in specs
            )
        )
        gfdl_prec = [
            item
            for item in specs
            if item["gcm"] == "GFDL-ESM4"
            and item["ssp"] == "ssp370"
            and item["variable"] == "prec"
        ]
        self.assertEqual(len(gfdl_prec), 4)
        self.assertIn(
            "/GFDL-ESM4/ssp370/wc2.1_10m_prec_GFDL-ESM4_ssp370_2021-2040.tif",
            build_worldclim_cmip6_download_url(cfg, gfdl_prec[0]),
        )

    def test_worldclim_cmip6_unavailable_selection_fails_before_download(self):
        cfg = load_yaml("configs/sources/worldclim/worldclim_cmip6_future.yaml")
        compiled = compile_source_config_for_run(
            cfg,
            {
                "select": {
                    "variables": ["tmin"],
                    "dimensions": {
                        "gcms": ["GFDL-ESM4"],
                        "ssps": ["ssp585"],
                        "periods": ["2021-2040"],
                    },
                }
            },
        )

        with self.assertRaisesRegex(ValueError, "No available WorldClim CMIP6 files"):
            get_file_specs(compiled)

    def test_osm_domains_follow_runner_contract(self):
        cfg = load_yaml("configs/sources/openstreetmap/openstreetmap_geofabrik_pyrenees.yaml")
        cfg["_config_path"] = "configs/sources/openstreetmap/openstreetmap_geofabrik_pyrenees.yaml"
        self.assertIn("domains", cfg)
        clip_aoi, output_aoi = _load_domain_configs(cfg)
        self.assertEqual(clip_aoi["name"], "pyrenees_full")
        self.assertEqual(output_aoi["name"], "experimental_pallars_sobira")

    def test_osm_empty_enabled_layer_fails_before_empty_raster(self):
        empty = gpd.GeoDataFrame(geometry=[], crs="EPSG:3035")
        with self.assertRaisesRegex(ValueError, "zero clipped features"):
            _build_layer_array(
                empty,
                {"output": "distance"},
                _DummyGrid(),
            )

    def test_feature_unit_inference_uses_source_units_and_log_transform(self):
        cfg = {
            "run": {
                "name": "unit_inference",
                "project_config": "configs/project.yaml",
                "crs": "EPSG:3035",
                "aoi_config": "configs/aoi/ursus_arctos_pyrenees.yaml",
                "resolution_m": 100,
                "stages": ["build"],
            },
            "features": [
                {
                    "name": "distance",
                    "build_type": "source_layer",
                    "source": {
                        "kind": "source",
                        "source_id": "openstreetmap_geofabrik_pyrenees",
                        "config": "configs/sources/openstreetmap/openstreetmap_geofabrik_pyrenees.yaml",
                        "layer": "transport.secondary_roads_distance",
                    },
                },
                {
                    "name": "log_distance",
                    "build_type": "expression",
                    "expression": "log10(x + 1)",
                    "inputs": {
                        "x": {
                            "kind": "feature",
                            "feature": "distance",
                        }
                    },
                },
            ],
        }
        compiled = compile_run_config(cfg)
        by_name = {item["name"]: item for item in compiled["derived_features"]}
        self.assertEqual(by_name["distance"]["unit"], "m")
        self.assertEqual(by_name["log_distance"]["unit"], "log10(m)")

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
            "_compiled_from_features": True,
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

    def test_download_only_feature_run_does_not_require_derived_output_contract(self):
        cfg = {
            "run": {
                "name": "download_only_feature_run",
                "project_config": "configs/project.yaml",
                "stages": ["download"],
            },
            "features": [
                {
                    "name": "elevation",
                    "build_type": "source_layer",
                    "source": {
                        "kind": "source",
                        "source_id": "copernicus_dem_glo30",
                        "config": "configs/sources/copernicus/copernicus_dem_glo30.yaml",
                        "variable": "elevation",
                    },
                }
            ],
            "outputs": {
                "dataset_dir": "data/processed/download_only_feature_run",
                "copy_rasters": False,
                "write_manifest": False,
            },
        }

        report = validate_researcher_run_config(cfg)
        self.assertTrue(report["ok"], report["errors"])


if __name__ == "__main__":
    unittest.main()
