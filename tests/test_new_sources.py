from __future__ import annotations

import unittest

from src.sources.registry import list_source_connectors
from src.workbench.catalog import list_source_catalogs
from src.workbench.compiler import validate_researcher_run_config


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
            12,
        )
        self.assertEqual(
            catalogs["ghsl_ghs_pop_r2023a"]["temporal"]["temporal_layers"]["years"],
            [1975, 1980, 1985, 1990, 1995, 2000, 2005, 2010, 2015, 2020, 2025, 2030],
        )
        self.assertEqual(
            catalogs["ghsl_ghs_pop_r2023a"]["source_resolution_options"],
            ["54009_100", "54009_1000", "4326_3ss", "4326_30ss"],
        )
        self.assertEqual(
            len(catalogs["esa_cci_biomass_agb_100m"].get("variables", [])),
            10,
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
                        "variables": ["population_count_2020"],
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
                        "variables": ["agb_2020"],
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


if __name__ == "__main__":
    unittest.main()
