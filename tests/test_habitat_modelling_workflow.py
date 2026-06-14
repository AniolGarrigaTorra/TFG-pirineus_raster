import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from notebooks.ursus_arctos_project.habitat_modelling_workflow import (
    WorkflowConfig,
    create_background_pools,
    model_sample_weights,
    parse_mixed_datetime,
    select_winners,
    spatial_block_ids,
)


class HabitatModellingWorkflowTests(unittest.TestCase):
    def test_workflow_config_normalizes_papermill_boolean_strings(self):
        config = WorkflowConfig(
            smoke_mode="false",
            run_tuning="true",
            run_xgboost="0",
            run_shap="yes",
            run_map_prediction="off",
        )

        self.assertFalse(config.smoke_mode)
        self.assertTrue(config.run_tuning)
        self.assertFalse(config.run_xgboost)
        self.assertTrue(config.run_shap)
        self.assertFalse(config.run_map_prediction)

    def test_parse_mixed_datetime_recovers_european_and_iso_values(self):
        values = pd.Series(["31/12/2023", "2024-08-23 00:00:00", "01/05/2024", "2007-05-01 03:01:00"])

        parsed = parse_mixed_datetime(values)

        self.assertTrue(parsed.notna().all())
        self.assertEqual(parsed.dt.month.tolist(), [12, 8, 5, 5])

    def test_daily_weights_equalize_bears_and_days(self):
        table = pd.DataFrame(
            {
                "label": [1, 1, 1, 1, 0, 0],
                "bear_name": ["A", "A", "A", "B", "background", "background"],
                "daily_group": ["A_1", "A_1", "A_2", "B_1", "background", "background"],
            }
        )

        weights = model_sample_weights(table)

        self.assertTrue(np.isclose(weights[:3].sum(), 1.5))
        self.assertTrue(np.isclose(weights[3], 1.5))
        self.assertTrue(np.isclose(weights[4:].sum(), 3.0))
        self.assertTrue(np.isclose(weights.sum(), len(table)))

    def test_spatial_blocks_are_defined_by_presence_distribution(self):
        presences = pd.DataFrame(
            {
                "x_3035": [8, 9, 11, 12],
                "y_3035": [8, 12, 9, 11],
                "label": 1,
            }
        )
        background = pd.DataFrame(
            {
                "x_3035": np.linspace(0, 100, 100),
                "y_3035": np.linspace(100, 0, 100),
                "label": 0,
            }
        )
        table = pd.concat([presences, background], ignore_index=True)

        blocks = spatial_block_ids(table, 2)
        presence_counts = pd.Series(blocks[: len(presences)]).value_counts()

        self.assertGreater(len(presence_counts), 1)
        self.assertEqual(int(presence_counts.sum()), len(presences))

    def test_background_pools_are_unique_and_disjoint(self):
        import rasterio
        from rasterio.transform import from_origin

        with TemporaryDirectory() as directory:
            grid = Path(directory) / "grid.tif"
            profile = {
                "driver": "GTiff",
                "height": 20,
                "width": 20,
                "count": 1,
                "dtype": "uint8",
                "crs": "EPSG:3035",
                "transform": from_origin(0, 2_000, 100, 100),
            }
            with rasterio.open(grid, "w", **profile) as dst:
                dst.write(np.ones((20, 20), dtype="uint8"), 1)
            masks = {
                "full_aoi": np.ones((20, 20), dtype=bool),
                "local_domain": np.pad(np.ones((10, 10), dtype=bool), ((5, 5), (5, 5))),
            }
            config = WorkflowConfig(n_background_train=40, n_background_test=10)

            pools = create_background_pools(masks, grid, config)

        all_ids = [set(frame["cell_id"]) for frame in pools.values()]
        self.assertTrue(all(len(frame) == len(set(frame["cell_id"])) for frame in pools.values()))
        for left in range(len(all_ids)):
            for right in range(left + 1, len(all_ids)):
                self.assertFalse(all_ids[left] & all_ids[right])

    def test_winner_rule_prioritizes_robust_floor_and_boyce_guardrail(self):
        rows = []
        for candidate, local_auc, full_auc, local_min, boyce in [
            ("rf", 0.78, 0.84, 0.62, 0.10),
            ("xgb", 0.85, 0.90, 0.45, -0.20),
        ]:
            for scenario, auc in [("local_test", local_auc), ("full_test", full_auc)]:
                rows.append(
                    {
                        "training_domain": "local_domain",
                        "cub_class": "lt6",
                        "candidate": candidate,
                        "algorithm": candidate,
                        "evaluation_scenario": scenario,
                        "auc_roc_mean": auc,
                        "auc_roc_min": local_min if scenario == "local_test" else auc - 0.1,
                        "boyce_mean": boyce,
                        "brier_mean": 0.1,
                    }
                )

        winner = select_winners(pd.DataFrame(rows)).iloc[0]

        self.assertEqual(winner["candidate"], "rf")
        self.assertEqual(winner["reliability_status"], "management_ready")


if __name__ == "__main__":
    unittest.main()
