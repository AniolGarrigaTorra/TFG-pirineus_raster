import unittest

import numpy as np

from src.pipeline.derived import (
    evaluate_derived_operation,
    evaluate_raster_expression,
    validate_derived_feature_definition,
)


class DerivedExpressionTests(unittest.TestCase):
    def test_expression_engine_rejects_unsafe_calls(self):
        with self.assertRaises(ValueError):
            evaluate_raster_expression(
                "__import__('os').system('echo nope')",
                {"x": np.ones((2, 2), dtype=np.float32)},
            )

    def test_expression_engine_supports_where_and_comparisons(self):
        result = evaluate_raster_expression(
            "where(x >= 2, 1, 0)",
            {"x": np.array([[1, 2], [3, np.nan]], dtype=np.float32)},
        )
        np.testing.assert_array_equal(
            result,
            np.array([[0, 1], [1, 0]], dtype=np.float32),
        )

    def test_recipe_thermal_range(self):
        result, operation, expression = evaluate_derived_operation(
            {
                "name": "thermal_range",
                "operation": "recipe",
                "recipe": "thermal_range",
                "inputs": {"tmax": {}, "tmin": {}},
            },
            {
                "tmax": np.array([[10, 12]], dtype=np.float32),
                "tmin": np.array([[1, 4]], dtype=np.float32),
            },
            grid_resolution_m=100,
        )
        self.assertEqual(operation, "recipe:thermal_range")
        self.assertEqual(expression, "tmax - tmin")
        np.testing.assert_array_equal(result, np.array([[9, 8]], dtype=np.float32))

    def test_focal_mean(self):
        result, operation, _ = evaluate_derived_operation(
            {
                "name": "focal_mean",
                "operation": "focal",
                "method": "mean",
                "parameters": {"radius": 1},
                "inputs": {"x": {}},
            },
            {
                "x": np.array(
                    [
                        [1, 1, 1],
                        [1, 9, 1],
                        [1, 1, 1],
                    ],
                    dtype=np.float32,
                )
            },
            grid_resolution_m=100,
        )
        self.assertEqual(operation, "focal:mean")
        self.assertAlmostEqual(float(result[1, 1]), 17 / 9, places=6)

    def test_terrain_slope_flat_dem_is_zero(self):
        result, operation, _ = evaluate_derived_operation(
            {
                "name": "slope",
                "operation": "terrain",
                "method": "slope",
                "inputs": {"dem": {}},
            },
            {"dem": np.ones((3, 3), dtype=np.float32) * 100},
            grid_resolution_m=30,
        )
        self.assertEqual(operation, "terrain:slope")
        np.testing.assert_allclose(result, np.zeros((3, 3), dtype=np.float32))

    def test_terrain_aspect_reports_downslope_bearing(self):
        east_rising = np.tile(np.arange(3, dtype=np.float32), (3, 1))
        result, operation, _ = evaluate_derived_operation(
            {
                "name": "aspect",
                "operation": "terrain",
                "method": "aspect",
                "inputs": {"dem": {}},
            },
            {"dem": east_rising},
            grid_resolution_m=30,
        )
        self.assertEqual(operation, "terrain:aspect")
        np.testing.assert_allclose(result, np.full((3, 3), 270.0, dtype=np.float32))

    def test_validation_accepts_non_expression_operations(self):
        warnings = validate_derived_feature_definition(
            {
                "name": "slope",
                "operation": "terrain",
                "method": "slope",
                "inputs": {"dem": {"variable": "elev"}},
            }
        )
        self.assertEqual(warnings, [])


if __name__ == "__main__":
    unittest.main()
