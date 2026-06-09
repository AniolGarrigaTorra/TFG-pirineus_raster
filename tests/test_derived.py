import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import from_origin

from src.pipeline.derived import (
    _evaluate_native_then_resample,
    evaluate_derived_operation,
    evaluate_raster_expression,
    validate_derived_feature_definition,
)
from src.pipeline.layers import LayerSpec
from src.pipeline.raster_ops import GridContext, read_raster_to_grid


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

    def test_expression_engine_supports_nan_constant(self):
        result = evaluate_raster_expression(
            "where(x > 0, x, nan)",
            {"x": np.array([[1, -1], [0, 2]], dtype=np.float32)},
        )

        self.assertTrue(np.isnan(result[0, 1]))
        self.assertTrue(np.isnan(result[1, 0]))
        np.testing.assert_array_equal(
            result[[0, 1], [0, 1]],
            np.array([1, 2], dtype=np.float32),
        )

    def test_expression_engine_rejects_wrong_function_arity(self):
        with self.assertRaisesRegex(ValueError, "clip"):
            evaluate_raster_expression(
                "clip(x)",
                {"x": np.ones((2, 2), dtype=np.float32)},
            )

        with self.assertRaisesRegex(ValueError, "minimum"):
            evaluate_raster_expression(
                "minimum(x)",
                {"x": np.ones((2, 2), dtype=np.float32)},
            )

    def test_expression_engine_rejects_bare_function_name(self):
        with self.assertRaisesRegex(ValueError, "must be called"):
            evaluate_raster_expression(
                "clip",
                {"x": np.ones((2, 2), dtype=np.float32)},
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

    def test_validation_rejects_unknown_evaluation_stage(self):
        with self.assertRaises(ValueError):
            validate_derived_feature_definition(
                {
                    "name": "slope",
                    "operation": "terrain",
                    "method": "slope",
                    "evaluation_stage": "somewhere_else",
                    "inputs": {"dem": {"variable": "elev"}},
                }
            )

    def test_native_then_resample_terrain_uses_native_detail_before_target_grid(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source_path = tmp_path / "dem_1m.tif"
            target_path = tmp_path / "dem_2m.tif"
            crs = CRS.from_epsg(3035)

            source_dem = np.array(
                [
                    [0, 0, 0, 0],
                    [0, 20, 20, 0],
                    [0, 20, 20, 0],
                    [0, 0, 0, 0],
                ],
                dtype=np.float32,
            )

            source_profile = {
                "driver": "GTiff",
                "height": 4,
                "width": 4,
                "count": 1,
                "dtype": "float32",
                "crs": crs,
                "transform": from_origin(0, 4, 1, 1),
                "nodata": -9999.0,
            }
            with rasterio.open(source_path, "w", **source_profile) as dst:
                dst.write(source_dem, 1)

            target_grid = GridContext(
                path=target_path,
                profile={**source_profile, "height": 2, "width": 2, "transform": from_origin(0, 4, 2, 2)},
                transform=from_origin(0, 4, 2, 2),
                crs=crs,
                height=2,
                width=2,
                resolution_m=2,
                aoi_name="test",
            )

            target_dem = read_raster_to_grid(
                raster_path=source_path,
                grid=target_grid,
                resampling=Resampling.average,
                resampling_method_name="average",
            )
            with rasterio.open(target_path, "w", **target_grid.profile) as dst:
                dst.write(target_dem, 1)

            layer = LayerSpec(
                name="dem",
                path=target_path,
                provider="test",
                product="dem",
                source_id="dem",
                variable="elev",
                resolution_m=2,
                crs=str(crs),
                metadata={
                    "source_clipped_path": str(source_path),
                    "native_resolution_m": 1,
                    "value_semantics": "intensive",
                    "resampling": "bilinear",
                },
            )
            derived_cfg = {
                "name": "slope",
                "operation": "terrain",
                "method": "slope",
                "evaluation_stage": "native_then_resample",
                "post_resampling": "average",
                "inputs": {"dem": {"variable": "elev"}},
            }

            native_result, _, _, metadata, warnings = _evaluate_native_then_resample(
                derived_cfg=derived_cfg,
                input_layers={"dem": layer},
                target_grid=target_grid,
                target_resolution_m=2,
            )
            target_result, _, _ = evaluate_derived_operation(
                derived_cfg={**derived_cfg, "evaluation_stage": "target_grid"},
                input_arrays={"dem": target_dem},
                grid_resolution_m=2,
            )

            self.assertEqual(metadata["evaluation_stage"], "native_then_resample")
            self.assertEqual(metadata["evaluation_resolution_m"], 1.0)
            self.assertEqual(warnings, [])
            self.assertEqual(native_result.shape, target_result.shape)
            self.assertFalse(np.allclose(native_result, target_result, equal_nan=True))


if __name__ == "__main__":
    unittest.main()
