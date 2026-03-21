"""
tests/test_transformer.py
-------------------------
Verifies that `src.data.transformer` correctly resamples, merges rasters,
and burns multiple vectors into masks.
"""

import sys
import shutil
import json
from pathlib import Path

# Ensure project root is on the path regardless of how the script is invoked
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.transformer import unify_and_merge_rasters, rasterize_vectors
import geopandas as gpd
import numpy as np
import rasterio
from shapely.geometry import box


def _make_dummy_raster(path: Path, res: float, count: int = 1) -> None:
    # 0 to 100 coverage
    width = int(100 / res)
    height = int(100 / res)
    transform = rasterio.transform.from_origin(0.0, 100.0, res, res)
    
    data = np.random.randint(0, 255, (count, height, width), dtype=np.uint8)
    
    with rasterio.open(
        path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=count,
        dtype=np.uint8,
        crs='EPSG:4326',
        transform=transform
    ) as dst:
        dst.write(data)
        dst.descriptions = tuple([f"band_{i}_res{res}" for i in range(1, count + 1)])


def _make_dummy_vector(path: Path, is_binary: bool) -> None:
    # Overlapping 0,0 to 50,50
    poly1 = box(10, 10, 30, 30)
    poly2 = box(60, 60, 80, 80)
    
    if is_binary:
        gdf = gpd.GeoDataFrame({"value": [1, 2]}, geometry=[poly1, poly2], crs="EPSG:4326")
    else:
        gdf = gpd.GeoDataFrame({"class_name": ["Forest", "Water"]}, geometry=[poly1, poly2], crs="EPSG:4326")
        
    gdf.to_file(path, driver="GPKG")


def test_transformer():
    tmp_dir = Path("tests/temp_transformer")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)
    
    try:
        # --- Create Datasets ---
        raster1 = tmp_dir / "r1_10m.tif"
        raster2 = tmp_dir / "r2_20m.tif"
        _make_dummy_raster(raster1, 10.0, count=1)
        _make_dummy_raster(raster2, 20.0, count=2)
        
        vector_bin = tmp_dir / "v_binary.gpkg"
        vector_multi = tmp_dir / "v_multi.gpkg"
        _make_dummy_vector(vector_bin, is_binary=True)
        _make_dummy_vector(vector_multi, is_binary=False)
        
        merged_out = tmp_dir / "merged_features.tif"
        
        # --- Test 1 unify_and_merge_rasters ---
        out_path = unify_and_merge_rasters(
            [str(raster1), str(raster2)], 
            target_res=10.0, 
            output_path=str(merged_out)
        )
        
        assert out_path.exists()
        with rasterio.open(out_path) as src:
            assert src.count == 3, f"Expected 3 bands, got {src.count}"
            assert src.res == (10.0, 10.0), f"Expected resolution 10x10, got {src.res}"
            assert src.width == 10 and src.height == 10
            assert src.descriptions == ("band_1_res10.0", "band_1_res20.0", "band_2_res20.0")
        print("✅ Test 1 passed: unify_and_merge_rasters creates correct shape, bands, and descriptions.")
        
        # --- Test 2 rasterize_vectors ---
        mask_out = tmp_dir / "masks.tif"
        dict_out = tmp_dir / "mapping.json"
        
        configs = [
            {"target_column": "value", "is_binary": True},
            {"target_column": "class_name", "is_binary": False}
        ]
        
        rasterize_vectors(
            vector_paths=[str(vector_bin), str(vector_multi)],
            reference_raster_path=str(merged_out),
            vector_configs=configs,
            output_mask_path=str(mask_out),
            output_dict_path=str(dict_out)
        )
        
        assert mask_out.exists()
        assert dict_out.exists()
        
        with rasterio.open(mask_out) as src:
            assert src.count == 2, f"Expected 2 mask bands, got {src.count}"
            data = src.read()
            # Binary band check
            assert set(np.unique(data[0])).issubset({0, 1})
            # Multi class band check
            assert set(np.unique(data[1])).issubset({0, 1, 2})
            
        with open(dict_out, "r") as f:
            mapping = json.load(f)
            assert "v_binary.gpkg" in mapping
            assert "v_multi.gpkg" in mapping
            assert mapping["v_binary.gpkg"] == {"Background": 0, "Target": 1}
            assert mapping["v_multi.gpkg"] == {"Background": 0, "Forest": 1, "Water": 2}
            
        print("✅ Test 2 passed: rasterize_vectors creates multi-band mask and nested mapping correctly.")
        
        print("\n🎉 All transformer tests passed!")
        
    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    test_transformer()
