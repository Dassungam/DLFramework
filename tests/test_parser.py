import os
import shutil
import numpy as np
import rasterio
from rasterio.transform import from_origin
import geopandas as gpd
from shapely.geometry import Point, Polygon
from pathlib import Path
import sys
from pathlib import Path

# Add the project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.parser import process_datasets, TARGET_CRS

def setup_dummy_data(temp_dir: Path):
    """Generates dummy files for testing."""
    raw_dir = temp_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Dummy .jp2 (EPSG:3857, 1 band)
    jp2_path = raw_dir / "test_raster.jp2"
    width, height = 10, 10
    transform = from_origin(1000000, 1000000, 10, 10) # Dummy EPSG:3857 coordinates
    data = np.zeros((1, height, width), dtype=np.uint16)
    
    with rasterio.open(
        jp2_path, 'w',
        driver='JP2OpenJPEG',
        height=height,
        width=width,
        count=1,
        dtype=np.uint16,
        crs='EPSG:3857',
        transform=transform
    ) as dst:
        dst.write(data)
        
    # 2. Dummy .shp (EPSG:32632, 1 column)
    shp_path = raw_dir / "test_vector_shp.shp"
    df_shp = gpd.GeoDataFrame({
        'val': [1.0, 2.0],
        'geometry': [Point(500000, 5000000), Point(500100, 5000100)]
    }, crs='EPSG:32632')
    df_shp.to_file(shp_path)
    
    # 3. Dummy .geojson (EPSG:4326, 2 columns)
    geojson_path = raw_dir / "test_vector_geojson.geojson"
    df_geojson = gpd.GeoDataFrame({
        'col1': ['a', 'b'],
        'col2': [10, 20],
        'geometry': [Polygon([(0,0), (1,0), (1,1), (0,1)]), Polygon([(2,2), (3,2), (3,3), (2,3)])]
    }, crs='EPSG:4326')
    df_geojson.to_file(geojson_path, driver='GeoJSON')
    
    return [jp2_path, shp_path, geojson_path]

def test_parser_standardization():
    temp_dir = Path("tests/temp_data")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    output_dir = temp_dir / "standardized"
    
    try:
        # Step 1: Generate dummy data
        input_files = setup_dummy_data(temp_dir)
        
        # Step 2: Run process_datasets
        processed_files = process_datasets([str(p) for p in input_files], str(output_dir))
        
        # Step 3: Assertions
        
        # 3.1 Three separate files are created
        assert len(processed_files) == 3
        expected_files = [
            output_dir / "test_raster_standardized.tif",
            output_dir / "test_vector_shp_standardized.gpkg",
            output_dir / "test_vector_geojson_standardized.gpkg"
        ]
        for f in expected_files:
            assert f.exists(), f"File {f} was not created."
            
        # 3.2 All output files share the exact same CRS (EPSG:4326)
        for f in expected_files:
            if f.suffix == '.tif':
                with rasterio.open(f) as src:
                    assert src.crs.to_string() == TARGET_CRS
            else:
                gdf = gpd.read_file(f)
                assert gdf.crs.to_string() == TARGET_CRS
                
        # 3.3 Rename rules
        # Raster: test_raster.jp2 (1 band) -> internal band named 'test_raster'
        with rasterio.open(output_dir / "test_raster_standardized.tif") as src:
            assert src.descriptions[0] == "test_raster"
            
        # Vector (SHP): test_vector_shp.shp (1 column) -> column named 'test_vector_shp'
        gdf_shp = gpd.read_file(output_dir / "test_vector_shp_standardized.gpkg")
        assert "test_vector_shp" in gdf_shp.columns
        assert "val" not in gdf_shp.columns
        
        # Vector (GeoJSON): test_vector_geojson.geojson (2 columns) -> preserve original names
        gdf_geojson = gpd.read_file(output_dir / "test_vector_geojson_standardized.gpkg")
        assert "col1" in gdf_geojson.columns
        assert "col2" in gdf_geojson.columns
        assert "test_vector_geojson" not in gdf_geojson.columns
        
        print("\n✅ All tests passed!")
        
    finally:
        # Cleanup
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

if __name__ == "__main__":
    test_parser_standardization()
