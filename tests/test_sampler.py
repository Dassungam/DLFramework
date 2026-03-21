import os
import pytest
import rasterio
import numpy as np
import pandas as pd
from pathlib import Path
from src.data.sampler import create_training_table

@pytest.fixture
def dummy_data_dir(tmp_path):
    """Fixture to create dummy GeoTIFFs for testing."""
    d = tmp_path / "data"
    d.mkdir()
    
    # Feature 1: B04 (Visible Red)
    f1_path = d / "B04.tif"
    with rasterio.open(
        f1_path, 'w', driver='GTiff', height=100, width=100, count=1,
        dtype='float32', crs='EPSG:4326', transform=rasterio.transform.from_origin(0, 0, 1, 1),
        nodata=-9999
    ) as dst:
        data = np.random.rand(1, 100, 100).astype('float32') * 100
        # Add some nodata
        data[0, 10, 10] = -9999
        dst.write(data)
        
    # Feature 2: NDVI
    f2_path = d / "NDVI.tif"
    with rasterio.open(
        f2_path, 'w', driver='GTiff', height=100, width=100, count=1,
        dtype='float32', crs='EPSG:4326', transform=rasterio.transform.from_origin(0, 0, 1, 1),
        nodata=-9999
    ) as dst:
        data = np.random.rand(1, 100, 100).astype('float32')
        # Add some NaN
        data[0, 20, 20] = np.nan
        dst.write(data)
        
    # Target Classification (0 and 1)
    t_class_path = d / "target_class.tif"
    with rasterio.open(
        t_class_path, 'w', driver='GTiff', height=100, width=100, count=1,
        dtype='int16', crs='EPSG:4326', transform=rasterio.transform.from_origin(0, 0, 1, 1),
        nodata=-1
    ) as dst:
        # 5000 zeros, 5000 ones
        data = np.zeros((1, 100, 100), dtype='int16')
        data[0, 50:, :] = 1
        dst.write(data)
        
    # Target Regression (floats)
    t_reg_path = d / "target_reg.tif"
    with rasterio.open(
        t_reg_path, 'w', driver='GTiff', height=100, width=100, count=1,
        dtype='float32', crs='EPSG:4326', transform=rasterio.transform.from_origin(0, 0, 1, 1),
        nodata=-9999
    ) as dst:
        data = np.random.rand(1, 100, 100).astype('float32') * 50
        dst.write(data)
        
    return {
        "features": [str(f1_path), str(f2_path)],
        "target_class": str(t_class_path),
        "target_reg": str(t_reg_path)
    }

def test_classification_sampling(dummy_data_dir):
    max_samples = 500
    df = create_training_table(
        dummy_data_dir["features"],
        dummy_data_dir["target_class"],
        "classification",
        max_samples
    )
    
    # Check shape: 2 features + 1 target = 3 columns
    assert df.shape[1] == 3
    assert "B04" in df.columns
    assert "NDVI" in df.columns
    assert "target" in df.columns
    
    # Check counts: up to 500 for class 0, up to 500 for class 1
    # Note: some might be dropped due to nodata in features at (10,10) and (20,20)
    class_counts = df['target'].value_counts()
    assert class_counts[0] <= max_samples
    assert class_counts[1] <= max_samples
    # With 10000 pixels total, sampling 1000 should be easy. 
    # At most 2 pixels are invalid, so we should have ~1000 rows.
    assert len(df) > 990
    
    # Ensure 0 is NOT filtered out
    assert 0 in df['target'].values

def test_regression_sampling(dummy_data_dir):
    max_samples = 1200
    df = create_training_table(
        dummy_data_dir["features"],
        dummy_data_dir["target_reg"],
        "regression",
        max_samples
    )
    
    # Check shape
    assert df.shape[1] == 3
    # Check total samples: should be exactly max_samples (minus any dropped for nodata)
    # Total pixels = 10000. 1200 sampled. 2 invalid pixels. 
    # Probability of hitting invalid pixels is low, but possible.
    assert len(df) <= max_samples
    assert len(df) > 1190

def test_nodata_filtering(dummy_data_dir):
    # We added nodata at (10,10) in B04 and NaN at (20,20) in NDVI.
    # If we sample EVERYTHING, we should see these dropped.
    df = create_training_table(
        dummy_data_dir["features"],
        dummy_data_dir["target_class"],
        "classification",
        10000 # Sample all
    )
    
    # Total pixels = 10000. 
    # Dropped: (10,10) and (20,20).
    # Result should be 9998.
    assert len(df) == 9998
    
    # Assert no NaNs or -9999 remain
    assert not df.isnull().values.any()
    assert not (df == -9999).values.any()
    assert not (df['target'] == -1).values.any()

def test_shape_and_names(dummy_data_dir):
    df = create_training_table(
        dummy_data_dir["features"],
        dummy_data_dir["target_reg"],
        "regression",
        10
    )
    assert list(df.columns) == ["B04", "NDVI", "target"]
