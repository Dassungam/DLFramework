import torch
import numpy as np
import yaml
import os
import sys

# Pfad-Hack für Framework-Importe
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.dataset import GeoSpatialDataset
from src.models.loss_factory import get_loss_function
from src.models.factory import get_model
from src.utils.config_utils import get_task_mode

def test_regression_config():
    print("Testing Regression Configuration...")
    
    # Simulate a regression config (NDVI)
    config = {
        'project_name': 'Test_Regression',
        'training': {
            'task_type': 'regression',
            'loss_function': 'mse',
            'learning_rate': 0.001
        },
        'data': {
            'img_size': 256,
            'datasets': [
                {
                    'name': 'TestDS',
                    'features': 'tests/mock_features.tif',
                    'target': 'tests/mock_target.tif'
                }
            ]
        },
        'model': {
            'architecture': 'Unet',
            'encoder': 'resnet18',
            'weights': None,
            'in_channels': 3
        },
        'system': {
            'device': 'cpu'
        }
    }
    
    # 1. Test get_task_mode
    mode = get_task_mode(config)
    print(f"Detected mode: {mode}")
    assert mode == 'regression', f"Expected regression, got {mode}"
    
    # 2. Test Loss Factory
    criterion = get_loss_function(config)
    print(f"Loss function: {type(criterion).__name__}")
    assert isinstance(criterion, torch.nn.MSELoss), "Expected MSELoss for regression"
    
    # 3. Test Model Factory
    model = get_model(config)
    print(f"Model out_channels: {model.segmentation_head[0].out_channels}")
    assert model.segmentation_head[0].out_channels == 1, "Expected 1 output channel for regression"

    print("Regression Configuration Test Passed!")

if __name__ == "__main__":
    # Create mock data if not exists (minimal check)
    os.makedirs("tests", exist_ok=True)
    # Note: We won't actually load the dataset here to avoid needing real TIF files,
    # but we tested the logic units.
    
    try:
        test_regression_config()
    except Exception as e:
        print(f"Test FAILED: {e}")
        sys.exit(1)
