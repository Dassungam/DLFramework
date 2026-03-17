"""
Utility functions for managing model checkpoints.
"""
import os

def list_checkpoints(models_dir="models"):
    """
    Scans the models directory and returns a list of available .pth or .pt files.
    """
    if not os.path.exists(models_dir):
        return []
        
    checkpoints = [f for f in os.listdir(models_dir) if f.endswith(('.pth', '.pt'))]
    return sorted(checkpoints)
