"""
Utility functions for managing model checkpoints.
"""
import os

def list_checkpoints(models_dir="models", extensions=('.pth', '.pt')):
    """
    Scans the models directory and returns a list of available files with given extensions.
    """
    if not os.path.exists(models_dir):
        return []
        
    checkpoints = [f for f in os.listdir(models_dir) if f.lower().endswith(extensions)]
    return sorted(checkpoints)
