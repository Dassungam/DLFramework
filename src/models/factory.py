"""
Modul: factory.py
Teil von: GeoAI_Framework
"""
import segmentation_models_pytorch as smp
import rasterio
import numpy as np
from pathlib import Path
from src.utils.config_utils import get_task_mode

def get_model(config):
    """
    Erstellt ein PyTorch Modell basierend auf der Konfiguration.
    """
    model_cfg = config['model']
    
    # 1. Parameter auslesen
    arch_name = model_cfg['architecture'] # z.B. "Unet"
    encoder_name = model_cfg['encoder']   # z.B. "resnet34"
    weights = model_cfg['weights']        # z.B. "imagenet" oder None
    activation = model_cfg.get('activation')
    if activation == '': activation = None

    # 2. Dynamisch in_channels und classes aus der Config oder den TIFs lesen
    in_channels = model_cfg.get('in_channels')
    
    # Falls in_channels nicht in der Config steht, versuchen wir es aus den Daten zu lesen
    if in_channels is None:
        datasets = config.get('data', {}).get('datasets', [])
        if datasets:
            features_path = Path(datasets[0]['features'])
        else:
            features_path = Path("data/processed_features.tif")
            if not features_path.exists():
                train_dir = config.get('data', {}).get('train_dir', 'data')
                features_path = Path(train_dir) / "image.tif"
        
        if features_path.exists():
            with rasterio.open(features_path) as src:
                in_channels = src.count
        else:
            # Letzter Fallback
            in_channels = 3 # Standard RGB
            print(f"Warning: Could not determine in_channels, using default: {in_channels}")

    # classes basierend auf mask_type ermitteln
    mask_type = get_task_mode(config)
    if mask_type in ['binary', 'regression']:
        classes = 1
    else:
        # 1. Bevorzugt: Class Map aus der Config (von app.py erstellt)
        class_map = config.get('data', {}).get('class_map')
        if class_map:
            classes = len(class_map)
        else:
            # Fallback: Wie zuvor aus der Maske lesen
            datasets = config.get('data', {}).get('datasets', [])
            if datasets:
                target_path = Path(datasets[0]['target'])
            else:
                target_path = Path("data/processed_target.tif")
                
            if target_path.exists():
                with rasterio.open(target_path) as src:
                    mask_data = src.read(1)
                    classes = int(np.nanmax(mask_data)) + 1
            else:
                classes = config.get('model', {}).get('classes', 1)

    print(f"Erstelle Modell: {arch_name} mit Encoder {encoder_name}")
    print(f"--> Auto-erkannt: {in_channels} Input Channels, {classes} Output Classes (Mode: {mask_type})")

    # 3. Dynamische Instanziierung
    if not hasattr(smp, arch_name):
        raise ValueError(f"Architektur '{arch_name}' nicht in SMP gefunden!")
    
    ModelClass = getattr(smp, arch_name)

    # 4. Modell bauen
    model = ModelClass(
        encoder_name=encoder_name,
        encoder_weights=weights,
        in_channels=in_channels,
        classes=classes,
        activation=activation
    )
    
    return model