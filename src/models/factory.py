"""
Modul: factory.py
Teil von: GeoAI_Framework
"""
import segmentation_models_pytorch as smp
import rasterio
import numpy as np
from pathlib import Path

def get_model(config):
    """
    Erstellt ein PyTorch Modell basierend auf der Konfiguration.
    """
    model_cfg = config['model']
    
    # 1. Parameter auslesen
    arch_name = model_cfg['architecture'] # z.B. "Unet"
    encoder_name = model_cfg['encoder']   # z.B. "resnet34"
    weights = model_cfg['weights']        # z.B. "imagenet" oder None
    activation = model_cfg['activation']  # Meistens None (für Logits)

    # 2. Dynamisch in_channels und classes aus den TIFs lesen
    features_path = Path("data/processed_features.tif")
    target_path = Path("data/processed_target.tif")
    
    # Fallback falls die processed Dateien nicht da sind
    if not features_path.exists():
        train_dir = config.get('data', {}).get('train_dir', 'data')
        features_path = Path(train_dir) / "image.tif"
        target_path = Path(train_dir) / "mask.tif"
        
    if not features_path.exists() or not target_path.exists():
        raise FileNotFoundError(f"Konnten Trainingsdaten nicht finden in {features_path} oder {target_path}")

    # in_channels aus Features lesen (Anzahl der Bänder)
    with rasterio.open(features_path) as src:
        in_channels = src.count

    # classes basierend auf mask_type und Target ermitteln
    mask_type = config.get('data', {}).get('mask_type', 'binary')
    if mask_type in ['binary', 'regression']:
        classes = 1
    else:
        # Bei Multiclass müssen wir die maximale Klasse + 1 im Target Raster finden
        with rasterio.open(target_path) as src:
            # Wir lesen die Daten um das exakte Maximum zu bestimmen
            mask_data = src.read(1)
            # Maximum extrahieren (ignoriert NaN Werte in der Maske, falls vorhanden)
            classes = int(np.nanmax(mask_data)) + 1

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