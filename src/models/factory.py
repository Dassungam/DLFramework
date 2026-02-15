"""
Modul: factory.py
Teil von: GeoAI_Framework
"""
import segmentation_models_pytorch as smp

def get_model(config):
    """
    Erstellt ein PyTorch Modell basierend auf der Konfiguration.
    """
    model_cfg = config['model']
    
    # 1. Parameter auslesen
    arch_name = model_cfg['architecture'] # z.B. "Unet"
    encoder_name = model_cfg['encoder']   # z.B. "resnet34"
    weights = model_cfg['weights']        # z.B. "imagenet" oder None
    in_channels = model_cfg['in_channels']
    classes = model_cfg['classes']
    activation = model_cfg['activation']  # Meistens None (für Logits)

    print(f"Erstelle Modell: {arch_name} mit Encoder {encoder_name}")

    # 2. Dynamische Instanziierung
    # Wir nutzen getattr, um die Klasse aus smp abzurufen (z.B. smp.Unet)
    if not hasattr(smp, arch_name):
        raise ValueError(f"Architektur '{arch_name}' nicht in SMP gefunden!")
    
    ModelClass = getattr(smp, arch_name)

    # 3. Modell bauen
    model = ModelClass(
        encoder_name=encoder_name,
        encoder_weights=weights,
        in_channels=in_channels,
        classes=classes,
        activation=activation
    )
    
    return model