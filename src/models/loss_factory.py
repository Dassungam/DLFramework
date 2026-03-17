"""
Modul: loss_factory.py
Teil von: GeoAI_Framework

Entscheidet, welche Loss-Funktion genutzt wird, abhängig vom Vorhersagetyp
(binary, classification, regression).
"""

import torch.nn as nn
from .losses import DiceLoss, BCEDiceLoss

def get_loss_function(config):
    """
    Gibt die Fehlerfunktion zurück, abhängig davon ob klassifiziert
    (Binär, Multiklassen) oder regrediert wird.
    """
    mode = config['data'].get('mask_type', 'binary')
    loss_name = config['training'].get('loss_function', 'bce')
    
    print(f"Lade Loss-Funktion: {loss_name} (Prediction-Typ: {mode})")

    # 1. Binäre Klassifikation
    if mode == 'binary':
        if loss_name == 'dice':
            return DiceLoss()
        elif loss_name == 'bce_dice':
            return BCEDiceLoss(bce_weight=0.5) # 50% BCE, 50% Dice
        else:
            return nn.BCEWithLogitsLoss()
    
    # 2. Regression (Kontinuierliche Werte)
    elif mode == 'regression':
        return nn.MSELoss()
    
    # 3. Multiclass Klassifikation (Mehrere Klassen)
    elif mode in ['multiclass', 'classification']:
        return nn.CrossEntropyLoss()
    
    else:
        raise ValueError(f"Unbekannter Modus (mask_type): {mode}")
