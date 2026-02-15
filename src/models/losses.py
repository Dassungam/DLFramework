"""
Modul: losses.py
Teil von: GeoAI_Framework
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # Sigmoid, da wir mit Logits arbeiten
        inputs = torch.sigmoid(logits)
        
        # Flatten (Batch, C, H, W) -> (Batch, -1)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        
        return 1 - dice

class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5):
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, inputs, targets):
        loss_bce = self.bce(inputs, targets)
        loss_dice = self.dice(inputs, targets)
        return self.bce_weight * loss_bce + (1 - self.bce_weight) * loss_dice

def get_loss_function(config):
    mode = config['data'].get('mask_type', 'binary')
    loss_name = config['training'].get('loss_function', 'bce') # Neu: Aus Config lesen
    
    print(f"Lade Loss-Funktion: {loss_name} (Mode: {mode})")

    if mode == 'binary':
        if loss_name == 'dice':
            return DiceLoss()
        elif loss_name == 'bce_dice':
            return BCEDiceLoss(bce_weight=0.5) # 50% BCE, 50% Dice
        else:
            return nn.BCEWithLogitsLoss()
    
    elif mode == 'regression':
        return nn.MSELoss()
    
    elif mode == 'multiclass':
        return nn.CrossEntropyLoss()
    
    else:
        raise ValueError(f"Unbekannter Modus: {mode}")