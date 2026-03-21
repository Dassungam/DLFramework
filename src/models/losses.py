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
        # Create a mask of valid pixels (where target is not 255)
        valid_mask = (targets != 255)
        
        # Filter inputs and targets
        inputs = torch.sigmoid(logits)[valid_mask]
        targets = targets[valid_mask]
        
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