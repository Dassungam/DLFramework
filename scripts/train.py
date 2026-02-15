"""
Skript: train.py
Teil von: GeoAI_Framework
"""

import os
import sys

import yaml
import argparse
import torch
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from src.data.datamodule import GeoDataModule # <--- Neu

# Pfad-Hack
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.dataset import GeoSpatialDataset
from src.models.factory import get_model
from src.models.losses import get_loss_function
from src.training.trainer import Trainer

def main(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config nicht gefunden: {config_path}")
        
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    print(f"--- Starte Projekt: {config['project_name']} ---")
    
    # 1. Splits berechnen
    # (Wir müssen hier kurz tricksen, da wir Transforms VOR dem Split zuweisen müssen,
    # aber Train und Val unterschiedliche Transforms brauchen.
    # Sauberste Lösung: Wir splitten erst die Indizes oder nutzen Subset.)
    
    # Vereinfachter Weg: Wir laden das Dataset zweimal (kostet kaum Speicher, da nur Dateilisten)
    
    # 1. Daten vorbereiten (Einzeiler!)
    dm = GeoDataModule(config)
    dm.setup()
    
    # 2. Modell & Training Setup
    model = get_model(config)
    criterion = get_loss_function(config)
    optimizer = optim.AdamW(model.parameters(), lr=config['training']['learning_rate'])

    # 3. Start
    trainer = Trainer(
        model=model,
        train_loader=dm.train_dataloader(), # <--- Sauberer Zugriff
        val_loader=dm.val_dataloader(),     # <--- Sauberer Zugriff
        criterion=criterion,
        optimizer=optimizer,
        config=config
    )
    
    trainer.fit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/default.yaml")
    args = parser.parse_args()
    main(args.config)