"""
Skript: train.py
Teil von: GeoAI_Framework
"""

import os
import sys

# Pfad-Hack für Framework-Importe
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import yaml
import argparse
import torch
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import wandb

from src.data.datamodule import GeoDataModule
from src.data.dataset import GeoSpatialDataset
from src.models.factory import get_model
from src.models.loss_factory import get_loss_function
from src.training.trainer import Trainer

def main(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config nicht gefunden: {config_path}")
        
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    print(f"--- Starte Projekt: {config['project_name']} ---", flush=True)
    
    # 1. Start lean wandb run
    wandb_cfg = config.get("training", {})
    wandb_mode = wandb_cfg.get("wandb_mode", "online")
    
    # Check if we are logged in, otherwise default to offline to avoid blocking
    if wandb_mode == "online":
        has_key = os.environ.get("WANDB_API_KEY") is not None
        has_netrc = os.path.exists(os.path.expanduser("~/.netrc"))
        if not (has_key or has_netrc):
            print("WandB authentication not found (no API key or .netrc). Switching to OFFLINE mode.", flush=True)
            wandb_mode = "offline"

    wandb.init(
        project=config.get("project_name", "GeoAI_Framework"),
        entity=config.get("wandb_entity"),
        name=config.get("experiment_name", "run_1"),
        mode=wandb_mode,
        config={
            "learning_rate": config["training"]["learning_rate"],
            "batch_size": config["training"]["batch_size"],
            "epochs": config["training"]["epochs"],
            "optimizer": config["training"].get("optimizer", "adamw"),
            "architecture": config["model"].get("architecture", "Unet"),
            "encoder": config["model"].get("encoder", "mit_b5"),
            "loss_function": config["training"].get("loss_function", "bce_dice")
        }
    )
    print(f"WandB initialized in {wandb_mode} mode.", flush=True)
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
    
    # Optional: Watch model topology only
    wandb.watch(model, log=None)

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
    
    print("\n--- Training finished successfully. Closing WandB run... ---")
    wandb.finish()
    print("--- WandB closed. Process exiting. ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/default.yaml")
    args = parser.parse_args()
    main(args.config)