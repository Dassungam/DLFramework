"""
Modul: trainer.py
Teil von: GeoAI_Framework
"""
import os
import torch
from tqdm import tqdm
import numpy as np

class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.cfg = config
        
        # Hardware Setup
        self.device = torch.device(config['system']['device'])
        self.model.to(self.device)
        
        # Mixed Precision Setup (WICHTIG für RTX 5070 / Blackwell)
        self.use_amp = config['training'].get('use_amp', False)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        
        # Logging
        self.best_val_loss = float('inf')
        self.project_dir = os.getcwd() # Oder aus Config
        self.save_path = os.path.join(self.project_dir, "models", "best_model.pth")

    def fit(self):
        """Startet den kompletten Trainings-Prozess."""
        epochs = self.cfg['training']['epochs']
        print(f"--- Starte Training für {epochs} Epochen (AMP={self.use_amp}) ---")
        
        for epoch in range(1, epochs + 1):
            # 1. Trainieren
            train_loss = self._train_one_epoch(epoch)
            
            # 2. Validieren
            val_loss = self._validate(epoch)
            
            # 3. Bestes Modell speichern
            if val_loss < self.best_val_loss:
                print(f"   [Save] Neuer Bestwert! ({self.best_val_loss:.4f} -> {val_loss:.4f})")
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.save_path)
            
            print(f"Epoche {epoch}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    def _train_one_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f"Train Ep {epoch}", leave=False)
        
        for batch in pbar:
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            # Optimizer Reset
            self.optimizer.zero_grad()
            
            # --- MIXED PRECISION CONTEXT ---
            # Das hier ist der Turbo für deine RTX 5070
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
            
            # Backward Pass mit Scaler (verhindert Underflow bei float16)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Statistik
            running_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
        return running_loss / len(self.train_loader)

    def _validate(self, epoch):
        self.model.eval()
        running_loss = 0.0
        
        # Torch.no_grad spart Speicher, da wir hier nicht lernen müssen
        with torch.no_grad():
            loader = self.val_loader if self.val_loader else self.train_loader
            
            for batch in loader:
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                # Auch bei Val nutzen wir autocast, spart VRAM
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)
                
                running_loss += loss.item()
            
            # Wir brechen die innere Schleife hier ab, da 'loader' iteriert wurde
            # (Kleine Logik-Korrektur für sauberen Code)
            return running_loss / len(loader)