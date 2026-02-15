"""
Skript: predict.py
Teil von: GeoAI_Framework
Zweck: Inferenz auf großen GeoTIFFs (Tiling) & Evaluation gegen Ground Truth.
"""
import os
import sys
import yaml
import torch
import rasterio
import numpy as np
import argparse
from tqdm import tqdm
from rasterio.windows import Window

# Pfad-Setup für Framework-Importe
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.factory import get_model

# --- KONFIGURATION ---
TILE_SIZE = 1024  # Größe der Kacheln für die Grafikkarte (1024x1024 passt gut auf RTX 5070)
OVERLAP = 0       # (Optional) Überlappung, hier erstmal 0 für Speed

def predict_large_image(model, img_path, out_path, device, config):
    """
    Führt Sliding-Window Inferenz durch. 
    Unterstützt jetzt Binary (Segmentation) und Regression (NDVI).
    """
    print(f"--- Starte Vorhersage für: {img_path} ---")
    
    # Modus aus Config lesen
    mode = config['data'].get('mask_type', 'binary')
    norm_max = config['data'].get('normalization_max', 3000.0)
    
    with rasterio.open(img_path) as src:
        meta = src.meta.copy()
        
        # Output-Metadaten anpassen je nach Modus
        if mode == 'regression':
            dtype = 'float32'
            nodata = -9999.0 # Ein sicherer Wert für float
        else:
            dtype = 'uint8'
            nodata = 0
            
        meta.update({
            'driver': 'GTiff',
            'count': 1,
            'dtype': dtype, 
            'compress': 'lzw',
            'nodata': nodata
        })
        
        width = src.width
        height = src.height
        
        with rasterio.open(out_path, 'w', **meta) as dst:
            
            for y in tqdm(range(0, height, TILE_SIZE), desc="Processing Tiles"):
                for x in range(0, width, TILE_SIZE):
                    
                    # 1. Fenster definieren
                    w_width = min(TILE_SIZE, width - x)
                    w_height = min(TILE_SIZE, height - y)
                    window = Window(x, y, w_width, w_height)
                    
                    # 2. Daten lesen
                    img_data = src.read([1, 2, 3, 4], window=window)
                    
                    # 3. Padding
                    pad_img = np.zeros((4, TILE_SIZE, TILE_SIZE), dtype=img_data.dtype)
                    pad_img[:, :w_height, :w_width] = img_data
                    
                    # --- WICHTIG: Normalisierung wie im Training ---
                    # Erst clippen (damit Wolken nicht explodieren)
                    pad_img = np.clip(pad_img, 0, norm_max)
                    # Dann skalieren
                    tensor = torch.from_numpy(pad_img).float() / norm_max
                    tensor = tensor.unsqueeze(0).to(device)
                    
                    # 4. Inferenz
                    with torch.no_grad():
                        logits = model(tensor)
                        
                        if mode == 'regression':
                            # Regression: Rohwerte nehmen (kein Sigmoid!)
                            # Output ist (1, 1, H, W) -> (H, W)
                            pred = logits.cpu().numpy()[0, 0]
                        else:
                            # Segmentation: Sigmoid + Threshold
                            probs = torch.sigmoid(logits)
                            pred = (probs > 0.5).byte().cpu().numpy()[0, 0]
                    
                    # 5. Un-Padding
                    pred_crop = pred[:w_height, :w_width]
                    
                    # 6. Speichern
                    dst.write(pred_crop, 1, window=window)
                    
    print(f"Vorhersage gespeichert unter: {out_path}")

def evaluate_metrics(pred_path, mask_path, config):
    """
    Berechnet IoU (für Binary) oder MSE (für Regression).
    """
    print(f"--- Starte Evaluation ---")
    
    mode = config['data'].get('mask_type', 'binary')
    
    # Variablen für Metriken
    intersection = 0
    union = 0
    total_squared_error = 0.0
    total_pixels = 0
    
    with rasterio.open(pred_path) as pred_src, rasterio.open(mask_path) as mask_src:
        
        # Check Dimensionen
        if pred_src.width != mask_src.width or pred_src.height != mask_src.height:
            print("WARNUNG: Dimensionen stimmen nicht überein!")
            return

        for ji, window in tqdm(pred_src.block_windows(1), desc="Calculating Metrics"):
            
            p = pred_src.read(1, window=window)
            m = mask_src.read(1, window=window)
            
            # --- REGRESSION EVALUATION ---
            if mode == 'regression':
                # Sicherstellen, dass wir gültige Werte vergleichen
                # (Optional: NoData maskieren, hier vereinfacht)
                valid_mask = ~np.isnan(m) & ~np.isnan(p)
                
                if np.any(valid_mask):
                    diff = p[valid_mask] - m[valid_mask]
                    total_squared_error += np.sum(diff ** 2)
                    total_pixels += np.sum(valid_mask)

            # --- SEGMENTATION EVALUATION ---
            else:
                m = (m > 0).astype(np.uint8)
                p = (p > 0).astype(np.uint8) # Sollte schon binär sein, aber sicher ist sicher
                
                intersection += np.logical_and(p, m).sum()
                union += np.logical_or(p, m).sum()

    print("\n" + "="*30)
    print(f"ERGEBNIS (Test Set) - Modus: {mode}")
    
    if mode == 'regression':
        if total_pixels > 0:
            mse = total_squared_error / total_pixels
            print(f"Mean Squared Error (MSE): {mse:.6f}")
            print(f"Root MSE (RMSE):        {np.sqrt(mse):.6f}")
        else:
            print("Keine gültigen Pixel für Evaluation gefunden.")
            
    else:
        iou = intersection / union if union > 0 else 1.0
        print(f"Intersection over Union (IoU): {iou:.4f}")
        
    print("="*30 + "\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml", help="Pfad zur Config")
    parser.add_argument("--input_image", required=True, help="Pfad zum Test-Bild (image.tif)")
    parser.add_argument("--input_mask", required=True, help="Pfad zur Test-Maske (mask.tif)")
    parser.add_argument("--output", default="prediction.tif", help="Pfad für output Tiff")
    parser.add_argument("--model_path", default="models/best_model.pth", help="Pfad zum trainierten Modell")
    
    args = parser.parse_args()
    
    # 1. Config & Device
    with open(args.config) as f:
        config = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. Modell laden
    print(f"Lade Modell von {args.model_path}...")
    model = get_model(config)
    
    # Gewichte laden (wichtig: map_location für GPU/CPU Sicherheit)
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    
    # Aufruf Predict mit Config
    predict_large_image(model, args.input_image, args.output, device, config)
    
    # Aufruf Evaluate mit Config
    if os.path.exists(args.input_mask):
        evaluate_metrics(args.output, args.input_mask, config)
    else:
        print("Keine Maske für Evaluation gefunden. Nur Vorhersage gespeichert.")

if __name__ == "__main__":
    main()