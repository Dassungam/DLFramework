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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.factory import get_model
from src.data.preprocessing import robust_normalize, standardize
from src.utils.config_utils import get_task_mode

# --- KONFIGURATION wird nun dynamisch aus der Config geladen ---

def predict_large_image(model, img_path, out_path, device, config):
    """
    Führt Sliding-Window Inferenz durch. 
    Unterstützt jetzt Binary (Segmentation) und Regression (NDVI).
    """
    print(f"--- Starte Vorhersage für: {img_path} ---")
    
    # Modus und Tiling aus Config lesen
    mode = get_task_mode(config)
    norm_max = config['data'].get('normalization_max', 3000.0)
    
    pred_cfg = config.get('prediction', {})
    tile_size = pred_cfg.get('tile_size', 512)
    overlap = pred_cfg.get('overlap', 64)
    stride = tile_size - overlap
    
    with rasterio.open(img_path) as src:
        print(f"Image Resolution: {src.res[0]:.4f} x {src.res[1]:.4f} units/pixel")
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
            
            total_steps = len(range(0, height, stride))
            for i, y in enumerate(tqdm(range(0, height, stride), desc="Processing Tiles")):
                # Print progress for Streamlit to parse
                print(f"PROGRESS:{(i+1)/total_steps*100:.2f}%", flush=True)
                for x in range(0, width, stride):
                    
                    # 1. Fenster definieren (darf nicht über das Bild hinausgehen)
                    w_width = min(tile_size, width - x)
                    w_height = min(tile_size, height - y)
                    window = Window(x, y, w_width, w_height)
                    
                    # 2. Daten lesen - Nur die in der Config definierten Bands nehmen!
                    bands = config['data'].get('input_channels', list(range(1, src.count + 1)))
                    img_data = src.read(bands, window=window)
                    
                    # 3. Padding (falls Fenster kleiner als tile_size)
                    pad_img = np.zeros((src.count, tile_size, tile_size), dtype=img_data.dtype)
                    pad_img[:, :w_height, :w_width] = img_data
                    
                    # --- WICHTIG: Normalisierung wie im Training ---
                    if "mean" in config["data"] and "std" in config["data"]:
                        pad_img = standardize(pad_img, config["data"]["mean"], config["data"]["std"])
                    else:
                        # Fallback falls keine Stats in Config (sollte nicht passieren bei neuem Workflow)
                        pad_img, _, _ = robust_normalize(pad_img)
                        
                    tensor = torch.from_numpy(pad_img).float()
                    tensor = tensor.unsqueeze(0).to(device)
                    
                    # 4. Inferenz
                    with torch.no_grad():
                        logits = model(tensor)
                        
                        if mode == 'regression':
                            # 5. Vorhersage (Logits -> Wahrscheinlichkeit)
                            pred = logits.cpu().numpy()[0, 0]
                            
                            # NEU: Automatische Denormalisierung falls Ziel standardisiert war
                            t_mean = config['data'].get('target_mean')
                            t_std = config['data'].get('target_std')
                            if t_mean is not None and t_std is not None:
                                pred = (pred * t_std) + t_mean
                                
                        elif mode in ['multiclass', 'classification']:
                            # Multiclass: Argmax über die Channel-Dimension (1)
                            # Logits Shape: (1, C, H, W) -> Indices: (H, W)
                            pred_idx = torch.argmax(logits, dim=1).cpu().numpy()[0]
                            
                            # --- NEU: Inverse Mapping ---
                            # Wir müssen die Indizes (0,1,2...) zurück in Originalwerte (10,20...) verwandeln
                            class_map = config['data'].get('class_map')
                            if class_map:
                                # Erstelle Inverse Map: {index: original_value}
                                inv_map = {int(v): int(k) for k, v in class_map.items()}
                                
                                # Anwenden der Inverse Map auf das gesamte Tile
                                # Wir nutzen eine Vectorized Approach für Performance
                                h, w = pred_idx.shape
                                pred = np.zeros((h, w), dtype=np.uint16 if max(inv_map.values()) > 255 else np.uint8)
                                for idx, orig_val in inv_map.items():
                                    pred[pred_idx == idx] = orig_val
                            else:
                                pred = pred_idx.astype(np.uint8)
                        else:
                            # Segmentation: Sigmoid + Threshold (Binary Default)
                            probs = torch.sigmoid(logits)
                            pred = (probs > 0.5).byte().cpu().numpy()[0, 0]
                    
                    # 5. Un-Padding
                    pred_crop = pred[:w_height, :w_width]
                    
                    # 6. Speichern
                    dst.write(pred_crop, 1, window=window)
                    
    print(f"Vorhersage gespeichert unter: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml", help="Pfad zur Config")
    parser.add_argument("--input_image", required=True, help="Pfad zum Test-Bild (image.tif)")
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
    

if __name__ == "__main__":
    main()