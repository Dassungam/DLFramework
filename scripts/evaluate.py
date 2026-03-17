"""
Skript: evaluate.py
Zweck: Unabhängiger Vergleich von Vorhersage-Tiffs und Ground-Truth-Masken.
"""
import rasterio
import numpy as np
import argparse
import yaml
from tqdm import tqdm

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--config", default="config/default.yaml")
    args = parser.parse_args()
    # Lade die YAML-Datei in ein Dictionary
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)

    # Übergib das geladene Dictionary an die Funktion
    evaluate_metrics(args.pred, args.mask, config_dict)