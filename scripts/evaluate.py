"""
    Skript: evaluate.py
Zweck: Unabhängiger Vergleich von Vorhersage-Tiffs und Ground-Truth-Masken.
"""
import os
import sys

# Pfad-Hack
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import rasterio
import numpy as np
import argparse
import yaml
from tqdm import tqdm
from src.utils.config_utils import get_task_mode

def calculate_metrics_from_arrays(p, m, mode):
    """
    Computes metrics from numpy arrays directly.
    """
    metrics = {}
    
    if mode == 'regression':
        valid_mask = ~np.isnan(m) & ~np.isnan(p)
        if np.any(valid_mask):
            diff = p[valid_mask] - m[valid_mask]
            mse = np.mean(diff ** 2)
            metrics['MSE'] = float(mse)
            metrics['RMSE'] = float(np.sqrt(mse))
            # R2 Score
            ss_res = np.sum((m[valid_mask] - p[valid_mask])**2)
            ss_tot = np.sum((m[valid_mask] - np.mean(m[valid_mask]))**2)
            metrics['R2'] = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
        else:
            metrics['MSE'] = None
            metrics['RMSE'] = None
            metrics['R2'] = None
            
    elif mode in ['multiclass', 'classification']:
        unique_classes = np.unique(m)
        num_classes = len(unique_classes)
        
        per_class_metrics = {}
        total_iou = 0
        total_acc_pixels = (p == m).sum()
        total_pixels = m.size
        
        for cls in unique_classes:
            m_cls = (m == cls)
            p_cls = (p == cls)
            
            tp = np.logical_and(p_cls, m_cls).sum()
            fp = np.logical_and(p_cls, ~m_cls).sum()
            fn = np.logical_and(~p_cls, m_cls).sum()
            
            iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 1.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            per_class_metrics[int(cls)] = {
                'IoU': float(iou),
                'Precision': float(precision),
                'Recall': float(recall),
                'F1-Score': float(f1),
                'TP': int(tp),
                'FP': int(fp),
                'FN': int(fn)
            }
            total_iou += iou
            
        metrics = {
            'Overall Accuracy': float(total_acc_pixels / total_pixels),
            'mIoU': float(total_iou / num_classes),
            'Per-Class': per_class_metrics
        }
    
    elif mode == 'binary': 
        m_bin = (m > 0).astype(np.uint8)
        p_bin = (p > 0).astype(np.uint8)
        
        tp = np.logical_and(p_bin, m_bin).sum()
        fp = np.logical_and(p_bin, ~m_bin).sum()
        fn = np.logical_and(~p_bin, m_bin).sum()
        
        intersection = tp
        union = np.logical_or(p_bin, m_bin).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        iou = intersection / union if union > 0 else 1.0
        
        metrics = {
            'IoU': float(iou),
            'Precision': float(precision),
            'Recall': float(recall),
            'F1-Score': float(f1_score),
            'TP': int(tp),
            'FP': int(fp),
            'FN': int(fn)
        }
        
    else: # Fallback / Unknown
        print(f"Unknown mode: {mode}. No metrics calculated.")
        metrics = {}
        
    return metrics

def evaluate_metrics(pred_path, mask_path, config):
    """
    Berechnet IoU (für Binary) oder MSE (für Regression).
    """
    print(f"--- Starte Evaluation ---")
    
    mode = get_task_mode(config)
    
    # Check if files fit in memory
    try:
        with rasterio.open(pred_path) as pred_src, rasterio.open(mask_path) as mask_src:
            if pred_src.width != mask_src.width or pred_src.height != mask_src.height:
                print("WARNUNG: Dimensionen stimmen nicht überein!")
                return
                
            p = pred_src.read(1)
            m = mask_src.read(1)
            
            metrics = calculate_metrics_from_arrays(p, m, mode)
            
    except Exception as e:
        print(f"Error reading files for evaluation: {e}")
        return

    print("\n" + "="*30)
    print(f"ERGEBNIS (Test Set) - Modus: {mode}")
    
    if mode == 'regression':
        if metrics.get('MSE') is not None:
            print(f"Mean Squared Error (MSE): {metrics['MSE']:.6f}")
            print(f"Root MSE (RMSE):        {metrics['RMSE']:.6f}")
        else:
            print("Keine gültigen Pixel für Evaluation gefunden.")
            
    else:
        print(f"Intersection over Union (IoU): {metrics['IoU']:.4f}")
        print(f"Precision: {metrics['Precision']:.4f}")
        print(f"Recall: {metrics['Recall']:.4f}")
        print(f"F1-Score: {metrics['F1-Score']:.4f}")
        
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