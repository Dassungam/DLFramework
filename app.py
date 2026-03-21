import streamlit as st
import yaml
import subprocess
from pathlib import Path
import rasterio
from rasterio.io import MemoryFile
import matplotlib.pyplot as plt
import numpy as np
import wandb
import pandas as pd

from scripts.evaluate import calculate_metrics_from_arrays
from src.utils.checkpoints import list_checkpoints
from src.utils.app_utils import calculate_global_stats, apply_percentile_stretch, plot_data_preview, plot_prediction_results, load_config, get_band_inventory, parse_band_selection
from src.utils.config_utils import get_task_mode
from src.data.preprocessing import standardize
from src.models.ml_trainer import train_ml_model
from src.models.ml_predictor import predict_ml

def render_evaluation_ui(config, pred_inventory, pred_metadata, framework_mode):
    """Refactored UI for showing prediction results and metrics."""
    st.markdown("---")
    st.subheader("Visualization")
    vis_bg_bands = st.multiselect("Select Ground Map Bands (Exactly 1 or 3)", options=pred_inventory, key=f"vis_bg_{config.get('framework_mode', 'default')}")
    vis_actual_mask = st.selectbox("Select Actual Target Mask (Optional)", options=["None"] + pred_inventory, key=f"vis_actual_{config.get('framework_mode', 'default')}")
    if vis_actual_mask != "None" and "[PROCESSED]" in vis_actual_mask:
        st.warning("⚠️ Warning: You've selected a [PROCESSED] file as the actual mask. This file contains standardized Z-scores (-3 to 3) instead of your raw values (0 to 1). Results may be misleading!")
    
    # Search for all available predictions in data/predictions/ folder
    pred_dir = Path("data/predictions")
    pred_dir.mkdir(exist_ok=True, parents=True)
    available_preds = sorted(list(pred_dir.glob("prediction_*.tif")), key=lambda p: p.stat().st_mtime, reverse=True)
    available_preds = [p.name for p in available_preds]
    
    if not available_preds:
        st.info("No prediction results found in `predictions/`. Run a prediction first.")
        return

    # Dropdown to choose prediction
    current_exp = config.get("model", {}).get("experiment_name") or config.get("experiment_name", "default")
    default_file = f"prediction_{current_exp}.tif"
    if default_file not in available_preds:
        default_idx = 0
    else:
        default_idx = available_preds.index(default_file)
        
    selected_pred_file = st.selectbox("Select Prediction Result to Evaluate", options=available_preds, index=default_idx, key=f"eval_select_{config.get('framework_mode', 'default')}")
    pred_out_path = pred_dir / selected_pred_file

    if st.button("👁️ Show Results", use_container_width=True, key=f"show_results_{config.get('framework_mode', 'default')}"):
        if len(vis_bg_bands) not in [1, 3]:
            st.error("Only exactly 1 or 3 bands are allowed for the ground map.")
        else:
            if not pred_out_path.exists():
                st.error(f"Prediction result ({pred_out_path}) not found.")
            else:
                try:
                    # Load and downsample
                    max_dim = 512
                    f_name, b_num = parse_band_selection(vis_bg_bands[0])
                    ref_meta = pred_metadata[f_name]
                    ref_f = ref_meta["file_obj"]
                    
                    with MemoryFile(ref_f.getvalue()) as memfile:
                        with memfile.open() as src:
                            scale = min(max_dim / src.width, max_dim / src.height)
                            out_shape = (int(src.height * scale), int(src.width * scale))
                            out_shape = (max(1, out_shape[0]), max(1, out_shape[1]))
                            
                    bg_arrays = []
                    for fb in vis_bg_bands:
                        f_name, b_num = parse_band_selection(fb)
                        f = pred_metadata[f_name]["file_obj"]
                        with MemoryFile(f.getvalue()) as memfile:
                            with memfile.open() as src:
                                bg_arrays.append(src.read(b_num, out_shape=out_shape).astype(np.float32))
                                
                    actual_mask_array = None
                    if vis_actual_mask != "None":
                        f_name, b_num = parse_band_selection(vis_actual_mask)
                        f = pred_metadata[f_name]["file_obj"]
                        with MemoryFile(f.getvalue()) as memfile:
                            with memfile.open() as src:
                                actual_mask_array = src.read(b_num, out_shape=out_shape).astype(np.float32)
                                
                    with rasterio.open(pred_out_path) as src:
                        pred_mask_array = src.read(1, out_shape=out_shape).astype(np.float32)
                        
                    # Determine task mode for visualization
                    task_mode = get_task_mode(config)
                    if task_mode == 'binary': task_mode = 'classification'

                    cnames = config.get('data', {}).get('class_names')
                    cmap = config.get('data', {}).get('class_map')
                    plot_prediction_results(bg_arrays, actual_mask_array, pred_mask_array, task_type=task_mode, class_names=cnames, class_map=cmap)
                    
                    # --- Evaluation Metrics ---
                    if actual_mask_array is not None:
                        st.markdown("### Quantitative Evaluation")
                        
                        metrics = calculate_metrics_from_arrays(pred_mask_array, actual_mask_array, task_mode)
                        
                        if task_mode == 'regression':
                            col1, col2, col3 = st.columns(3)
                            if metrics.get('MSE') is not None:
                                col1.metric("MSE", f"{metrics['MSE']:.4f}")
                                col2.metric("RMSE", f"{metrics['RMSE']:.4f}")
                                col3.metric("R² Score", f"{metrics.get('R2', 0):.4f}")
                            else:
                                st.warning("No valid pixels found for regression evaluation.")
                        elif 'mIoU' in metrics:
                            # Multiclass Mode
                            col1, col2 = st.columns(2)
                            col1.metric("Mean IoU (mIoU)", f"{metrics['mIoU'] * 100:.2f}%")
                            col2.metric("Overall Accuracy", f"{metrics['Overall Accuracy'] * 100:.2f}%")
                            
                            st.write("**Per-Class Metrics:**")
                            per_class = metrics['Per-Class']
                            cnames = config.get('data', {}).get('class_names', {})
                            
                            class_data = []
                            for cls_idx, c_metrics in per_class.items():
                                name = cnames.get(str(cls_idx), f"Class {cls_idx}")
                                class_data.append({
                                    "ID": cls_idx,
                                    "Name": name,
                                    "IoU": f"{c_metrics['IoU']*100:.2f}%",
                                    "F1": f"{c_metrics['F1-Score']*100:.2f}%",
                                    "Precision": f"{c_metrics['Precision']*100:.2f}%",
                                    "Recall": f"{c_metrics['Recall']*100:.2f}%",
                                    "Pixels": c_metrics['TP'] + c_metrics['FN'] # Total true pixels for this class
                                })
                            
                            st.table(pd.DataFrame(class_data))
                        else:
                            # Fallback / Binary Mode
                            col1, col2, col3, col4 = st.columns(4)
                            col1.metric("IoU", f"{metrics['IoU'] * 100:.2f}%")
                            col2.metric("Precision", f"{metrics['Precision'] * 100:.2f}%")
                            col3.metric("Recall", f"{metrics['Recall'] * 100:.2f}%")
                            col4.metric("F1-Score", f"{metrics['F1-Score'] * 100:.2f}%")
                            
                            st.caption(f"Raw Counts: True Positives ({metrics['TP']}) | False Positives ({metrics['FP']}) | False Negatives ({metrics['FN']})")
                            
                        # Wandb Logging
                        if wandb.run is not None:
                            wandb_metrics = {f"Final UI Evaluation/{k}": v for k, v in metrics.items() if v is not None}
                            wandb.log(wandb_metrics)
                            st.success("Metrics logged to Weights & Biases!")
                            
                    # --- Feature Importance (ML Only) ---
                    if "Traditional ML" in framework_mode:
                        st.markdown("### Feature Importance")
                        # Guess model path from prediction filename
                        import re
                        match = re.search(r"prediction_(.*)\.tif", pred_out_path.name)
                        exp_label = match.group(1) if match else "default"
                        model_joblib = Path(f"models/{exp_label}.joblib")
                        
                        if not model_joblib.exists():
                             # Try looking for any joblib in models/
                             joblibs = list(Path("models").glob("*.joblib"))
                             if joblibs: model_joblib = joblibs[0] # Fallback to first found

                        if model_joblib and model_joblib.exists():
                            try:
                                import joblib
                                model = joblib.load(model_joblib)
                                if hasattr(model, "feature_importances_"):
                                    importances = model.feature_importances_
                                    feature_names = getattr(model, "feature_names_in_", [f"Band {i+1}" for i in range(len(importances))])
                                    
                                    # If length mismatch (e.g. model from different run), truncate or pad
                                    if len(feature_names) != len(importances):
                                        feature_names = [f"Feature {i+1}" for i in range(len(importances))]
                                        
                                    df_imp = pd.DataFrame({"Feature": feature_names, "Importance": importances})
                                    df_imp = df_imp.sort_values("Importance", ascending=False)
                                    
                                    st.write(f"Top predictors for model: `{model_joblib.name}`")
                                    st.bar_chart(df_imp, x="Feature", y="Importance")
                                else:
                                    st.info("Selected model architecture does not provide feature importance scores.")
                            except Exception as e:
                                st.warning(f"Could not load feature importances from {model_joblib.name}: {e}")
                        else:
                            st.info("No matching .joblib model found in `models/` to show feature importance.")
                    
                except Exception as e:
                    st.error(f"Error loading visualization data: {e}")

def render_prediction_section(config, active_config_path, framework_mode):
    """Refactored UI for the complete prediction workflow (Inference)."""
    st.markdown("---")
    st.header("🔮 Prediction (Inference)")
    
    st.sidebar.markdown("---")
    st.sidebar.header("Prediction Data")
    pred_uploaded_files = st.sidebar.file_uploader(f"Upload GeoTIFFs for {framework_mode} Prediction", type=["tif", "tiff"], accept_multiple_files=True, key=f"pred_uploader_{framework_mode}")
    
    if pred_uploaded_files:
        pred_inventory, pred_metadata = get_band_inventory(pred_uploaded_files)
        st.subheader("Band Selection for Prediction")
        
        # Try to get training reference for band order
        datasets = config.get("data", {}).get("datasets", [])
        features_path = Path(datasets[0]["features"]) if datasets else None
        
        trained_bands_info = None
        if features_path and features_path.exists():
            try:
                with rasterio.open(features_path) as f_src:
                    if f_src.descriptions and any(f_src.descriptions):
                        trained_bands_info = f_src.descriptions
            except Exception: pass
                
        if trained_bands_info:
            st.info("**Training was performed with the following bands in exact order:**\n" + "\n".join([f"- {b}" for b in trained_bands_info]))
            
        pred_feature_bands = st.multiselect("Select Features for Prediction (MUST match training order exactly)", options=pred_inventory, key=f"pred_multis_{framework_mode}")
        
        # Model Checkpoint Selector
        st.markdown("---")
        st.subheader("Model Selection")
        ext = ('.pth', '.pt') if "Deep Learning" in framework_mode else ('.joblib')
        available_checkpoints = list_checkpoints("models", extensions=ext)
        
        if not available_checkpoints:
            st.warning(f"No model checkpoints found with {ext} in `models/`. Train a model first.")
            selected_checkpoint = None
        else:
            selected_checkpoint = st.selectbox("Select Model Checkpoint", options=available_checkpoints, key=f"model_select_{framework_mode}")
            st.info(f"Using checkpoint: `models/{selected_checkpoint}`")

        col_pred1, col_pred2 = st.columns(2)
        with col_pred1:
            if st.button("🚀 Run Prediction", use_container_width=True, key=f"run_pred_{framework_mode}"):
                required_channels = config.get("model", {}).get("in_channels")
                if not pred_feature_bands:
                    st.error("Please select at least one feature band for prediction.")
                elif required_channels and len(pred_feature_bands) != required_channels:
                    st.error(f"Error: Number of selected bands ({len(pred_feature_bands)}) does not match the model's expected input channels ({required_channels}).")
                else:
                    try:
                        # Save prediction input TIF
                        pred_selected_files = set([parse_band_selection(b)[0] for b in pred_feature_bands])
                        ref_file = list(pred_selected_files)[0]
                        ref_meta = pred_metadata[ref_file]
                        
                        pred_feature_arrays = []
                        for fb in pred_feature_bands:
                            f_name, b_num = parse_band_selection(fb)
                            f = pred_metadata[f_name]["file_obj"]
                            with MemoryFile(f.getvalue()) as memfile:
                                with memfile.open() as src:
                                    pred_feature_arrays.append(src.read(b_num))
                        
                        pred_stacked = np.stack(pred_feature_arrays)
                        
                        # Save prediction input TIF (Raw bands, normalization happens in predictor)
                        out_dir = Path("data")
                        out_dir.mkdir(exist_ok=True)
                        
                        # Output predictions in their own folder
                        pred_dir = Path("data/predictions")
                        pred_dir.mkdir(exist_ok=True, parents=True)
                        import re
                        safe_mode = re.sub(r'[^a-zA-Z0-9]', '_', framework_mode)
                        pred_input_path = out_dir / f"prediction_input_{safe_mode}.tif"
                        pred_profile = ref_meta["profile"].copy()
                        pred_profile.update(count=pred_stacked.shape[0], dtype=rasterio.float32)
                        
                        with rasterio.open(pred_input_path, 'w', **pred_profile) as dst:
                            dst.write(pred_stacked)
                            dst.descriptions = tuple(pred_feature_bands)
                            
                        exp_name = config.get("model", {}).get("experiment_name") or config.get("experiment_name", "default")
                        pred_out_path = pred_dir / f"prediction_{exp_name}.tif"
                        checkpoint_path = f"models/{selected_checkpoint}"
                        
                        progress_bar = st.progress(0, text=f"Initializing prediction...")
                        
                        if "Deep Learning" in framework_mode:
                            cmd = ["python", "scripts/predict.py", "--config", active_config_path, "--input_image", str(pred_input_path), "--output", str(pred_out_path), "--model_path", checkpoint_path]
                            
                            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
                            
                            import queue
                            import threading
                            import time

                            def enqueue_output(out, q):
                                for line in iter(out.readline, ''):
                                    q.put(line)
                                out.close()

                            q = queue.Queue()
                            t = threading.Thread(target=enqueue_output, args=(process.stdout, q))
                            t.daemon = True
                            t.start()
                            
                            while True:
                                # Check if process is still running
                                is_running = process.poll() is None
                                
                                # Read all currently available lines
                                while not q.empty():
                                    line = q.get_nowait()
                                    if "PROGRESS:" in line:
                                        try:
                                            percent_str = line.split("PROGRESS:")[1].strip().replace("%", "")
                                            percent = float(percent_str) / 100.0
                                            progress_bar.progress(min(1.0, percent), text=f"Deep Learning Inference: {percent*100:.1f}%")
                                        except: pass
                                
                                if not is_running:
                                    break
                                
                                time.sleep(0.1) # Yield to Streamlit and reduce CPU usage
                            
                            
                            return_code = process.wait()
                            if return_code == 0:
                                st.success(f"Prediction successfully saved to {pred_out_path}!")
                            else:
                                st.error(f"Prediction failed with exit code {return_code}. Check logs.")
                        else:
                            # ML Prediction
                            def ml_progress_update(fraction):
                                progress_bar.progress(fraction, text=f"ML Inference: {fraction*100:.1f}%")
                                
                            predict_ml(
                                model_path=checkpoint_path, 
                                feature_paths=[str(pred_input_path)], 
                                output_path=str(pred_out_path),
                                mean=config["data"].get("mean"),
                                std=config["data"].get("std"),
                                target_mean=config["data"].get("target_mean"),
                                target_std=config["data"].get("target_std"),
                                progress_callback=ml_progress_update
                            )
                            st.success(f"ML Prediction successfully saved to {pred_out_path}!")
                        
                        progress_bar.empty()
                                
                    except Exception as e:
                        st.error(f"Error preparing or running prediction: {e}")
                        
        with col_pred2:
            if st.button("⚖️ Check Band Order", use_container_width=True, key=f"check_bands_{framework_mode}"):
                import re
                safe_mode = re.sub(r'[^a-zA-Z0-9]', '_', framework_mode)
                pred_input_path = Path(f"data/prediction_input_{safe_mode}.tif")
                if not features_path or not features_path.exists():
                    st.error("Original training features from the first dataset not found.")
                elif not pred_input_path.exists():
                    st.error("Prediction input file not found. Run prediction first.")
                else:
                    try:
                        with rasterio.open(features_path) as src_train, rasterio.open(pred_input_path) as src_pred:
                            train_desc = src_train.descriptions or tuple(f"Band {i+1}" for i in range(src_train.count))
                            pred_desc = src_pred.descriptions or tuple(f"Band {i+1}" for i in range(src_pred.count))
                            st.write("### Band Order Comparison")
                            md_table = "| Index | Training Band | Prediction Band | Match |\n|---|---|---|---|\n"
                            max_len = max(len(train_desc), len(pred_desc))
                            all_match = len(train_desc) == len(pred_desc)
                            for i in range(max_len):
                                t_band = train_desc[i] if i < len(train_desc) else "N/A"
                                p_band = pred_desc[i] if i < len(pred_desc) else "N/A"
                                match_icon = "✅" if t_band == p_band else "❌"
                                if t_band != p_band: all_match = False
                                md_table += f"| {i+1} | {t_band} | {p_band} | {match_icon} |\n"
                            st.markdown(md_table)
                            if all_match: st.success("Success: Band orders match perfectly!")
                            else: st.error("Warning: Band orders do NOT match!")
                    except Exception as e:
                        st.error(f"Error checking band order: {e}")

        # Show Evaluation/Visualization
        render_evaluation_ui(config, pred_inventory, pred_metadata, framework_mode)

def render_data_selection(config, active_config_path, uploaded_files):
    """
    Renders the data selection UI (Band Selection) and manages the training suite.
    """
    if not uploaded_files:
        return config
        
    inventory, file_metadata = get_band_inventory(uploaded_files)
    
    st.header("📂 Data Selection")
    
    col1, col2 = st.columns(2)
    with col1:
        ds_name = st.text_input("Dataset Name (e.g. Region or Year)", value="Region_1")
        feature_bands = st.multiselect("Select Input Features", options=inventory)
    with col2:
        target_band = st.selectbox("Select Target Mask", options=inventory)
        task_type = st.radio("Task Type", options=["classification", "regression"], horizontal=True, 
                             index=0 if config.get("training", {}).get("task_type", "classification") == "classification" else 1)
        
        # New: Target Standardization Toggle for Regression
        standardize_target = False
        if task_type == 'regression':
            standardize_target = st.checkbox("Standardize Target (Recommended)", value=True, help="Transforms target to mean=0, std=1. Greatly improves DL convergence. Predictor will automatically denormalize.")
    
    # Adapt to Traditional ML requirements if needed
    is_ml = "ml_default.yaml" in active_config_path
    
    if st.button("➕ Add Dataset to Training Suite", use_container_width=True):
        if not feature_bands or not target_band:
            st.error("Please select at least one input feature and a target mask.")
        else:
            selected_files = set([parse_band_selection(b)[0] for b in feature_bands] + [parse_band_selection(target_band)[0]])
            ref_file = list(selected_files)[0]
            ref_meta = file_metadata[ref_file]
            
            valid = True
            for f in selected_files:
                meta = file_metadata[f]
                res_match = all(np.isclose(meta["res"][i], ref_meta["res"][i]) for i in range(2))
                if meta["crs"] != ref_meta["crs"] or meta["width"] != ref_meta["width"] or meta["height"] != ref_meta["height"] or not res_match:
                    valid = False
                    st.error(f"Error: {f} does not share the same CRS/dimensions/resolution as {ref_file}.")
                    break
            
            # Check band count consistency
            existing_datasets = config.get("data", {}).get("datasets", [])
            if valid and existing_datasets:
                first_ds_path = Path(existing_datasets[0]["features"])
                if first_ds_path.exists():
                    with rasterio.open(first_ds_path) as src:
                        required_bands = src.count
                        if len(feature_bands) != required_bands:
                            valid = False
                            st.error(f"Error: Number of selected bands ({len(feature_bands)}) does not match existing datasets in suite ({required_bands} bands).")

            if valid:
                try:
                    # Stacking & Saving Features
                    feature_arrays = []
                    for fb in feature_bands:
                        f_name, b_num = parse_band_selection(fb)
                        f = file_metadata[f_name]["file_obj"]
                        with MemoryFile(f.getvalue()) as memfile:
                            with memfile.open() as src:
                                feature_arrays.append(src.read(b_num))
                    
                    stacked_features = np.stack(feature_arrays)
                    
                    st.info(f"Processing '{ds_name}'...")
                    
                    # Normalization: Calculate or Use Global Stats
                    if "data" not in config: config["data"] = {}
                    
                    # Recalculate if stats missing, dimension mismatch, or starting a new suite
                    stats_missing = "mean" not in config["data"] or "std" not in config["data"]
                    dim_mismatch = not stats_missing and len(config["data"]["mean"]) != stacked_features.shape[0]
                    is_new_suite = not config.get("data", {}).get("datasets", [])

                    if stats_missing or dim_mismatch or is_new_suite:
                        if dim_mismatch:
                            st.warning(f"Band dimension mismatch (Config: {len(config['data']['mean'])}, Data: {stacked_features.shape[0]}). Recalculating stats...")
                        elif is_new_suite:
                            st.info("Initializing new training suite. Calculating normalization stats...")
                        else:
                            st.info("Calculating new global normalization stats (Z-score)...")
                        
                        g_means, g_stds = calculate_global_stats(stacked_features)
                        config["data"]["mean"] = g_means
                        config["data"]["std"] = g_stds
                        
                    st.info("Applying Z-score standardization...")
                    # Show stats for verification
                    st.write("**Normalization Stats (per band):**")
                    for idx, (m, s) in enumerate(zip(config["data"]["mean"], config["data"]["std"])):
                        st.write(f"- Band {idx+1}: Mean={m:.2f}, Std={s:.2f}")
                    
                    stacked_features = standardize(stacked_features, config["data"]["mean"], config["data"]["std"])
                    
                    out_dir = Path("data")
                    out_dir.mkdir(exist_ok=True)
                    
                    features_profile = ref_meta["profile"].copy()
                    features_profile.update(count=stacked_features.shape[0], dtype=rasterio.float32)
                    
                    safe_name = ds_name.replace(" ", "_").lower()
                    features_filename = f"features_{safe_name}.tif"
                    target_filename = f"target_{safe_name}.tif"
                    
                    features_path = out_dir / features_filename
                    with rasterio.open(features_path, 'w', **features_profile) as dst:
                        dst.write(stacked_features)
                        dst.descriptions = tuple(feature_bands)
                        
                    # Target Saving
                    t_name, t_num = parse_band_selection(target_band)
                    t = file_metadata[t_name]["file_obj"]
                    target_profile = file_metadata[t_name]["profile"].copy()
                    target_profile.update(count=1, dtype=rasterio.float32)
                    
                    target_path = out_dir / target_filename
                    with MemoryFile(t.getvalue()) as memfile:
                        with memfile.open() as src:
                            target_array = src.read(t_num).astype(np.float32)
                    
                    if standardize_target:
                        st.info("Standardizing Target Mask...")
                        t_mean = float(np.nanmean(target_array))
                        t_std = float(np.nanstd(target_array))
                        if t_std > 0:
                            target_array = (target_array - t_mean) / (t_std + 1e-8)
                            config["data"]["target_mean"] = t_mean
                            config["data"]["target_std"] = t_std
                        else:
                            st.warning("Target has zero variance. Skipping target standardization.")

                    with rasterio.open(target_path, 'w', **target_profile) as dst:
                        dst.write(target_array, 1)

                    # --- Automatic Class Mapping for Classification ---
                    if task_type == 'classification':
                        unique_values = np.unique(target_array)
                        # Remove common NoData values if they are extreme, but keep them if they look like small class indices
                        # If the user has specific values, we map ALL unique values found.
                        
                        existing_map = config.get("data", {}).get("class_map", {})
                        if not existing_map:
                            # Create new map: {original_value: index}
                            # Ensure we handle native python types for YAML serialization
                            new_map = {int(val): i for i, val in enumerate(sorted(unique_values))}
                            if "data" not in config: config["data"] = {}
                            config["data"]["class_map"] = new_map
                            st.info(f"Detected {len(new_map)} unique classes: {list(new_map.keys())}")
                        else:
                            # Check if new unique values are already in the map
                            for val in unique_values:
                                if int(val) not in existing_map:
                                    # Append new value
                                    new_idx = max(existing_map.values()) + 1
                                    existing_map[int(val)] = new_idx
                                    st.warning(f"New class value {int(val)} detected and added to map as index {new_idx}")
                            config["data"]["class_map"] = existing_map

                    # Update Config
                    if "data" not in config: config["data"] = {}
                    if "datasets" not in config["data"]: config["data"]["datasets"] = []
                    
                    new_ds = {"name": ds_name, "features": str(features_path), "target": str(target_path)}
                    config["data"]["datasets"] = [d for d in config["data"]["datasets"] if d['name'] != ds_name]
                    config["data"]["datasets"].append(new_ds)
                    
                    if "model" not in config: config["model"] = {}
                    config["model"]["in_channels"] = stacked_features.shape[0]
                    config["data"]["input_channels"] = list(range(1, stacked_features.shape[0] + 1))
                    
                    # Update common training task type
                    if "training" not in config: config["training"] = {}
                    config["training"]["task_type"] = task_type
                    
                    # ML Specific Updates
                    if is_ml:
                        config["data"]["features"] = str(features_path)
                        config["data"]["target"] = str(target_path)
                        config["data"]["mask_type"] = task_type
                    
                    # Save persistence
                    with open(active_config_path, "w") as f:
                        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
                        
                    st.success(f"Dataset '{ds_name}' successfully added to Training Suite!")
                except Exception as e:
                    st.error(f"Error processing data: {e}")

    # Show currently added datasets
    if "data" in config and config["data"].get("datasets"):
        st.subheader("📊 Current Training Suite")
        for i, ds in enumerate(config["data"]["datasets"]):
            col_ds1, col_ds2 = st.columns([4, 1])
            col_ds1.write(f"**{ds['name']}**: `{ds['features']}` | `{ds['target']}`")
            if col_ds2.button("remove", key=f"remove_ds_{active_config_path}_{i}"):
                config["data"]["datasets"].pop(i)
                # If no more datasets, maybe clear class map? (Optional)
                if not config["data"]["datasets"]:
                    if "class_map" in config["data"]: del config["data"]["class_map"]
                with open(active_config_path, "w") as f:
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
                st.rerun()
        
        # Display and Edit Mapping
        if "class_map" in config["data"]:
            with st.expander("🏷️ Class Mapping & Names", expanded=True):
                cmap = config["data"]["class_map"]
                # Default to mapping by original value (str(k) for YAML consistency)
                cnames = config["data"].get("class_names", {str(k): f"Class {k}" for k in cmap.keys()})
                
                # Check if we have old-style index-based mapping and migrate it
                # If all keys are small integers (indices) but original values are different, we might need migration
                # But for simplicity, we'll just let the user re-assign them if they were already there, 
                # or try a heuristic: if key not in cmap keys, it's probably an index.
                
                st.write("Assign names to your classes (used for visualization and reports):")
                
                updated_names = {}
                cols_map = st.columns(len(cmap) if len(cmap) < 5 else 4)
                for i, (orig_val, idx) in enumerate(sorted(cmap.items(), key=lambda t: t[1])):
                    with cols_map[i % len(cols_map)]:
                        # Try to find current name using original value
                        current_name = cnames.get(str(orig_val))
                        if current_name is None:
                            # Fallback to index-based if migratng
                            current_name = cnames.get(str(idx), f"Class {orig_val}")
                            
                        new_name = st.text_input(f"Value {orig_val} (Idx {idx})", value=current_name, key=f"cname_{active_config_path}_{idx}")
                        updated_names[str(orig_val)] = new_name
                
                if updated_names != cnames:
                    config["data"]["class_names"] = updated_names
                    with open(active_config_path, "w") as f:
                        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
                
                st.caption("0 is typically background. These names will be used in the visualizer.")
    return config

def render_config_editor(config_dict, save_path, schema=None, exclude=None):
    """
    Dynamically renders Streamlit inputs for a given configuration dictionary.
    Supports a schema to define specific widget types and optionally excludes certain keys.
    """
    st.subheader(f"⚙️ Configuration Editor ({Path(save_path).name})")
    
    updated_config = config_dict.copy()
    schema = schema or {}
    exclude = exclude or []
    
    # Iterate through top-level sections
    for section, content in config_dict.items():
        if section in exclude:
            continue
            
        if isinstance(content, dict):
            section_label = section.replace("_", " ").title()
            
            # Check if all keys in this section are excluded
            visible_keys = [k for k in content.keys() if k not in exclude]
            if not visible_keys:
                continue
                
            with st.expander(section_label, expanded=True):
                # Basic 2-column layout for section fields
                cols = st.columns(2)
                section_schema = schema.get(section, {})
                
                col_idx = 0
                for key, value in content.items():
                    if key in exclude:
                        continue
                        
                    with cols[col_idx % 2]:
                        col_idx += 1
                        label = key.replace("_", " ").title()
                        field_schema = section_schema.get(key, {})
                        widget_type = field_schema.get("type")
                        
                        key_id = f"{section}_{key}"
                        
                        if widget_type == "selectbox":
                            options = field_schema.get("options", [])
                            # Handle current value being null/None
                            current_val_str = "null" if value is None else value
                            try:
                                idx = options.index(current_val_str) if current_val_str in options else 0
                            except ValueError:
                                idx = 0
                            
                            selected = st.selectbox(label, options=options, index=idx, key=key_id)
                            
                            # Handle "Custom..." and "null"
                            if selected == "Custom...":
                                custom_val = st.text_input(f"Custom {label}", key=f"custom_{key_id}")
                                updated_config[section][key] = custom_val if custom_val else value
                            elif selected == "null":
                                updated_config[section][key] = None
                            else:
                                updated_config[section][key] = selected
                                
                        elif widget_type == "slider":
                            min_val = field_schema.get("min", 1)
                            max_val = field_schema.get("max", 100)
                            step = field_schema.get("step", 1)
                            
                            # Special case: handle None for max_depth
                            if key == "max_depth" and value is None:
                                use_limit = st.checkbox("Limit Depth", value=False, key=f"limit_{key_id}")
                                if use_limit:
                                    updated_config[section][key] = st.slider(label, min_val, max_val, 20, step, key=f"slider_{key_id}")
                                else:
                                    updated_config[section][key] = None
                                    st.info("No Depth Limit.")
                            else:
                                updated_config[section][key] = st.slider(label, min_val, max_val, value if value else min_val, step, key=key_id)
                        
                        elif isinstance(value, bool):
                            updated_config[section][key] = st.checkbox(label, value=value, key=key_id)
                        elif isinstance(value, (int, float)):
                            step = field_schema.get("step", 0.01 if isinstance(value, float) else 1)
                            fmt = field_schema.get("format")
                            updated_config[section][key] = st.number_input(label, value=value, step=step, format=fmt, key=key_id)
                        elif isinstance(value, str):
                            updated_config[section][key] = st.text_input(label, value=value, key=key_id)
                        elif value is None:
                            updated_config[section][key] = st.text_input(label, value="", key=key_id)
        else:
            # Top-level keys
            label = section.replace("_", " ").title()
            key_id = f"top_{section}"
            if isinstance(content, bool):
                updated_config[section] = st.checkbox(label, value=content, key=key_id)
            elif isinstance(content, (int, float)):
                updated_config[section] = st.number_input(label, value=content, key=key_id)
            elif isinstance(content, str):
                updated_config[section] = st.text_input(label, value=content, key=key_id)

    if st.button(f"💾 Save Configuration to {Path(save_path).name}", use_container_width=True):
        with open(save_path, "w") as f:
            yaml.dump(updated_config, f, default_flow_style=False, sort_keys=False)
        st.success(f"Configuration saved to {save_path}!")
        return updated_config
    
    return updated_config

def main():
    st.set_page_config(page_title="GeoAI Framework", layout="wide")
    st.title("🌍 GeoAI Framework")
    
    # Paradigm Toggle
    framework_mode = st.radio("Framework Mode", options=["Deep Learning (PyTorch)", "Traditional ML (RF/XGB)"], horizontal=True)
    
    # Sidebar
    st.sidebar.header("Data Input")
    uploaded_files = st.sidebar.file_uploader("Upload GeoTIFFs", type=["tif", "tiff"], accept_multiple_files=True)
    
    # Config Paths
    DL_CONFIG_PATH = "config/default.yaml"
    ML_CONFIG_PATH = "config/ml_default.yaml"
    
    # Initialize configs in session state
    if "dl_config" not in st.session_state:
        st.session_state.dl_config = load_config(DL_CONFIG_PATH) if Path(DL_CONFIG_PATH).exists() else {}
    if "ml_config" not in st.session_state:
        if Path(ML_CONFIG_PATH).exists():
            st.session_state.ml_config = load_config(ML_CONFIG_PATH)
        else:
            st.session_state.ml_config = {
                "model": {"architecture": "Random Forest", "experiment_name": "rf_test_01"},
                "training": {"n_estimators": 100, "max_depth": 20, "task_type": "classification", "val_split": 0.2},
                "data": {"max_pixels_per_class": 10000}
            }
            Path("config").mkdir(exist_ok=True)
            with open(ML_CONFIG_PATH, "w") as f:
                yaml.dump(st.session_state.ml_config, f, default_flow_style=False, sort_keys=False)
    
    if "ml_data_loaded" not in st.session_state:
        st.session_state.ml_data_loaded = False
    
    # Select active config based on mode
    if framework_mode == "Deep Learning (PyTorch)":
        config = st.session_state.dl_config
        active_config_path = DL_CONFIG_PATH
    else:
        config = st.session_state.ml_config
        active_config_path = ML_CONFIG_PATH # Corrected from ML
    # Data Selection UI (Identity Flow across modes)
    config = render_data_selection(config, active_config_path, uploaded_files)
    
    # Configuration Schema Definitions
    # --- Dynamic Schema based on task_type ---
    current_task = config.get("training", {}).get("task_type", "binary")
    
    # Filter loss functions based on task type
    if current_task == "regression":
        loss_options = ["mse", "mae"]
    elif current_task in ["multiclass", "classification"]:
        loss_options = ["cross_entropy"]
    else: # binary
        loss_options = ["bce_dice", "jaccard", "focal", "bce"]

    DL_SCHEMA = {
        "project_name": {"type": "text_input"},
        "experiment_name": {"type": "text_input"},
        "data": {
            "img_size": {"type": "number_input", "step": 32},
            "val_split": {"type": "slider", "min": 0.05, "max": 0.5, "step": 0.05}
        },
        "model": {
            "architecture": {"type": "selectbox", "options": ["Unet", "UnetPlusPlus", "DeepLabV3", "DeepLabV3Plus", "FPN", "PAN", "PSPNet", "Linknet", "MAnet", "Custom..."]},
            "encoder": {"type": "selectbox", "options": ["resnet18", "resnet34", "resnet50", "resnet101", "efficientnet-b0", "efficientnet-b4", "mit_b0", "mit_b3", "mit_b5", "Custom..."]},
            "weights": {"type": "selectbox", "options": ["imagenet", "null"]}
        },
        "training": {
            "learning_rate": {"type": "number_input", "step": 0.0001 if current_task == "regression" else 0.00001, "format": "%.5f"},
            "loss_function": {"type": "selectbox", "options": loss_options},
            "optimizer": {"type": "selectbox", "options": ["adamw", "adam", "sgd"]}
        }
    }
    
    ML_SCHEMA = {
        "model": {
            "architecture": {"type": "selectbox", "options": ["Random Forest", "XGBoost", "Gradient Boosting", "Support Vector Machine (SVM)", "K-Nearest Neighbors (KNN)"]},
            "experiment_name": {"type": "text_input"}
        },
        "training": {
            "val_split": {"type": "slider", "min": 0.1, "max": 0.5, "step": 0.05},
            "n_estimators": {"type": "slider", "min": 10, "max": 500, "step": 10},
            "max_depth": {"type": "slider", "min": 2, "max": 50, "step": 1}
        }
    }

    schema = DL_SCHEMA if framework_mode == "Deep Learning (PyTorch)" else ML_SCHEMA

    # Fields to exclude from the UI (Managed automatically or sensitive)
    EXCLUDE_FIELDS = [
        "device", "seed", "num_workers", "system", # System
        "train_dir", "val_dir", "datasets", "features", "target", # Data Paths
        "train_features", "train_target",
        "in_channels", "input_channels", # Auto-calculated
        "mask_type", "normalization_max" # Deprecated/Unified
    ]

    # Configuration Editor (Unified Component)
    config = render_config_editor(config, active_config_path, schema=schema, exclude=EXCLUDE_FIELDS)
    
    # Sync back to correct session state source of truth
    if framework_mode == "Deep Learning (PyTorch)":
        st.session_state.dl_config = config
    else:
        st.session_state.ml_config = config
        
    # Use the synced config for the rest of the app
    st.session_state.config = config 

    st.markdown("---")
    
    # --- Training Section ---
    if framework_mode == "Deep Learning (PyTorch)":
        # Initialize session state for process
        if "process_active" not in st.session_state:
            st.session_state.process_active = False
        if "process" not in st.session_state:
            st.session_state.process = None
        if "data_loaded" not in st.session_state:
            st.session_state.data_loaded = False
            
        st.subheader("Data Loading")
        
        datasets = config.get("data", {}).get("datasets", [])
        
        if st.button("📥 Load Processed Training Suite", use_container_width=True):
            if not datasets:
                st.error("No datasets found in training suite. Please add some first.")
                st.session_state.data_loaded = False
            else:
                missing = []
                for ds in datasets:
                    if not Path(ds['features']).exists() or not Path(ds['target']).exists():
                        missing.append(ds['name'])
                
                if not missing:
                    st.session_state.data_loaded = True
                    st.success(f"Successfully loaded {len(datasets)} dataset(s)!")
                else:
                    st.session_state.data_loaded = False
                    st.error(f"Missing data for: {', '.join(missing)}. Please re-process them.")
                
        if st.session_state.data_loaded and datasets:
            st.info(f"Ready to train with {len(datasets)} dataset(s).")
            with st.expander("📊 Data Preview", expanded=False):
                show_mask_toggle = st.checkbox("Show Mask Overlay", value=True)
                # Preview the first dataset by default
                selected_prev = st.selectbox("Select Dataset to Preview", [ds['name'] for ds in datasets])
                ds_prev = [d for d in datasets if d['name'] == selected_prev][0]
                task_mode = get_task_mode(config)
                if task_mode == 'binary': task_mode = 'classification'
                cnames = config.get('data', {}).get('class_names')
                cmap = config.get('data', {}).get('class_map')
                with st.spinner("Generating preview..."):
                    plot_data_preview(ds_prev['features'], ds_prev['target'], show_mask=show_mask_toggle, task_type=task_mode, class_names=cnames, class_map=cmap)
        
        st.markdown("---")
        
        # Training Control Section
        st.subheader("Training Controls")
        
        # Disable training if process is active or data is not loaded
        training_disabled = st.session_state.process_active or not st.session_state.data_loaded
        
        if st.button("🚀 Start Training", use_container_width=True, 
                     disabled=training_disabled):
            st.session_state.process_active = True
            st.rerun()
            
        if st.button("🛑 Cancel Training", use_container_width=True, 
                     disabled=not st.session_state.process_active):
            if st.session_state.process:
                st.session_state.process.terminate()
            st.session_state.process_active = False
            st.session_state.process = None
            st.warning("Training cancelled.")
            st.rerun()

        if st.session_state.process_active:
            st.subheader("Training Logs")
            
            # Inject the auto-scroll script ONCE outside the loop.
            # This script uses an interval to dynamically find the code block and scroll its nearest scrollable parent.
            st.components.v1.html(
                """
                <script>
                    const doc = window.parent.document;
                    const autoScroll = setInterval(function() {
                        // Find all code blocks
                        const codeBlocks = doc.querySelectorAll('code');
                        if (codeBlocks.length > 0) {
                            // Get the most recent one being updated
                            const lastCodeBlock = codeBlocks[codeBlocks.length - 1];
                            
                            // Walk up the DOM to find the nearest scrollable container
                            let parent = lastCodeBlock.parentElement;
                            while (parent && parent !== doc.body) {
                                const overflowY = window.getComputedStyle(parent).overflowY;
                                if (overflowY === 'auto' || overflowY === 'scroll') {
                                    parent.scrollTop = parent.scrollHeight;
                                    break;
                                }
                                parent = parent.parentElement;
                            }
                        }
                    }, 100); // Checks and scrolls every 100ms
                </script>
                """,
                height=0
            )
            
            # Fixed-height scrollable container
            with st.container(height=400):
                log_placeholder = st.empty()
                
                # Start the training process if not already started
                if st.session_state.process is None:
                    process = subprocess.Popen(
                        ["python", "scripts/train.py", "--config", active_config_path],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1
                    )
                    st.session_state.process = process
                else:
                    process = st.session_state.process
                
                import queue
                import threading
                import time

                def enqueue_output(out, q):
                    for line in iter(out.readline, ''):
                        q.put(line)
                    out.close()

                q = queue.Queue()
                t = threading.Thread(target=enqueue_output, args=(process.stdout, q))
                t.daemon = True
                t.start()

                log_output = ""
                while True:
                    # Check if process is still running
                    is_running = process.poll() is None
                    
                    # Read all currently available lines
                    new_data = False
                    while not q.empty():
                        line = q.get_nowait()
                        log_output += line
                        new_data = True
                    
                    if new_data:
                        log_placeholder.code(log_output, language="text")
                    
                    if not is_running:
                        break
                    
                    # Small sleep to yield control and allow UI interactions (like Cancel)
                    time.sleep(0.1)
                
                    
                process.stdout.close()
                return_code = process.wait()
                
                st.session_state.process = None
                st.session_state.process_active = False
                
                if return_code == 0:
                    st.success("Training completed successfully!")
                elif return_code in [-15, 15]: 
                    st.info("Training process terminated.")
                else:
                    st.error(f"Training failed with exit code {return_code}")
                
                if st.button("Done"):
                    st.rerun()

        render_prediction_section(config, active_config_path, framework_mode)

    elif framework_mode == "Traditional ML (RF/XGB)":
        st.subheader("Data Loading")
        
        ml_datasets = config.get("data", {}).get("datasets", [])
        
        if st.button("📥 Load ML Training Suite", use_container_width=True):
            if not ml_datasets:
                st.error("No ML datasets found. Please configure features/targets first.")
                st.session_state.ml_data_loaded = False
            else:
                missing = []
                for ds in ml_datasets:
                    if not Path(ds['features']).exists() or not Path(ds['target']).exists():
                        missing.append(ds['name'])
                
                if not missing:
                    st.session_state.ml_data_loaded = True
                    st.success(f"Successfully loaded {len(ml_datasets)} dataset(s)!")
                else:
                    st.session_state.ml_data_loaded = False
                    st.error(f"Missing data for: {', '.join(missing)}.")
        
        if st.session_state.ml_data_loaded and ml_datasets:
            st.info(f"Ready to train with {len(ml_datasets)} dataset(s).")
            with st.expander("📊 Data Preview", expanded=False):
                show_mask_toggle = st.checkbox("Show Mask Overlay (ML)", value=True)
                selected_prev = st.selectbox("Select ML Dataset to Preview", [ds['name'] for ds in ml_datasets])
                ds_prev = [d for d in ml_datasets if d['name'] == selected_prev][0]
                task_mode = get_task_mode(config)
                cnames = config.get('data', {}).get('class_names')
                cmap = config.get('data', {}).get('class_map')
                with st.spinner("Generating ML preview..."):
                    plot_data_preview(ds_prev['features'], ds_prev['target'], show_mask=show_mask_toggle, task_type=task_mode, class_names=cnames, class_map=cmap)
        
        st.markdown("---")
        st.subheader("ML Execution")
        
        col_ml1, col_ml2 = st.columns(2)
        with col_ml1:
            if st.button("🚀 Train ML Model", use_container_width=True, disabled=not st.session_state.ml_data_loaded):
                progress_bar = st.progress(0, text="Initializing ML training...")
                with st.status("Training Machine Learning Model...", expanded=True) as status:
                    def ml_train_callback(msg, fraction):
                        status.write(msg)
                        progress_bar.progress(min(1.0, fraction), text=f"Training Step: {msg}")
                    
                    try:
                        metrics = train_ml_model(config_path=active_config_path, progress_callback=ml_train_callback)
                        status.update(label="Training complete!", state="complete", expanded=False)
                        st.success("Model trained successfully!")
                        
                        # Display metrics
                        m_cols = st.columns(len(metrics))
                        for i, (name, val) in enumerate(metrics.items()):
                            m_cols[i].metric(name, f"{val:.4f}")
                    except Exception as e:
                        status.update(label="Training failed!", state="error")
                        st.error(f"Error during ML training: {e}")
                progress_bar.empty()
                
        render_prediction_section(config, active_config_path, framework_mode)

if __name__ == "__main__":
    main()