import streamlit as st
import yaml
import subprocess
from pathlib import Path
import rasterio
from rasterio.io import MemoryFile
import matplotlib.pyplot as plt
import numpy as np
import wandb

from scripts.evaluate import calculate_metrics_from_arrays
from src.utils.checkpoints import list_checkpoints

def plot_data_preview(input_file, mask_file, show_mask: bool = True):
    """Generate a downsampled side-by-side preview of the input image and mask."""
    try:
        def open_src(file_obj):
            if hasattr(file_obj, "getvalue"):
                # Use MemoryFile for uploaded streamlit objects
                with MemoryFile(file_obj.getvalue()) as memfile:
                    return memfile.open()
            else:
                # Open directly for paths/strings
                return rasterio.open(file_obj)

        with open_src(input_file) as src_in, open_src(mask_file) as src_mask:
            # Calculate downsampled shape (e.g., max 512 pixels on longest side to save memory)
            max_dim = 512
            scale = min(max_dim / src_in.width, max_dim / src_in.height)
            out_shape = (int(src_in.height * scale), int(src_in.width * scale))
            out_shape = (max(1, out_shape[0]), max(1, out_shape[1]))
            
            # Determine if we can show RGB (at least 3 bands)
            if src_in.count >= 3:
                # Extract first 3 bands for RGB
                img_data = src_in.read([1, 2, 3], out_shape=out_shape).astype(np.float32)
                # Transpose to (H, W, C)
                img_data = np.transpose(img_data, (1, 2, 0))
                is_rgb = True
            else:
                # Extract first band for grayscale
                img_data = src_in.read(1, out_shape=out_shape).astype(np.float32)
                is_rgb = False
                
            mask_data = src_mask.read(1, out_shape=out_shape).astype(np.float32)
            
        # Normalize for visualization
        if is_rgb:
            # Per-channel normalization
            for i in range(3):
                ch = img_data[:, :, i]
                if ch.max() > ch.min():
                    img_data[:, :, i] = (ch - ch.min()) / (ch.max() - ch.min())
        else:
            if img_data.max() > img_data.min():
                img_data = (img_data - img_data.min()) / (img_data.max() - img_data.min())
                
        if mask_data.max() > mask_data.min():
            mask_data = (mask_data - mask_data.min()) / (mask_data.max() - mask_data.min())
            
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Plot the input image
        if is_rgb:
            st.write(f"Preview RGB Shape: {img_data.shape}") # Debug
            ax.imshow(img_data)
        else:
            st.write(f"Preview Grayscale Shape: {img_data.shape}") # Debug
            ax.imshow(img_data, cmap='gray')
            
        ax.set_title("Input Image with Mask Overlay")
        ax.axis('off')
        
        # Mask out 0/background values of the target mask so they are transparent
        # In a normalized binary/multiclass mask, 0 usually represents background
        if show_mask:
            masked_mask = np.ma.masked_where(mask_data == 0, mask_data)
            
            # Overlay the mask with a color map and some transparency
            # Using 'viridis' to make it stand out against the B&W image
            ax.imshow(masked_mask, cmap='viridis', alpha=0.5, interpolation='none')
        
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error generating preview: {e}")

def plot_prediction_results(bg_arrays, actual_mask_array, pred_mask_array):
    """
    Plot actual, predicted, and error maps over a ground map.
    """
    try:
        # Normalize BG
        bg_stacked = np.stack(bg_arrays, axis=-1) # H,W,C
        b_min = bg_stacked.min(axis=(0,1), keepdims=True)
        b_max = bg_stacked.max(axis=(0,1), keepdims=True)
        # Avoid div by zero
        b_diff = b_max - b_min
        b_diff[b_diff == 0] = 1.0
        bg_norm = (bg_stacked - b_min) / b_diff
        
        if bg_norm.shape[-1] == 1:
            bg_norm = bg_norm.squeeze(-1) # For grayscale
            cmap = 'gray'
        else:
            # Ensure 3 bands for RGB
            if bg_norm.shape[-1] >= 3:
                bg_norm = bg_norm[:, :, :3]
            cmap = None
            
        st.write(f"Prediction BG Shape: {bg_norm.shape}, Cmap: {cmap}") # Debug
            
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Plot 1: Actual
        axes[0].set_title("Actual Target")
        if cmap is None:
            axes[0].imshow(bg_norm)
        else:
            axes[0].imshow(bg_norm, cmap=cmap)
        if actual_mask_array is not None:
            actual_overlay = np.ma.masked_where(actual_mask_array == 0, actual_mask_array)
            axes[0].imshow(actual_overlay, cmap='viridis', vmin=0, vmax=1, alpha=0.5, interpolation='none')
        else:
            axes[0].text(0.5, 0.5, 'No Actual Mask Selected', ha='center', va='center', transform=axes[0].transAxes, color='red', fontsize=12)
        axes[0].axis('off')
        
        # Plot 2: Predicted
        axes[1].set_title("Predicted Target")
        if cmap is None:
            axes[1].imshow(bg_norm)
        else:
            axes[1].imshow(bg_norm, cmap=cmap)
        pred_overlay = np.ma.masked_where(pred_mask_array == 0, pred_mask_array)
        axes[1].imshow(pred_overlay, cmap='viridis', vmin=0, vmax=1, alpha=0.5, interpolation='none')
        axes[1].axis('off')
        
        # Plot 3: Error Map
        axes[2].set_title("Error Map (Red = Mistakes)")
        if cmap is None:
            axes[2].imshow(bg_norm)
        else:
            axes[2].imshow(bg_norm, cmap=cmap)
        
        if actual_mask_array is not None:
            H, W = pred_mask_array.shape
            error_map = np.zeros((H, W, 4), dtype=np.float32)
            
            a_bin = (actual_mask_array > 0)
            p_bin = (pred_mask_array > 0)
            
            errors = a_bin != p_bin
            
            error_map[errors] = [1.0, 0.0, 0.0, 0.8] # Strong Red
            
            axes[2].imshow(error_map, interpolation='none')
        else:
            axes[2].text(0.5, 0.5, 'No Actual Mask Selected', ha='center', va='center', transform=axes[2].transAxes, color='red', fontsize=12)
        
        axes[2].axis('off')
        
        plt.tight_layout()
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error generating prediction plots: {e}")

def load_config(config_path: str) -> dict:
    """Read and parse the YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def get_band_inventory(uploaded_files):
    inventory = []
    metadata = {}
    for f in uploaded_files:
        try:
            with MemoryFile(f.getvalue()) as memfile:
                with memfile.open() as src:
                    num_bands = src.count
                    metadata[f.name] = {
                        "crs": src.crs,
                        "width": src.width,
                        "height": src.height,
                        "transform": src.transform,
                        "res": src.res, # (x_res, y_res)
                        "profile": src.profile,
                        "file_obj": f
                    }
                    for b in range(1, num_bands + 1):
                        inventory.append(f"{f.name} - Band {b}")
        except Exception as e:
            st.error(f"Error reading {f.name}: {e}")
    return inventory, metadata

def main():
    st.set_page_config(page_title="GeoAI Framework", layout="wide")
    st.title("🌍 GeoAI Framework")
    
    # Sidebar
    st.sidebar.header("Data Input")
    uploaded_files = st.sidebar.file_uploader("Upload GeoTIFFs", type=["tif", "tiff"], accept_multiple_files=True)
    
    # Load config into session state
    config_file = "config/default.yaml"
    if "config" not in st.session_state:
        if Path(config_file).exists():
            st.session_state.config = load_config(config_file)
        else:
            st.session_state.config = {}
            st.warning(f"Configuration file not found at {config_file}")
            
    config = st.session_state.config
    
    if uploaded_files:
        inventory, file_metadata = get_band_inventory(uploaded_files)
        
        st.header("Band Selection")
        feature_bands = st.multiselect("Select Input Features", options=inventory)
        target_band = st.selectbox("Select Target Mask", options=inventory)
        
        if st.button("Process & Prepare Data"):
            if not feature_bands or not target_band:
                st.error("Please select at least one input feature and a target mask.")
            else:
                selected_files = set([b.split(" - Band ")[0] for b in feature_bands] + [target_band.split(" - Band ")[0]])
                
                ref_file = list(selected_files)[0]
                ref_meta = file_metadata[ref_file]
                
                valid = True
                for f in selected_files:
                    meta = file_metadata[f]
                    res_match = all(np.isclose(meta["res"][i], ref_meta["res"][i]) for i in range(2))
                    if meta["crs"] != ref_meta["crs"] or meta["width"] != ref_meta["width"] or meta["height"] != ref_meta["height"] or not res_match:
                        valid = False
                        break
                        
                if not valid:
                    st.error("Error: Selected bands come from files that do not share the exact same CRS and spatial dimensions.")
                else:
                    try:
                        # Stacking & Saving Features
                        feature_arrays = []
                        for fb in feature_bands:
                            f_name, b_num = fb.split(" - Band ")
                            b_num = int(b_num)
                            f = file_metadata[f_name]["file_obj"]
                            with MemoryFile(f.getvalue()) as memfile:
                                with memfile.open() as src:
                                    feature_arrays.append(src.read(b_num))
                        
                        stacked_features = np.stack(feature_arrays)
                        
                        out_dir = Path("data")
                        out_dir.mkdir(exist_ok=True)
                        
                        features_profile = ref_meta["profile"].copy()
                        features_profile.update(count=stacked_features.shape[0])
                        
                        features_path = out_dir / "processed_features.tif"
                        with rasterio.open(features_path, 'w', **features_profile) as dst:
                            dst.write(stacked_features)
                            dst.descriptions = tuple(feature_bands)
                            
                        # Target Saving
                        t_name, t_num = target_band.split(" - Band ")
                        t_num = int(t_num)
                        t = file_metadata[t_name]["file_obj"]
                        
                        target_profile = file_metadata[t_name]["profile"].copy()
                        target_profile.update(count=1)
                        
                        target_path = out_dir / "processed_target.tif"
                        with MemoryFile(t.getvalue()) as memfile:
                            with memfile.open() as src:
                                target_array = src.read(t_num)
                                
                        with rasterio.open(target_path, 'w', **target_profile) as dst:
                            dst.write(target_array, 1)
                            
                        # Update Config
                        if "data" not in st.session_state.config:
                            st.session_state.config["data"] = {}
                        if "model" not in st.session_state.config:
                            st.session_state.config["model"] = {}
                            
                        st.session_state.config["data"]["train_features"] = str(features_path)
                        st.session_state.config["data"]["train_target"] = str(target_path)
                        st.session_state.config["model"]["in_channels"] = stacked_features.shape[0]
                        st.session_state.config["data"]["input_channels"] = list(range(1, stacked_features.shape[0] + 1))
                        
                        st.success(f"Data successfully processed and saved to {features_path} and {target_path}!")
                        
                    except Exception as e:
                        st.error(f"Error processing data: {e}")

    # Main Area: Config Editor
    st.header("⚙️ Configuration Editor")
    
    if config:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Main Settings")
            new_experiment_name = st.text_input("Experiment Name", value=config.get("experiment_name", ""))
            
            model_cfg = config.get("model", {})
            
            arch_options = ["Unet", "UnetPlusPlus", "DeepLabV3", "DeepLabV3Plus", "FPN", "PAN", "PSPNet", "Linknet", "MAnet", "Custom..."]
            current_arch = model_cfg.get("architecture", "Unet")
            arch_idx = arch_options.index(current_arch) if current_arch in arch_options else 0
            new_architecture = st.selectbox("Architecture", options=arch_options, index=arch_idx)
            
            custom_arch = ""
            if new_architecture == "Custom...":
                custom_arch = st.text_input("Enter custom architecture name:")
            
            encoder_options = [
                "resnet18", "resnet34", "resnet50", "resnet101",
                "resnext50_32x4d", "resnext101_32x8d", "resnest50d", "resnest101e",
                "efficientnet-b0", "efficientnet-b3", "efficientnet-b4", "efficientnet-b7", "tu-tf_efficientnetv2_s",
                "mit_b0", "mit_b3", "mit_b5",
                "convnext_tiny", "convnext_small", "convnext_base",
                "mobilenet_v2", "mobilenet_v3_large",
                "Custom..."
            ]
            current_encoder = model_cfg.get("encoder", "mit_b5")
            encoder_idx = encoder_options.index(current_encoder) if current_encoder in encoder_options else 0
            new_encoder = st.selectbox("Encoder", options=encoder_options, index=encoder_idx)
            
            custom_encoder = ""
            if new_encoder == "Custom...":
                custom_encoder = st.text_input("Enter custom encoder name:")
            
            weights_options = ["imagenet", "null"]
            current_weights = model_cfg.get("weights", "imagenet")
            current_weights_str = "null" if current_weights is None else current_weights
            weights_idx = weights_options.index(current_weights_str) if current_weights_str in weights_options else 0
            new_weights_str = st.selectbox("Weights", options=weights_options, index=weights_idx)
            new_weights = None if new_weights_str == "null" else new_weights_str
            
        with col2:
            st.subheader("Hyperparameters")
            train_cfg = config.get("training", {})
            new_epochs = st.number_input("Epochs", min_value=1, value=int(train_cfg.get("epochs", 50)), step=1)
            new_batch_size = st.number_input("Batch Size", min_value=1, value=int(train_cfg.get("batch_size", 16)), step=1)
            new_lr = st.number_input("Learning Rate", min_value=0.0, value=float(train_cfg.get("learning_rate", 0.0001)), step=0.00001, format="%.5f")
            
            st.divider()
            st.subheader("Prediction Tiling")
            pred_cfg = config.get("prediction", {})
            new_tile_size = st.number_input("Prediction Tile Size", min_value=32, value=int(pred_cfg.get("tile_size", 256)), step=32)
            new_overlap = st.number_input("Prediction Overlap", min_value=0, value=int(pred_cfg.get("overlap", 64)), step=8)
            
        with st.expander("Project Settings"):
            col_proj1, col_proj2 = st.columns(2)
            with col_proj1:
                new_project_name = st.text_input("Project Name", value=config.get("project_name", "N/A"))
                new_wandb_entity = st.text_input("W&B Entity", value=config.get("wandb_entity", "N/A"))
            with col_proj2:
                data_cfg = config.get("data", {})
                new_norm_max = st.number_input("Normalization Max", value=int(data_cfg.get("normalization_max", 4000)), step=100)

        with st.expander("Advanced Settings"):
            col3, col4 = st.columns(2)
            with col3:
                loss_options = ["bce_dice", "jaccard", "focal"]
                current_loss = train_cfg.get("loss_function", "bce_dice")
                loss_idx = loss_options.index(current_loss) if current_loss in loss_options else 0
                new_loss = st.selectbox("Loss Function", options=loss_options, index=loss_idx)
                
                opt_options = ["adamw", "adam", "sgd"]
                current_opt = train_cfg.get("optimizer", "adamw")
                opt_idx = opt_options.index(current_opt) if current_opt in opt_options else 0
                new_optimizer = st.selectbox("Optimizer", options=opt_options, index=opt_idx)
                
            with col4:
                new_val_split = st.number_input("Validation Split", min_value=0.0, max_value=1.0, value=float(data_cfg.get("val_split", 0.1)), step=0.05)
                new_img_size = st.number_input("Image Size", min_value=32, value=int(data_cfg.get("img_size", 256)), step=32)
        
        submit_config = st.button("Apply Configuration")
        
        if submit_config:
            # Update config dictionary
            config["project_name"] = new_project_name
            config["wandb_entity"] = new_wandb_entity
            config["experiment_name"] = new_experiment_name
            
            if "model" not in config:
                config["model"] = {}
                
            final_arch = custom_arch if new_architecture == "Custom..." and custom_arch else new_architecture
            final_encoder = custom_encoder if new_encoder == "Custom..." and custom_encoder else new_encoder
                
            config["model"]["architecture"] = final_arch
            config["model"]["encoder"] = final_encoder
            config["model"]["weights"] = new_weights
            
            if "training" not in config:
                config["training"] = {}
            config["training"]["epochs"] = new_epochs
            config["training"]["batch_size"] = new_batch_size
            config["training"]["learning_rate"] = new_lr
            config["training"]["loss_function"] = new_loss
            config["training"]["optimizer"] = new_optimizer
            
            if "data" not in config:
                config["data"] = {}
            config["data"]["val_split"] = new_val_split
            config["data"]["img_size"] = new_img_size
            config["data"]["normalization_max"] = new_norm_max
            
            if "prediction" not in config:
                config["prediction"] = {}
            config["prediction"]["tile_size"] = new_tile_size
            config["prediction"]["overlap"] = new_overlap
            
            st.session_state.config = config
            
            # Save to file
            with open(config_file, "w") as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
                
            st.success(f"Configuration updated and saved to {config_file}!")
    
    st.markdown("---")
    
    # Initialize session state for process
    if "process_active" not in st.session_state:
        st.session_state.process_active = False
    if "process" not in st.session_state:
        st.session_state.process = None
    if "data_loaded" not in st.session_state:
        st.session_state.data_loaded = False
        
    st.subheader("Data Loading")
    
    features_path = Path("data/processed_features.tif")
    target_path = Path("data/processed_target.tif")
    
    if st.button("📥 Load Processed Data", use_container_width=True):
        if features_path.exists() and target_path.exists():
            st.session_state.data_loaded = True
            st.success("Successfully loaded processed features and target data!")
        else:
            st.session_state.data_loaded = False
            st.error("Processed data files not found. Please upload and process data first.")
            
    if st.session_state.data_loaded:
        st.info(f"Ready to train with features: {features_path} and target: {target_path}")
        with st.expander("📊 Data Preview", expanded=True):
            show_mask_toggle = st.checkbox("Show Mask Overlay", value=True)
            plot_data_preview(features_path, target_path, show_mask=show_mask_toggle)
    
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
                    ["python", "scripts/train.py", "--config", config_file],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1
                )
                st.session_state.process = process
            else:
                process = st.session_state.process
            
            log_output = ""
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                if line:
                    log_output += line
                    # Update the log output. The JS interval will catch the update and scroll it.
                    log_placeholder.code(log_output, language="text")
                
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

    st.markdown("---")
    st.header("🔮 Prediction (Inference)")
    
    st.sidebar.markdown("---")
    st.sidebar.header("Prediction Data")
    pred_uploaded_files = st.sidebar.file_uploader("Upload GeoTIFFs for Prediction", type=["tif", "tiff"], accept_multiple_files=True, key="pred_uploader")
    
    if pred_uploaded_files:
        pred_inventory, pred_metadata = get_band_inventory(pred_uploaded_files)
        
        st.subheader("Band Selection for Prediction")
        
        # Try to load training feature descriptions to guide the user
        trained_bands_info = None
        if features_path.exists():
            try:
                with rasterio.open(features_path) as f_src:
                    if f_src.descriptions and any(f_src.descriptions):
                        trained_bands_info = f_src.descriptions
            except Exception:
                pass
                
        if trained_bands_info:
            st.info("**Training was performed with the following bands in exact order:**\n" + "\n".join([f"- {b}" for b in trained_bands_info]))
            
        pred_feature_bands = st.multiselect("Select Features for Prediction (MUST match training order exactly)", options=pred_inventory, key="pred_multiselect")
        
        # Model Checkpoint Selector
        st.markdown("---")
        st.subheader("Model Selection")
        available_checkpoints = list_checkpoints("models")
        if not available_checkpoints:
            st.warning("No model checkpoints found in `models/` directory. Train a model first or ensure .pth files exist.")
            selected_checkpoint = None
        else:
            # Default to the first one (often the alphabetical first, which might be a good default)
            selected_checkpoint = st.selectbox("Select Model Checkpoint", options=available_checkpoints)
            st.info(f"Using checkpoint: `models/{selected_checkpoint}`")

        col_pred1, col_pred2 = st.columns(2)
        
        with col_pred1:
            if st.button("🚀 Run Prediction", use_container_width=True):
                if not pred_feature_bands:
                    st.error("Please select at least one feature band for prediction.")
                else:
                    try:
                        # Save prediction input TIF
                        pred_selected_files = set([b.split(" - Band ")[0] for b in pred_feature_bands])
                        ref_file = list(pred_selected_files)[0]
                        ref_meta = pred_metadata[ref_file]
                        
                        pred_feature_arrays = []
                        for fb in pred_feature_bands:
                            f_name, b_num = fb.split(" - Band ")
                            b_num = int(b_num)
                            f = pred_metadata[f_name]["file_obj"]
                            with MemoryFile(f.getvalue()) as memfile:
                                with memfile.open() as src:
                                    pred_feature_arrays.append(src.read(b_num))
                        
                        pred_stacked = np.stack(pred_feature_arrays)
                        
                        out_dir = Path("data")
                        out_dir.mkdir(exist_ok=True)
                        
                        pred_input_path = out_dir / "prediction_input.tif"
                        pred_profile = ref_meta["profile"].copy()
                        pred_profile.update(count=pred_stacked.shape[0])
                        
                        with rasterio.open(pred_input_path, 'w', **pred_profile) as dst:
                            dst.write(pred_stacked)
                            dst.descriptions = tuple(pred_feature_bands)
                            
                        # Run predict.py
                        pred_out_path = out_dir / "prediction.tif"
                        checkpoint_path = f"models/{selected_checkpoint}" if selected_checkpoint else "models/best_model.pth"
                        
                        with st.spinner(f"Running prediction with {checkpoint_path}..."):
                            cmd = [
                                "python", "scripts/predict.py",
                                "--config", config_file,
                                "--input_image", str(pred_input_path),
                                "--output", str(pred_out_path),
                                "--model_path", checkpoint_path
                            ]
                            result = subprocess.run(cmd, capture_output=True, text=True)
                            
                            if result.returncode == 0:
                                st.success(f"Prediction successfully saved to {pred_out_path}!")
                            else:
                                st.error(f"Prediction failed:\n{result.stderr}")
                    except Exception as e:
                        st.error(f"Error preparing or running prediction data: {e}")
                        
        with col_pred2:
            if st.button("⚖️ Check Band Order", use_container_width=True):
                pred_input_path = Path("data/prediction_input.tif")
                if not features_path.exists():
                    st.error("Original training features (data/processed_features.tif) not found.")
                elif not pred_input_path.exists():
                    st.error("Prediction input (data/prediction_input.tif) not found. Run prediction first or ensure it was prepared.")
                else:
                    try:
                        with rasterio.open(features_path) as src_train, rasterio.open(pred_input_path) as src_pred:
                            train_desc = src_train.descriptions or tuple(f"Band {i+1}" for i in range(src_train.count))
                            pred_desc = src_pred.descriptions or tuple(f"Band {i+1}" for i in range(src_pred.count))
                            
                            st.write("### Band Order Comparison")
                            
                            # Using markdown table for comparison
                            md_table = "| Index | Training Band | Prediction Band | Match |\n|---|---|---|---|\n"
                            max_len = max(len(train_desc), len(pred_desc))
                            
                            all_match = len(train_desc) == len(pred_desc)
                            
                            for i in range(max_len):
                                t_band = train_desc[i] if i < len(train_desc) else "N/A"
                                p_band = pred_desc[i] if i < len(pred_desc) else "N/A"
                                match_icon = "✅" if t_band == p_band else "❌"
                                if t_band != p_band:
                                    all_match = False
                                md_table += f"| {i+1} | {t_band} | {p_band} | {match_icon} |\n"
                            
                            st.markdown(md_table)
                            
                            # Resolution Info
                            t_res = src_train.res
                            p_res = src_pred.res
                            res_match = all(np.isclose(t_res[i], p_res[i]) for i in range(2))
                            
                            st.write(f"**Resolution Info:**")
                            st.write(f"- Training Resolution: `{t_res[0]:.4f} x {t_res[1]:.4f}`")
                            st.write(f"- Prediction Resolution: `{p_res[0]:.4f} x {p_res[1]:.4f}`")
                            
                            if not res_match:
                                st.warning("⚠️ Warning: Pixel resolutions do not match! This can lead to poor model performance.")
                                all_match = False
                            
                            if all_match:
                                st.success("Success: Band orders match perfectly!")
                            else:
                                st.error("Warning: Band orders do NOT match!")
                    except Exception as e:
                        st.error(f"Error checking band order: {e}")

        # Visualization UI
        st.markdown("---")
        st.subheader("Visualization")
        
        vis_bg_bands = st.multiselect("Select Ground Map Bands (Exactly 1 or 3)", options=pred_inventory, key="vis_bg")
        vis_actual_mask = st.selectbox("Select Actual Target Mask (Optional)", options=["None"] + pred_inventory, key="vis_actual")
        
        if st.button("👁️ Show Results", use_container_width=True):
            if len(vis_bg_bands) not in [1, 3]:
                st.error("Only exactly 1 or 3 bands are allowed for the ground map.")
            else:
                pred_out_path = Path("data/prediction.tif")
                if not pred_out_path.exists():
                    st.error("Prediction result (data/prediction.tif) not found. Please run prediction first.")
                else:
                    try:
                        # Load and downsample
                        max_dim = 512
                        f_name, b_num = vis_bg_bands[0].split(" - Band ")
                        ref_meta = pred_metadata[f_name]
                        ref_f = ref_meta["file_obj"]
                        
                        with MemoryFile(ref_f.getvalue()) as memfile:
                            with memfile.open() as src:
                                scale = min(max_dim / src.width, max_dim / src.height)
                                out_shape = (int(src.height * scale), int(src.width * scale))
                                out_shape = (max(1, out_shape[0]), max(1, out_shape[1]))
                                
                        bg_arrays = []
                        for fb in vis_bg_bands:
                            f_name, b_num = fb.split(" - Band ")
                            b_num = int(b_num)
                            f = pred_metadata[f_name]["file_obj"]
                            with MemoryFile(f.getvalue()) as memfile:
                                with memfile.open() as src:
                                    bg_arrays.append(src.read(b_num, out_shape=out_shape).astype(np.float32))
                                    
                        actual_mask_array = None
                        if vis_actual_mask != "None":
                            f_name, b_num = vis_actual_mask.split(" - Band ")
                            b_num = int(b_num)
                            f = pred_metadata[f_name]["file_obj"]
                            with MemoryFile(f.getvalue()) as memfile:
                                with memfile.open() as src:
                                    actual_mask_array = src.read(b_num, out_shape=out_shape).astype(np.float32)
                                    
                        with rasterio.open(pred_out_path) as src:
                            pred_mask_array = src.read(1, out_shape=out_shape).astype(np.float32)
                            
                        plot_prediction_results(bg_arrays, actual_mask_array, pred_mask_array)
                        
                        # --- Evaluation Metrics ---
                        if actual_mask_array is not None:
                            st.markdown("### Quantitative Evaluation")
                            
                            mode = config.get('data', {}).get('mask_type', 'binary')
                            metrics = calculate_metrics_from_arrays(pred_mask_array, actual_mask_array, mode)
                            
                            if mode == 'regression':
                                col1, col2 = st.columns(2)
                                if metrics.get('MSE') is not None:
                                    col1.metric("Mean Squared Error (MSE)", f"{metrics['MSE']:.4f}")
                                    col2.metric("Root MSE (RMSE)", f"{metrics['RMSE']:.4f}")
                                else:
                                    st.warning("No valid pixels found for regression evaluation.")
                            else:
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
                        
                    except Exception as e:
                        st.error(f"Error loading visualization data: {e}")

if __name__ == "__main__":
    main()