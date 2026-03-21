import streamlit as st
import yaml
import subprocess
from pathlib import Path
import rasterio
from rasterio.io import MemoryFile
import matplotlib.pyplot as plt
import numpy as np
import wandb


def apply_percentile_stretch(stacked_array, lower_percentile=2, upper_percentile=98):
    """
    Applies a percentile stretch to a stacked numpy array band-by-band 
    and normalizes the output to a 0.0 - 1.0 range.
    Expected input shape: (Bands, Height, Width)
    """
    stretched_bands = []
    
    for band in stacked_array:
        # Calculate percentiles, ignoring NaNs if they exist
        vmin, vmax = np.nanpercentile(band, (lower_percentile, upper_percentile))
        
        # Clip the extreme outliers
        clipped = np.clip(band, vmin, vmax)
        
        # Normalize to 0.0 - 1.0
        # Check to avoid division by zero in case of an empty/constant band
        if vmax - vmin > 0:
            normalized = (clipped - vmin) / (vmax - vmin)
        else:
            normalized = clipped
            
        stretched_bands.append(normalized)
        
    # Stack back together and ensure it's a float32 array for ML
    return np.stack(stretched_bands).astype(np.float32)

def calculate_global_stats(stacked_array):
    """
    Calculates global mean and standard deviation for each band in a stacked array.
    Expected input shape: (Bands, Height, Width)
    Returns: (list of means, list of stds)
    """
    means = []
    stds = []
    
    for band in stacked_array:
        # Calculate stats, ignoring NaNs
        means.append(float(np.nanmean(band)))
        stds.append(float(np.nanstd(band)))
        
    return means, stds

def get_classification_colors(n):
    """Returns a list of n distinct colors using the tab10 or tab20 colormap."""
    if n <= 10:
        return [plt.cm.tab10(i) for i in range(n)]
    else:
        return [plt.cm.tab20(i % 20) for i in range(n)]

def plot_data_preview(input_file, mask_file, show_mask: bool = True, task_type='classification', class_names=None, class_map=None):
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
        
        # Regression mask normalization (for preview only)
        if task_type == 'regression':
            if mask_data.max() > mask_data.min():
                mask_data = (mask_data - mask_data.min()) / (mask_data.max() - mask_data.min())
            
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Plot the input image
        if is_rgb:
            ax.imshow(img_data)
        else:
            ax.imshow(img_data, cmap='gray')
            
        ax.set_title("Input Image with Mask Overlay")
        ax.axis('off')
        
        # Overlay the mask
        if show_mask:
            if task_type in ['classification', 'multiclass', 'binary']:
                # Mask out 0/background values of the target mask so they are transparent
                masked_mask = np.ma.masked_where(mask_data == 0, mask_data)
                
                # Get unique values present in mask
                present_classes = sorted([int(v) for v in np.unique(mask_data) if not np.isnan(v)])
                
                # Create a qualitative colormap for the present classes
                from matplotlib.colors import ListedColormap
                colors = get_classification_colors(len(present_classes))
                # Add transparency
                rgba_colors = [list(c[:3]) + [0.5] for c in colors]
                
                # Map mask values to color indices
                color_mapped_mask = np.zeros((*mask_data.shape, 4))
                for i, val in enumerate(present_classes):
                    if val == 0: continue # Skip background transparency
                    color_mapped_mask[mask_data == val] = rgba_colors[i]
                
                ax.imshow(color_mapped_mask, interpolation='none')
                
                # Add Legend
                from matplotlib.patches import Patch
                legend_elements = []
                for i, val in enumerate(present_classes):
                    # Resolve name: try original value first, then mapped index if provided
                    name = None
                    if class_names:
                        name = class_names.get(str(val))
                        if name is None and class_map:
                            idx = class_map.get(str(val)) or class_map.get(val)
                            if idx is not None:
                                name = class_names.get(str(idx))
                    
                    if name is None:
                        name = f"Class {val}"
                        
                    legend_elements.append(Patch(facecolor=colors[i], label=name, alpha=0.5))
                
                if legend_elements:
                    ax.legend(handles=legend_elements, title="Classes", loc='center left', bbox_to_anchor=(1, 0.5))
            else:
                # For regression, show the full range
                im = ax.imshow(mask_data, cmap='magma', alpha=0.5, interpolation='none')
                fig.colorbar(im, ax=ax, label="Value")
        
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error generating preview: {e}")

def plot_prediction_results(bg_arrays, actual_mask_array, pred_mask_array, task_type='classification', class_names=None, class_map=None):
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
            cmap_bg = 'gray'
        else:
            # Ensure 3 bands for RGB
            if bg_norm.shape[-1] >= 3:
                bg_norm = bg_norm[:, :, :3]
            cmap_bg = None
            
        fig, axes = plt.subplots(1, 3, figsize=(18, 7))
        
        # Determine unique classes across both masks
        if task_type in ['classification', 'multiclass', 'binary']:
            all_vals = set(np.unique(pred_mask_array))
            if actual_mask_array is not None:
                all_vals.update(np.unique(actual_mask_array))
            present_classes = sorted([int(v) for v in all_vals if not np.isnan(v)])
            colors = get_classification_colors(len(present_classes))
            rgba_colors = [list(c[:3]) + [0.5] for c in colors]
            
            def map_to_rgba(arr):
                res = np.zeros((*arr.shape, 4))
                for i, val in enumerate(present_classes):
                    if val == 0: continue
                    res[arr == val] = rgba_colors[i]
                return res
        else: # regression
            vmin = min(np.nanmin(pred_mask_array), np.nanmin(actual_mask_array) if actual_mask_array is not None else 0)
            vmax = max(np.nanmax(pred_mask_array), np.nanmax(actual_mask_array) if actual_mask_array is not None else 1)
            cmap_mask = 'magma'

        # Plot 1: Actual
        axes[0].set_title("Actual Target")
        if cmap_bg is None:
            axes[0].imshow(bg_norm)
        else:
            axes[0].imshow(bg_norm, cmap=cmap_bg)
        if actual_mask_array is not None:
            if task_type in ['classification', 'multiclass', 'binary']:
                axes[0].imshow(map_to_rgba(actual_mask_array), interpolation='none')
            else:
                axes[0].imshow(actual_mask_array, cmap=cmap_mask, vmin=vmin, vmax=vmax, alpha=0.5, interpolation='none')
        else:
            axes[0].text(0.5, 0.5, 'No Actual Mask Selected', ha='center', va='center', transform=axes[0].transAxes, color='red', fontsize=12)
        axes[0].axis('off')
        
        # Plot 2: Predicted
        axes[1].set_title("Predicted Target")
        if cmap_bg is None:
            axes[1].imshow(bg_norm)
        else:
            axes[1].imshow(bg_norm, cmap=cmap_bg)
        if task_type in ['classification', 'multiclass', 'binary']:
            im_pred = axes[1].imshow(map_to_rgba(pred_mask_array), interpolation='none')
        else:
            im_pred = axes[1].imshow(pred_mask_array, cmap=cmap_mask, vmin=vmin, vmax=vmax, alpha=0.5, interpolation='none')
        axes[1].axis('off')
        
        # Plot 3: Error Map
        axes[2].set_title("Error Map")
        if cmap_bg is None:
            axes[2].imshow(bg_norm)
        else:
            axes[2].imshow(bg_norm, cmap=cmap_bg)
        
        if actual_mask_array is not None:
            H, W = pred_mask_array.shape
            error_map = np.zeros((H, W, 4), dtype=np.float32)
            
            if task_type in ['classification', 'multiclass', 'binary']:
                # Show where labels differ
                errors = (actual_mask_array != pred_mask_array)
                error_map[errors] = [1.0, 0.0, 0.0, 0.8] # Strong Red for any mismatch
                axes[2].set_title("Error Map (Red = Mismatches)")
            else:
                # Show residual error for regression
                abs_err = np.abs(actual_mask_array - pred_mask_array)
                max_err = np.nanmax(abs_err)
                if max_err > 0:
                    # Scale alpha by error intensity
                    error_map[:, :, 0] = 1.0 # Red
                    error_map[:, :, 3] = (abs_err / max_err) * 0.8 # Scale alpha
                axes[2].set_title(f"Error Map (Residuals, max={max_err:.2f})")
            
            axes[2].imshow(error_map, interpolation='none')
        else:
            axes[2].text(0.5, 0.5, 'No Actual Mask Selected', ha='center', va='center', transform=axes[2].transAxes, color='red', fontsize=12)
        
        axes[2].axis('off')
        
        # Add a shared legend/colorbar
        if task_type in ['classification', 'multiclass', 'binary']:
            from matplotlib.patches import Patch
            legend_elements = []
            for i, val in enumerate(present_classes):
                name = None
                if class_names:
                    name = class_names.get(str(val))
                    if name is None and class_map:
                        idx = class_map.get(str(val)) or class_map.get(val)
                        if idx is not None:
                            name = class_names.get(str(idx))
                
                if name is None:
                    name = f"Class {val}"
                legend_elements.append(Patch(facecolor=colors[i], label=name, alpha=0.5))
            
            if legend_elements:
                fig.legend(handles=legend_elements, title="Classes", loc='lower center', ncol=min(len(legend_elements), 5), bbox_to_anchor=(0.5, -0.05))
        else:
            # Regression Colorbar
            fig.colorbar(im_pred, ax=axes.ravel().tolist(), label="Value", orientation='horizontal', fraction=0.05, pad=0.1)

        plt.tight_layout()
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error generating prediction plots: {e}")

def load_config(config_path: str) -> dict:
    """Read and parse the YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def get_band_inventory(uploaded_files):
    """Retrieve an inventory of bands and their descriptions."""
    inventory = []
    metadata = {}
    for f in uploaded_files:
        try:
            with MemoryFile(f.getvalue()) as memfile:
                with memfile.open() as src:
                    num_bands = src.count
                    descriptions = src.descriptions
                    
                    # Mark data/ folder files as processed (standardized)
                    prefix = "[PROCESSED] " if "data/" in f.name else ""
                    
                    metadata[f.name] = {
                        "crs": src.crs,
                        "width": src.width,
                        "height": src.height,
                        "transform": src.transform,
                        "res": src.res,
                        "profile": src.profile,
                        "file_obj": f
                    }
                    for b in range(1, num_bands + 1):
                        desc = descriptions[b-1]
                        label = f"{prefix}{f.name} - Band {b}"
                        if desc: label += f" ({desc})"
                        inventory.append(label)
        except Exception as e:
            st.error(f"Error reading {f.name}: {e}")
    return inventory, metadata

def parse_band_selection(selection_str):
    """Extracts the file name and band number from the UI selection string."""
    f_name, rest = selection_str.rsplit(" - Band ", 1)
    b_num = int(rest.split(" ")[0])
    return f_name, b_num