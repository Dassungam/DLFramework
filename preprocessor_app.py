"""
preprocessor_app.py
-------------------
Standalone Streamlit app for geospatial data pre-processing.

Tab 1 – Ingest & Standardize : Upload raw rasters / vectors → parser.py
Tab 2 – Intersect & Clip     : Visualise individual + customise intersection bboxes on a Folium
                               map (temporarily projected to EPSG:4326) → clip everything.
Tab 3 – Resample & Merge     : Unify spatial resolutions and stack into features.tif
Tab 4 – Rasterize Labels     : Burn vectors into masks.tif and visualize the output
"""

import shutil
import tempfile
from pathlib import Path

import folium  # pip install folium streamlit-folium
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import streamlit as st
from pyproj import Transformer
from streamlit_folium import st_folium

# ── project imports ───────────────────────────────────────────────────────────
import sys
sys.path.append(str(Path(__file__).resolve().parent))

from src.data.clipper import clip_datasets, get_intersection_bounds
from src.data.parser import process_datasets
from src.data.transformer import unify_and_merge_rasters, rasterize_vectors

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GeoAI Preprocessor",
    page_icon="🌍",
    layout="wide",
)

# ── styling ───────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main { background: #0f1117; color: #f0f2f6; }
    .block-container { padding-top: 2rem; }

    /* Tab pill style */
    button[data-baseweb="tab"] {
        font-weight: 600;
        font-size: 0.95rem;
        padding: 0.6rem 1.4rem;
        border-radius: 8px 8px 0 0;
    }

    /* Cards */
    .info-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
    }

    /* Status badges */
    .badge-success {
        background: #14532d; color: #4ade80;
        border-radius: 20px; padding: 2px 10px;
        font-size: 0.78rem; font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── session state ─────────────────────────────────────────────────────────────
if "standardized_files" not in st.session_state:
    st.session_state.standardized_files: list[str] = []
if "standardize_tmp_dir" not in st.session_state:
    st.session_state.standardize_tmp_dir: str | None = None

if "clipped_files" not in st.session_state:
    st.session_state.clipped_files: list[str] = []
if "merged_features_path" not in st.session_state:
    st.session_state.merged_features_path: str | None = None
if "masks_path" not in st.session_state:
    st.session_state.masks_path: str | None = None

# ── helper function for reprojection ─────────────────────────────────────────
def transform_bbox(bbox, transformer):
    """Transforms a bounding box (min_x, min_y, max_x, max_y) using a pyproj Transformer."""
    min_x, min_y, max_x, max_y = bbox
    # Transform all 4 corners to account for curve distortion
    xs = [min_x, max_x, max_x, min_x]
    ys = [min_y, min_y, max_y, max_y]
    tx, ty = transformer.transform(xs, ys)
    return (min(tx), min(ty), max(tx), max(ty))

# ── helper function for visualization ────────────────────────────────────────
def get_classification_colors(n):
    """Returns a list of n distinct colors using the tab10 or tab20 colormap."""
    if n <= 10:
        return [plt.cm.tab10(i) for i in range(n)]
    else:
        return [plt.cm.tab20(i % 20) for i in range(n)]

def plot_data_preview(input_file, mask_file, img_bands: list, mask_band: int, show_mask: bool = True, alpha: float = 0.5):
    """Generate a downsampled side-by-side preview of custom bands from the image and mask."""
    try:
        with rasterio.open(input_file) as src_in, rasterio.open(mask_file) as src_mask:
            # Calculate downsampled shape (e.g., max 512 pixels on longest side to save memory)
            max_dim = 512
            scale = min(max_dim / src_in.width, max_dim / src_in.height)
            out_shape = (int(src_in.height * scale), int(src_in.width * scale))
            out_shape = (max(1, out_shape[0]), max(1, out_shape[1]))
            
            is_rgb = len(img_bands) == 3
            
            # Extract requested image bands (rasterio bands are 1-indexed)
            img_data = src_in.read(img_bands, out_shape=out_shape).astype(np.float32)
            if is_rgb:
                img_data = np.transpose(img_data, (1, 2, 0)) # Transpose to (H, W, C)
            else:
                img_data = img_data[0] # Squeeze to (H, W)
                
            # Extract requested mask band
            mask_data = src_mask.read(mask_band, out_shape=out_shape).astype(np.float32)
            
        # Normalize image data for visualization
        if is_rgb:
            for i in range(3):
                ch = img_data[:, :, i]
                if ch.max() > ch.min():
                    img_data[:, :, i] = (ch - ch.min()) / (ch.max() - ch.min())
        else:
            if img_data.max() > img_data.min():
                img_data = (img_data - img_data.min()) / (img_data.max() - img_data.min())
                
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Plot the input image
        if is_rgb:
            ax.imshow(np.clip(img_data, 0, 1))
        else:
            ax.imshow(img_data, cmap='gray')
            
        ax.set_title("Input Features with Mask Overlay")
        ax.axis('off')
        
        if show_mask:
            # Mask out 0/background values of the target mask so they are transparent
            present_classes = sorted([int(v) for v in np.unique(mask_data) if not np.isnan(v)])
            
            # Use qualitative colors for discrete classes
            colors = get_classification_colors(len(present_classes))
            rgba_colors = [list(c[:3]) + [alpha] for c in colors]
            
            color_mapped_mask = np.zeros((*mask_data.shape, 4))
            for i, val in enumerate(present_classes):
                if val == 0: continue # Keep background transparent
                color_mapped_mask[mask_data == val] = rgba_colors[i]
                
            ax.imshow(color_mapped_mask, interpolation='none')
            
            # Add a small legend for the classes present
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor=colors[i], label=f"Class {val}", alpha=alpha) for i, val in enumerate(present_classes) if val != 0]
            if legend_elements:
                ax.legend(handles=legend_elements, title="Detected Classes", loc='center left', bbox_to_anchor=(1, 0.5))
        
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error generating preview: {e}")

# ── header ────────────────────────────────────────────────────────────────────
st.markdown("## 🌍 GeoAI Preprocessor")
st.markdown(
    "Standardize geospatial datasets to a common CRS, inspect their "
    "spatial intersection, optionally customise the bounding box, and clip. Finally, "
    "resample, merge, and rasterize your labels."
)
st.divider()

tab1, tab2, tab3, tab4 = st.tabs([
    "**1 · Ingest & Standardize**", 
    "**2 · Intersect & Clip**",
    "**3 · Resample & Merge**",
    "**4 · Rasterize Labels & View**"
])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 – Ingest & Standardize
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Upload Raw Files")
    st.markdown(
        "Accepted formats: **`.tif`**, **`.tiff`**, **`.jp2`** (rasters) "
        "and **`.shp`**, **`.geojson`**, **`.gpkg`** (vectors)."
    )

    uploaded = st.file_uploader(
        "Drop files here or click to browse",
        type=["tif", "tiff", "jp2", "shp", "geojson", "gpkg", "dbf", "prj", "shx"],
        accept_multiple_files=True,
        key="uploader",
    )

    if uploaded:
        st.markdown(f"**{len(uploaded)} file(s) selected.**")

        if st.button("▶ Standardize All Files", type="primary"):
            tmp_dir = Path(tempfile.mkdtemp(prefix="geoai_raw_"))
            for uf in uploaded:
                dest = tmp_dir / uf.name
                dest.write_bytes(uf.getvalue())

            RASTER_EXT = {".tif", ".tiff", ".jp2"}
            VECTOR_EXT = {".shp", ".geojson", ".gpkg"}
            primary_files = [
                str(p)
                for p in tmp_dir.iterdir()
                if p.suffix.lower() in RASTER_EXT | VECTOR_EXT
            ]

            if not primary_files:
                st.error("No recognised primary geospatial files found.")
            else:
                out_dir = tmp_dir / "standardized"
                with st.spinner("Standardizing …"):
                    try:
                        result = process_datasets(primary_files, str(out_dir))
                        st.session_state.standardized_files = [str(p) for p in result]
                        st.session_state.standardize_tmp_dir = str(tmp_dir)
                        st.success(
                            f"✅ {len(result)} file(s) standardized dynamically."
                        )
                    except Exception as exc:
                        st.error(f"Standardization failed: {exc}")

    if st.session_state.standardized_files:
        st.divider()
        st.markdown("#### Ready for Tab 2")
        for fp in st.session_state.standardized_files:
            p = Path(fp)
            icon = "🗺️" if p.suffix == ".tif" else "📦"
            st.markdown(
                f'<div class="info-card">{icon} <code>{p.name}</code>'
                f' &emsp; <span class="badge-success">standardized</span></div>',
                unsafe_allow_html=True,
            )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 – Intersect & Clip
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    files = st.session_state.standardized_files

    if not files:
        st.warning("⚠️ No standardized files yet. Complete Tab 1 first.")
        st.stop()

    native_crs = "EPSG:4326"
    for fp in files:
        if fp.endswith((".tif", ".tiff")):
            with rasterio.open(fp) as src:
                native_crs = src.crs
                break
        elif fp.endswith((".gpkg", ".shp", ".geojson")):
            gdf = gpd.read_file(fp)
            native_crs = gdf.crs
            break
            
    to_4326 = Transformer.from_crs(native_crs, "EPSG:4326", always_xy=True)
    from_4326 = Transformer.from_crs("EPSG:4326", native_crs, always_xy=True)

    individual_bboxes_native: list[tuple] = []
    for fp in files:
        p = Path(fp)
        try:
            if p.suffix in [".tif", ".tiff"]:
                with rasterio.open(fp) as src:
                    b = src.bounds
                    individual_bboxes_native.append((b.left, b.bottom, b.right, b.top, p.name))
            else:
                gdf = gpd.read_file(fp)
                b = gdf.total_bounds
                individual_bboxes_native.append((b[0], b[1], b[2], b[3], p.name))
        except Exception:
            pass

    try:
        if len(files) >= 2:
            intersection_native = get_intersection_bounds(files)
        else:
            intersection_native = individual_bboxes_native[0][:4]
        
        inter_4326 = transform_bbox(intersection_native, to_4326)
        has_intersection = True
    except ValueError as exc:
        st.error(f"❌ No spatial overlap natively: {exc}")
        inter_4326 = (0.0, 0.0, 0.0, 0.0)
        has_intersection = False

    st.subheader("Customise Clipping Bounding Box (Lat/Lon)")
    st.markdown(f"Native CRS detected as **{native_crs}**. Coordinates below are temporarily projected to **EPSG:4326** for visualisation.")
    
    col1, col2, col3, col4 = st.columns(4)
    custom_min_x = col1.number_input("Min Longitude (West)", value=inter_4326[0], format="%.6f", step=0.01)
    custom_min_y = col2.number_input("Min Latitude (South)", value=inter_4326[1], format="%.6f", step=0.01)
    custom_max_x = col3.number_input("Max Longitude (East)", value=inter_4326[2], format="%.6f", step=0.01)
    custom_max_y = col4.number_input("Max Latitude (North)", value=inter_4326[3], format="%.6f", step=0.01)

    custom_bbox_4326 = (custom_min_x, custom_min_y, custom_max_x, custom_max_y)

    if individual_bboxes_native:
        centre_lat = (custom_min_y + custom_max_y) / 2
        centre_lon = (custom_min_x + custom_max_x) / 2

        if centre_lat == 0 and centre_lon == 0:
            b4326 = transform_bbox(individual_bboxes_native[0][:4], to_4326)
            centre_lat = (b4326[1] + b4326[3]) / 2
            centre_lon = (b4326[0] + b4326[2]) / 2

        m = folium.Map(
            location=[centre_lat, centre_lon],
            zoom_start=6,
            tiles="CartoDB dark_matter",
        )

        for min_x, min_y, max_x, max_y, name in individual_bboxes_native:
            b4326 = transform_bbox((min_x, min_y, max_x, max_y), to_4326)
            b_min_x, b_min_y, b_max_x, b_max_y = b4326
            poly_coords = [
                [b_min_y, b_min_x],
                [b_max_y, b_min_x],
                [b_max_y, b_max_x],
                [b_min_y, b_max_x],
                [b_min_y, b_min_x],
            ]
            folium.Polygon(
                locations=poly_coords,
                color="#3b82f6",
                weight=2,
                fill=True,
                fill_color="#3b82f6",
                fill_opacity=0.10,
                tooltip=f"📄 {name}",
            ).add_to(m)

        custom_coords = [
            [custom_min_y, custom_min_x],
            [custom_max_y, custom_min_x],
            [custom_max_y, custom_max_x],
            [custom_min_y, custom_max_x],
            [custom_min_y, custom_min_x],
        ]
        folium.Polygon(
            locations=custom_coords,
            color="#ef4444",
            weight=4,
            fill=True,
            fill_color="#ef4444",
            fill_opacity=0.18,
            tooltip="🔴 Active Clipping Box",
        ).add_to(m)

        st_folium(m, width="100%", height=520, returned_objects=[])

    st.divider()
    OUT_DIR = "data/clipped"

    if st.button("✂️ Confirm & Clip All Files", type="primary"):
        with st.spinner(f"Transforming bounding box to native CRS and clipping → `{OUT_DIR}/` …"):
            try:
                custom_bbox_native = transform_bbox(custom_bbox_4326, from_4326)
                
                clipped = clip_datasets(
                    files, custom_bbox_native, OUT_DIR
                )
                st.session_state.clipped_files = [str(p) for p in clipped]
                # Reset downstream steps
                st.session_state.merged_features_path = None
                st.session_state.masks_path = None
                
                st.success(
                    f"✅ {len(clipped)} file(s) clipped and saved to `{OUT_DIR}/`."
                )
                for cp in clipped:
                    p = Path(cp)
                    icon = "🗺️" if p.suffix == ".tif" else "📦"
                    st.markdown(
                        f'<div class="info-card">{icon} <code>{p.name}</code>'
                        f' &emsp; <span class="badge-success">clipped</span></div>',
                        unsafe_allow_html=True,
                    )
            except Exception as exc:
                st.error(f"Clipping failed: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 – Resample & Merge
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    c_files = st.session_state.clipped_files
    
    if not c_files:
        st.warning("⚠️ No clipped files yet. Complete Tab 2 first.")
        st.stop()
        
    clipped_rasters = [fp for fp in c_files if fp.endswith((".tif", ".tiff"))]
    
    if not clipped_rasters:
        st.warning("⚠️ No clipped rasters found. Cannot merge features.")
        st.stop()
        
    st.subheader("Raster Resolutions")
    st.markdown("Select a target resolution for all features. The system will resample and stack them into a single file.")
    
    resolutions = []
    for fp in clipped_rasters:
        with rasterio.open(fp) as src:
            res = src.res[0]
            resolutions.append((Path(fp).name, res))
            
    for name, res in resolutions:
        st.markdown(f"**{name}**: `{res:.6f} units/px`")
        
    best_res = min(r[1] for r in resolutions)
    worst_res = max(r[1] for r in resolutions)
    
    st.divider()
    choice = st.radio(
        "Target Resolution Strategy",
        options=["Best (Finest)", "Worst (Coarsest)", "Custom"],
        captions=[f"{best_res:.6f}", f"{worst_res:.6f}", "Enter manually"]
    )
    
    target_res = best_res
    if choice == "Worst (Coarsest)":
        target_res = worst_res
    elif choice == "Custom":
        target_res = st.number_input("Custom Resolution", value=best_res, min_value=0.000001, format="%.6f")
        
    if st.button("🚀 Resample & Merge Rasters", type="primary"):
        OUT_PATH = "data/processed/features.tif"
        with st.spinner(f"Resampling to {target_res:.6f} and merging into {OUT_PATH} ..."):
            try:
                merged_path = unify_and_merge_rasters(clipped_rasters, target_res, OUT_PATH)
                st.session_state.merged_features_path = str(merged_path)
                st.session_state.masks_path = None # Reset mask path since features changed
                st.success(f"✅ Successfully created merged features: `{merged_path}`")
                
                with rasterio.open(merged_path) as src:
                    st.markdown(f"**Final Shape:** `{src.width}x{src.height}` | **Bands:** `{src.count}`")
                    st.code("Band Descriptions:\n" + "\n".join([f"{i}: {d}" for i, d in enumerate(src.descriptions, 1)]))
                    
                st.info("Switch to **Tab 4** to rasterize label vectors and view results.")
            except Exception as exc:
                st.error(f"Merge failed: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 – Rasterize Labels & View
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    c_files = st.session_state.clipped_files
    merged_features = st.session_state.merged_features_path
    
    if not merged_features:
        st.warning("⚠️ No merged `features.tif` yet. Complete Tab 3 first.")
        st.stop()
        
    clipped_vectors = [fp for fp in c_files if fp.endswith(".gpkg")]
    
    if not clipped_vectors:
        st.warning("⚠️ No clipped vector files (.gpkg) found. Nothing to rasterize.")
        with open(merged_features, "rb") as f:
            st.download_button("⬇️ Download features.tif", f, file_name="features.tif", mime="image/tiff")
        st.stop()
        
    st.subheader("Configure Vector Classes")
    st.markdown("Each vector file will be burned as an individual band into a multi-band `masks.tif`.")
    
    vector_configs = []
    
    for v_path in clipped_vectors:
        name = Path(v_path).name
        with st.expander(f"📦 Configure: {name}", expanded=True):
            gdf = gpd.read_file(v_path)
            cols = [c for c in gdf.columns if c != "geometry"]
            
            p_col1, p_col2 = st.columns(2)
            
            if not cols:
                st.info("No data columns found. Forcing binary mask.")
                target_col = None
                is_binary = True
            else:
                target_col = p_col1.selectbox(f"Value Column for {name}", options=cols, key=f"col_{name}")
                is_binary = p_col2.checkbox("Force Binary Mask (1 for targets, 0 for background)", value=False, key=f"bin_{name}")
                
            vector_configs.append({
                "target_column": target_col,
                "is_binary": is_binary
            })
            
    st.divider()
    OUT_MASK_PATH = "data/processed/masks.tif"
    OUT_DICT_PATH = "data/processed/class_mapping.json"
    
    if st.button("🔥 Rasterize Labels", type="primary"):
        with st.spinner("Rasterizing vectors..."):
            try:
                Path(OUT_MASK_PATH).parent.mkdir(parents=True, exist_ok=True)
                rasterize_vectors(
                    clipped_vectors,
                    merged_features,
                    vector_configs,
                    OUT_MASK_PATH,
                    OUT_DICT_PATH
                )
                # Store the successful generation in session state so the UI persists
                st.session_state.masks_path = OUT_MASK_PATH
                st.success(f"✅ Created `{OUT_MASK_PATH}` and `{OUT_DICT_PATH}`")
                
            except Exception as exc:
                st.error(f"Rasterization failed: {exc}")
                
    # ── Visulisation and Download Block (Shows up after rasterization) ────────
    if st.session_state.masks_path and Path(st.session_state.masks_path).exists():
        st.markdown("### 💾 Download Processed Datasets")
        c1, c2, c3 = st.columns(3)
        with open(merged_features, "rb") as f:
            c1.download_button("⬇️ feature.tif", f, file_name="features.tif", mime="image/tiff", use_container_width=True)
        with open(st.session_state.masks_path, "rb") as f:
            c2.download_button("⬇️ masks.tif", f, file_name="masks.tif", mime="image/tiff", use_container_width=True)
        with open(OUT_DICT_PATH, "rb") as f:
            c3.download_button("⬇️ mapping.json", f, file_name="class_mapping.json", mime="application/json", use_container_width=True)
            
        st.divider()
        st.subheader("👁️ Preview Dataset")
        
        # Read band metadata to populate UI
        with rasterio.open(merged_features) as src_feat, rasterio.open(st.session_state.masks_path) as src_mask:
            feat_bands = [f"Band {i}: {d or 'Unknown'}" for i, d in enumerate(src_feat.descriptions, 1)]
            mask_bands = [f"Band {i}: {d or 'Unknown'}" for i, d in enumerate(src_mask.descriptions, 1)]
            
            feat_band_indices = list(range(1, src_feat.count + 1))
            mask_band_indices = list(range(1, src_mask.count + 1))
            
        v_col1, v_col2 = st.columns([1, 2])
        
        with v_col1:
            st.markdown("#### Display Settings")
            view_mode = st.radio("Image Display Mode", options=["Grayscale (1 Band)", "RGB (3 Bands)"])
            
            img_selection = []
            if view_mode == "Grayscale (1 Band)":
                idx = st.selectbox("Select Image Band", options=feat_band_indices, format_func=lambda x: feat_bands[x-1])
                img_selection = [idx]
            else:
                if len(feat_band_indices) < 3:
                    st.warning("Not enough bands for RGB. Falling back to first available.")
                    img_selection = feat_band_indices + [feat_band_indices[0]] * (3 - len(feat_band_indices))
                else:
                    r = st.selectbox("Red Channel", options=feat_band_indices, index=0, format_func=lambda x: feat_bands[x-1])
                    g = st.selectbox("Green Channel", options=feat_band_indices, index=1, format_func=lambda x: feat_bands[x-1])
                    b = st.selectbox("Blue Channel", options=feat_band_indices, index=2, format_func=lambda x: feat_bands[x-1])
                    img_selection = [r, g, b]
                    
            st.markdown("---")
            mask_selection = st.selectbox("Select Label Mask to Overlay", options=mask_band_indices, format_func=lambda x: mask_bands[x-1])
            show_mask = st.checkbox("Show Mask Overlay", value=True)
            alpha_val = st.slider("Mask Opacity", min_value=0.1, max_value=1.0, value=0.5)

        with v_col2:
            plot_data_preview(
                input_file=merged_features, 
                mask_file=st.session_state.masks_path, 
                img_bands=img_selection, 
                mask_band=mask_selection, 
                show_mask=show_mask, 
                alpha=alpha_val
            )