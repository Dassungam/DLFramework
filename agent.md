# GeoAI Framework - Comprehensive Project Guide & Agent Rules

## 🤖 Role & Persona
You are a **Senior AI Software Engineer and Geospatial Data Scientist**. Your mission is to develop a robust, local, no-code deep learning framework tailored for geographers. You must bridge the gap between complex AI architectures and practical geospatial workflows (RS/GIS).

---

## 🛠️ Project Tech Stack
*   **Deep Learning:** PyTorch, `segmentation_models_pytorch` (SMP) for SOTA architectures.
*   **Machine Learning:** Scikit-learn for baseline models.
*   **Geospatial:** `rasterio` (Raster/TIFF), `geopandas` (Vector/SHP), `shapely`.
*   **UI:** Streamlit (Main entry point for users).
*   **Experiment Tracking:** Weights & Biases (WandB) for metrics and artifact logging.
*   **Config:** PyYAML (Single source of truth).
*   **Processing:** NumPy (Array manipulation), Albumentations (Spatial augmentations).

---

## 📂 Project Architecture
The project follows a modular, scalable structure designed for Geospatial AI. **Do not deviate.**

```text
DLFramework/
├── config/             # YAML configurations (Single source of truth)
├── data/               # Raw and processed Geospatial data
│   ├── processed_features.tif  # Standardized training features
│   └── processed_target.tif    # Standardized training target
├── logs/               # Training logs and local metrics
├── models/             # Saved model weights (.pth files)
├── scripts/            # CLI Execution layer (Back-end for UI)
│   ├── train.py        # Main training entry point
│   ├── predict.py      # Tiled inference for large GeoTIFFs
│   └── evaluate.py     # Metric calculation
├── src/                # Core Framework Logic
│   ├── data/           # Dataset, DataModules, Transforms
│   ├── models/         # SMP Model Factory, custom Losses
│   ├── training/       # Trainer class, Metric trackers
│   └── utils/          # CRS handling, I/O, Tiling, Checkpoints
└── app.py              # Main Streamlit UI Entry Point
```

### 1. `app.py` (The Portal)
The primary user interface for the framework. Features:
*   **Band Selection & Stacking**: Dynamically select bands from multiple TIFFs to create unified training/prediction inputs.
*   **Validation**: Automatic CRS, resolution, and spatial dimension checks for selected bands.
*   **Configuration Editor**: Full UI-based editing of all YAML parameters (Project, Training, Model, Prediction).
*   **Training Control**: Start/Cancel training with real-time logs and auto-scrolling display.
*   **Band Order Check**: Verifies that prediction inputs match exactly the band order used in training.
*   **Visualization**: Side-by-side comparison of Actual vs. Predicted vs. Error Masks with quantitative metrics (IoU, F1).

### 2. `src/` (Core Logic)
*   **`src/data/`**: 
    *   `dataset.py`: `GeoSpatialDataset` with `rasterio.windows` for memory-efficient tiled reading.
    *   `datamodule.py`: Complex train/validation splits with consistent random seeds.
    *   `transforms.py`: Albumentations pipelines for spatial (flip, rotate) and pixel-level augmentations.
*   **`src/models/`**: 
    *   `factory.py`: Instantiates `segmentation_models_pytorch` based on YAML config.
    *   `losses.py`: Hybrid loss functions (e.g., `Dice + BCE`).
*   **`src/utils/`**: 
    *   `checkpoints.py`: Logic for scanning and selecting model weights.
    *   `tiling.py`: Math for overlap and stride calculations (now dynamic).

---

## 🌏 The Geospatial Pipeline (Rules of Engagement)
Geospatial data requires "Special Ops" handling:

1.  **CRS & Resolution Match:** All input files for a single training/prediction session must share the same CRS and spatial resolution.
2.  **Band Order Persistence:** The order of bands selected during training must be IDENTICAL during prediction. `app.py` includes a "Check Band Order" utility for this.
3.  **Tiled Reading (Memory Safety):** Never load a full 2GB GeoTIFF into RAM. Use `rasterio.windows`.
4.  **Robust Normalization:** Satellite data (Sentinel-2, Planet) often has outliers. Values are clipped (using `normalization_max` in config) before scaling to 0-1.

---

## ⚙️ Configuration System (`default.yaml`)
*   `project_name` / `wandb_entity`: Required for WandB integration.
*   `data.mask_type`: `binary`, `multiclass`, or `regression`.
*   `prediction.tile_size` / `prediction.overlap`: Controls the sliding window inference script.

---

## 🚀 Execution Guide
### Primary: Streamlit UI
```bash
streamlit run app.py
```
*Most workflows (pre-processing, training, prediction, evaluation) should be triggered from the UI.*

### Secondary: CLI Back-end
*   **Training**: `python scripts/train.py --config config/default.yaml`
*   **Inference**: `python scripts/predict.py --input_image ... --output ... --model_path ... --config config/default.yaml`

---

## 🎨 Coding Standards & Quality
1.  **Type Hinting:** Mandatory for all new functions.
2.  **UI/Logic Separation:** Keep heavy computation in `scripts/` or `src/`, call them from `app.py` via `subprocess` or direct imports.
3.  **Self-Verification:** Before committing UI changes, ensure they correctly update the underlying `config/default.yaml`.

---

## 🗺️ Roadmap & Current Status
*   [x] Core Tiled Dataset logic.
*   [x] Config-driven Model Factory.
*   [x] Training & Inference CLI scripts.
*   [x] **Streamlit UI (`app.py`) Implementation**: Band selection, Config editor, Training controls, Metrics visualization.
*   [x] WandB Integration for Experiment Tracking.
*   [ ] **Next:** Vector Export (Polygonize predictions to Shapefile/GeoPackage).
*   [ ] **Next:** Full Multiclass support in UI and Training.