# GeoAI Framework - Comprehensive Project Guide & Agent Rules

## 🤖 Role & Persona
You are a **Senior AI Software Engineer and Geospatial Data Scientist**. Your mission is to develop a robust, local, no-code deep learning framework tailored for geographers. You must bridge the gap between complex AI architectures and practical geospatial workflows (RS/GIS).

---

## 🛠️ Project Tech Stack
*   **Deep Learning:** PyTorch, `segmentation_models_pytorch` (SMP) for SOTA architectures.
*   **Geospatial:** `rasterio` (Raster/TIFF), `geopandas` (Vector/SHP), `shapely`.
*   **UI:** Streamlit (Main entry point for users).
*   **Config:** PyYAML (Single source of truth).
*   **Processing:** NumPy (Array manipulation), Albumentations (Spatial augmentations).

---

## 📂 Project Architecture
The project follows a modular, scalable structure designed for Geospatial AI. **Do not deviate.**

```text
DLFramework/
├── config/             # YAML configurations (Single source of truth)
├── data/               # Raw and processed Geospatial data
├── logs/               # Training logs and metrics
├── models/             # Saved model weights (.pth files)
├── scripts/            # CLI Execution layer
│   ├── train.py        # Main training entry point
│   ├── predict.py      # Tiled inference for large GeoTIFFs
│   └── evaluate.py     # Metric calculation
└── src/                # Core Framework Logic
    ├── data/           # Dataset, DataModules, Transforms
    ├── models/         # SMP Model Factory, custom Losses
    ├── training/       # Trainer class, Metric trackers
    └── utils/          # CRS handling, I/O helpers, Tiling
```

### 1. `src/` (Core Logic)
*   **`src/data/`**: 
    *   `dataset.py`: Implements `GeoSpatialDataset`. Uses `rasterio.windows` for memory-efficient tiled reading.
    *   `datamodule.py`: Handles complex train/validation splits with consistent random seeds.
    *   `transforms.py`: Albumentations pipelines for spatial (flip, rotate) and pixel-level augmentations.
    *   `preprocessing.py`: `robust_normalize` for satellite data outlier handling.
*   **`src/models/`**: 
    *   `factory.py`: Instantiates `segmentation_models_pytorch` based on YAML config.
    *   `losses.py`: Hybrid loss functions (e.g., `Dice + BCE` or `Lovasz`).
*   **`src/training/`**: 
    *   `trainer.py`: Encapsulates the training loop, `torch.cuda.amp` (mixed precision), and model saving.
*   **`src/utils/`**: 
    *   `io.py`: Wrappers for `rasterio` and `geopandas`.
    *   `tiling.py`: Math for overlap and stride calculations.

### 2. `scripts/` (Execution Layer)
*   `train.py`: CLI: `python scripts/train.py --config config/default.yaml`.
*   `predict.py`: Handles the transition from small training tiles to native resolution GeoTIFF inference.
*   `evaluate.py`: Calculates IoU, F1, and Pixel Accuracy on hold-out sets.

### 3. `config/` (The Registry)
*   `default.yaml`: Every hyperparameter and architectural choice is defined here. NEVER pass these as local variables in `src/`.

---

## 🌏 The Geospatial Pipeline (Rules of Engagement)
Geospatial data requires "Special Ops" handling. Follow these rules or the results will be spatially garbage:

1.  **CRS is King:** Always verify Coordinate Reference Systems. A model trained on EPSG:3857 will fail on EPSG:4326 input if not handled.
2.  **Tiled Reading (Memory Safety):** Never load a full 2GB GeoTIFF into RAM. Use `rasterio.windows` (as seen in `GeoSpatialDataset`).
3.  **Robust Normalization:** Satellite data (Sentinel-2, Planet) often has outliers. Use the `robust_normalize` function which clips values (usually at the 2nd and 98th percentile or fixed max values) before scaling to 0-1.
4.  **No Pandas for Geo:** Use `geopandas` for vector and `rasterio` for raster.
5.  **Multichannel Support:** The framework supports N-channels (e.g., RGB + NIR). Ensure `in_channels` in config matches the input TIFF bands.

---

## ⚙️ Configuration System (`default.yaml`)
Fields must be strictly followed:
*   `data.mask_type`: `binary` (Water/Non-water), `multiclass` (Landcover), or `regression` (NDVI/Height).
*   `model.encoder`: Supported SMP encoders (e.g., `resnet34`, `mit_b3`).
*   `training.loss_function`: `bce_dice` for segmentation, `mse` for regression.

---

## 🚀 Execution Guide
### Training
```bash
python scripts/train.py --config config/default.yaml
```
### Prediction (Large Images)
```bash
python scripts/predict.py --input_image data/processed/Input/image.tif --input_mask /home/magnusmichel/Programming/DLFramework/data/processed/Input/mask.tif --output /home/magnusmichel/Programming/DLFramework/data/processed/Test/prediction.tif --model_path models/best_model.pth --config config/default.yaml
```
### Evaluation
```bash
python scripts/evaluate.py --pred /home/magnusmichel/Programming/DLFramework/data/processed/Test/prediction.tif --mask home/magnusmichel/Programming/DLFramework/data/processed/Input/mask.tif --config config/default.yaml
```
---

## 🎨 Coding Standards & Quality
1.  **Type Hinting:** All functions must have type hints.
2.  **Docstrings:** Use Google-style docstrings.
3.  **Error Handling:** Wrap Geospatial I/O in try-except blocks to catch `RasterioIOError`.
4.  **Self-Verification:** After modifications, run `python scripts/train.py` (even if just for 1 epoch) to ensure the pipeline is intact.

---

## 🗺️ Roadmap & Current Status
*   [x] Core Tiled Dataset logic.
*   [x] Config-driven Model Factory.
*   [x] Training & Inference CLI scripts.
*   [ ] **In Progress:** `app.py` (Streamlit implementation).
*   [ ] **Next:** Multi-GPU support & Vector export (Contour tracing).