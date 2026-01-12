# Segmentation Pipeline Summary

## 1. Methodology: 3D nnU-Net (V2)
We employed **nnU-Net V2**, a state-of-the-art self-configuring framework for medical image segmentation, specifically configured for **3D Full-Resolution** processing (`3d_fullres`). This ensures that the volumetric nature of the Z-stack microscopy data is fully utilized, rather than treating it as independent 2D slices.

## 2. Data Preparation & Preprocessing
To ensure high-quality training, we implemented a robust data preparation pipeline:

*   **Format Conversion**: Raw `.nrrd` microscopy files were converted to NIfTI (`.nii.gz`), the standard format for neuroimaging and deep learning.
*   **Dimensionality Standardization**: A heuristic was implemented to correct inconsistent axis ordering. Any data found in `(Z, Y, X)` orientation was automatically transposed to `(X, Y, Z)` to match the standard expected by nnU-Net.
*   **Data Matching**: A strict 2-stage verification system was used to pair raw images with manual segmentations:
    *   **Stage 1**: Strict name matching (normalized).
    *   **Stage 2**: Fuzzy matching with manual overrides to handle inconsistent naming conventions (e.g., `+Col` vs `col`, `cell1` vs `Cell 1`).
*   **Data Split**:
    *   **Training**: Used properly verified manual segmentations.
    *   **Testing**: All remaining samples (not seen during training) were reserved for inference.

## 3. Quantitative Results
Both models were trained for the full schedule of **1000 epochs** to convergence.

### Dataset 001: Actin
*   **Training Cases**: 47 Manual Segmentations
*   **Validation Set**: 10 randomly held-out cases
*   **Model Performance**:
    *   **Mean Validation Dice Score: 86.8%** (`0.8677`)
    *   This indicates excellent agreement between the automated model and manual expert segmentation.

### Dataset 002: Mitochondria
*   **Training Cases**: 38 Manual Segmentations
*   **Validation Set**: 8 randomly held-out cases
*   **Model Performance**:
    *   **Mean Validation Dice Score: 80.3%** (`0.8025`)
    *   This represents a strong segmentation capability, successfully capturing the complex fragmentation of mitochondrial networks.

### Dataset 003: Nucleus
*   **Status**: Training In Progress
*   **Configuration**: Similar 3D Full-Resolution architecture.

## 4. Inference (Current Status)
We are currently generating predictions for the vast majority of the dataset:
*   **Actin Test Set**: 153 new cases
*   **Mitochondria Test Set**: 162 new cases
*   **Output**: Probability maps and binary segmentation masks are being generated for downstream analysis.
