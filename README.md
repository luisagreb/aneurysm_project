# CellVision: OIR Analysis Platform

## Tool Description
**CellVision is a specialized web-based application designed for the segmentation, visualization, and classification of 3D cellular microscopy data.** Users upload raw OIR files, which are processed using an automated deep learning pipeline (nnU-Net) to segment key organelles: the Nucleus, Actin cytoskeleton, and Mitochondria. The platform provides interactive 3D visualizations and extracts quantitative morphological features—such as volume, surface area, and sphericity. These features are analyzed by a trained machine learning model to predict the cell's condition (Healthy vs. Diseased/TAA) with a calculated probability score, offering researchers an integrated workflow from raw images to diagnostic insights.

## Key Features
- **Automated Segmentation**: Uses nnU-Net for precise 3D segmentation of Nucleus, Actin, and Mitochondria.
- **Interactive Visualization**: Explore 2D slices and 3D reconstructed models directly in the browser.
- **Quantitative Metrics**: Automatically extracts Volume (µm³), Sphericity, surface Area (µm²), and more.
- **Integrated ML Classification**: Predicts disease state (Healthy/TAA) using a Random Forest model trained on morphological features.
- **Data Export**: Download processed features (CSV) and segmentation masks (NIfTI/ZIP) for further analysis.