# Methods

## Deep Learning Segmentation (nnU-Net)

To automate the segmentation of cellular structures (Actin, Mitochondria, and Nuclei), we employed **nnU-Net V2**, a self-configuring deep learning framework that automatically adapts its architecture and hyperparameters to the dataset properties.

### Dataset Preparation
Three separate datasets were curated for training, utilizing manual expert segmentations:
1.  **Dataset 001 (Actin)**: Single-channel fluorescence microscopy images (47 manually segmented 3D cases).
2.  **Dataset 002 (Mitochondria)**: Single-channel mitochondrial stain (38 manually segmented 3D cases).
3.  **Dataset 003 (Nucleus)**: Single-channel nuclear stain (93 manually segmented 3D cases).

Input data were converted from NRRD to NIfTI format. A custom preprocessing pipeline was developed to ensure data integrity:
-   **Orientation Standardization**: Volumes were heuristically checked and transposed to ensure consistent (X, Y, Z) orientation, correcting instances where Z-stacks were misinterpreted as spatial dimensions.
-   **Alignment**: Raw intensity images were paired with manual segmentation masks using a fuzzy matching algorithm with manual overrides to resolve filename inconsistencies.
-   **Quality Control**: Samples with dimension mismatches between image and label were automatically excluded.

### Model Configuration and Training
We utilized the `3d_fullres` configuration of nnU-Net, which trains a 3D U-Net on full-resolution patches. The framework automatically determined optimal patch sizes, batch sizes, and resampling strategies based on the median shape and spacing of the training data.
-   **Architecture**: U-Net with residual connections and deep supervision.
-   **Loss Function**: A combination of Dice Loss and Cross-Entropy Loss to handle class imbalance.
-   **Optimization**: Stochastic Gradient Descent (SGD) with Nesterov momentum and a polynomial learning rate decay.
-   **Training Strategy**: Models were trained using a 5-fold cross-validation setup (1000 epochs). Validation was conducted on held-out sets consisting of 10 actin cases, 8 mitochondria cases, and 19 nucleus cases. Segmentation accuracy was quantified using the Dice similarity coefficient, yielding mean values of 86.8% for actin, 80.3% for mitochondria, and 90.8% for nucleus on the validation sets. These results supported the use of nnU-Net segmentations for downstream quantitative phenotyping. For final inference, the model from Fold 0 was selected based on validation performance.

### Inference
Inference was performed using a sliding window approach with Gaussian overlap weighting to minimize stitching artifacts at patch boundaries. Test Time Augmentation (TTA) was employed to further enhance prediction robustness by averaging predictions across multiple geometric transformations (mirroring). The resulting probability maps were thresholded to generate final binary segmentation masks for downstream morphometric analysis.

## Image Processing and Feature Extraction

To quantify cellular morphology, 3D segmentation masks of Actin, Mitochondria, and Nuclei were processed using custom Python scripts. Voxel dimensions were extracted directly from the original NRRD files when available to ensure accurate physical measurements (µm), falling back to NIfTI headers only when necessary. All binary masks underwent morphological binary closing with a 3x3x3 structuring element to fill small holes and reduce surface noise before analysis.

### Morphological Metrics
For each structure, we calculated total volume and surface area using the Marching Cubes algorithm (scikit-image). Sphericity was derived as a normalized ratio of volume to surface area ($Ψ = \frac{\pi^{1/3} (6V)^{2/3}}{A}$), ranging from 0 (non-spherical) to 1 (perfect sphere). 

Specific structural features included:
- **Mitochondria**: We analyzed mitochondrial network topology using the *skan* library. Skeletons were generated to compute total network length, branch count, and junction density. Tortuosity was quantified for each branch as the ratio of path length to Euclidean distance. We also calculated the number of distinct mitochondrial fragments and their individual sphericity.
- **Actin**: Skeletonization was performed to measure total filament length. To assess cell shape complexity, we computed the Convex Hull volume and Solidity (Volume / Convex Hull Volume). Principal Component Analysis (PCA) was applied to the voxel coordinates to extract the major, intermediate, and minor axis lengths, from which we derived Fractional Anisotropy to quantify directional alignment.
- **Nucleus**: In addition to volume and sphericity, we calculated Circularity from the maximum 2D cross-sectional area and Elongation based on the ratio of major to minor axes derived from the inertia tensor.

## Statistical Analysis

Statistical comparisons were performed between four groups: Healthy (Untreated), Healthy (Collagen-treated), TAA (Untreated), and TAA (Collagen-treated). All statistical analyses were conducted using Python (scipy, statsmodels).

### Group Comparisons
Due to the non-normal distribution of morphological features (confirmed by Shapiro-Wilk test), we employed the non-parametric Kruskal-Wallis H test to detect omnibus differences across groups. For features showing significant differences ($p < 0.05$), we performed post-hoc pairwise comparisons using Dunn’s test with Bonferroni correction to control the family-wise error rate.

### Linear Mixed Models (LMM)
To rigorously account for the hierarchical structure of the data (multiple cells sampled from the same subject), we implemented Linear Mixed Models (LMM). This approach addresses pseudoreplication by modeling 'Subject' as a random effect, while 'Disease' (Healthy vs TAA) and 'Collagen' (Treatment status) were modeled as fixed effects. The interaction term (`Disease × Collagen`) was included to specifically test for differential treatment responses in diseased cells (rescue effect). Model p-values were corrected for multiple testing using the Benjamini-Hochberg False Discovery Rate (FDR) method.

## Machine Learning Classification

We developed machine learning classifiers to evaluate whether morphological features could predict disease state and treatment response. The analysis focused on three specific tasks:
1.  **Disease Classification**: Distinguishing Healthy vs. TAA cells (Untreated).
2.  **Treatment Classification**: Distinguishing TAA cells (Untreated) vs. TAA cells (Collagen-treated).
3.  **Rescue Assessment**: Classifying TAA+Collagen cells based on their phenotypic distance to the Healthy Centroid (Rescued vs. Non-Rescued).

### Model Development
Feature selection was performed using univariate statistical tests (ANOVA F-value) to identify the top 10-15 most discriminative features for each task. Features were standardized (zero mean, unit variance) prior to training.

We compared four classifier architectures:
- **Random Forest**: Regularized with limited depth (`max_depth=3`) and minimum samples per leaf to prevent overfitting.
- **Logistic Regression**: Applied with L2 regularization.
- **Support Vector Machine (SVM)**: Using an RBF kernel with probability estimates.
- **Multi-Layer Perceptron (MLP)**: A neural network with a simplified architecture (single hidden layer of 10 neurons) and strong L2 regularization (`alpha=2.0`) to enforce generalizability.

### Validation
Model performance was evaluated using Stratified 5-Fold Cross-Validation, ensuring that class proportions were preserved across folds. We reported Accuracy, F1-Score, and Area Under the Receiver Operating Characteristic Curve (AUC-ROC). Models were specifically tuned to minimize the generalization gap (difference between training and test performance) to ensure robustness on unseen data.
