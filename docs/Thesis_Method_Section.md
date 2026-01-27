# Methods

## 3.1 3D Image Preprocessing and Dataset Preparation
Raw confocal microscopy images were processed to ensure compatibility with deep learning pipelines. Original files in `.nrrd` format were converted to NIfTI (`.nii.gz`) format using the `nibabel` library. A dimensionality standardization step was implemented to correct inconsistent axis ordering observed in the raw data. A heuristic algorithm checked if the first dimension was significantly smaller than the others (indicating a Z-stack depth); in such cases (Z, Y, X), the array was transposed to the standard (X, Y, Z) orientation `(2, 1, 0)` to match the expected input for 3D analysis.

To pair raw images with their corresponding manual segmentation masks, a robust two-stage matching algorithm was developed using Python scripts.
1.  **Strict Normalization**: Filenames were normalized using NFC unicode normalization, lowercased, and stripped of whitespace to identify exact matches.
2.  **Fuzzy and Manual Matching**: A secondary pass utilized a predefined dictionary of manual overrides to resolve specific naming discrepancies (e.g., `01asc-230col` vs `01asc-230+col`) and a fuzzy logic matcher that compared key substrings while ignoring characters like `+` or `-` to handle inconsistent annotation styles.

## 3.2 3D Segmentation with nnU-Net V2
Automated segmentation was performed using **nnU-Net V2**, a state-of-the-art self-configuring framework for biomedical image segmentation. The environment was configured for **3D Full-Resolution** processing (`3d_fullres`) to leverage the volumetric nature of the Z-stack data.

Two separate datasets were configured:
*   **Dataset 001 (Actin)**: Trained on 47 manually segmented cases.
*   **Dataset 002 (Mitochondria)**: Trained on 38 manually segmented cases.

Models were trained for **1000 epochs** on a GPU-accelerated environment (NVIDIA RTX 3090 / A100). The standard nnU-Net loss function (a combination of Dice loss and Cross-Entropy loss) was minimized using the Adam stochastic optimization method. Model performance was evaluated on a held-out validation set (10 cases for Actin, 8 for Mitochondria), achieving mean Dice scores of **86.8%** and **80.3%** respectively, indicating high fidelity to expert segmentation.

## 3.3 Morphometric Feature Extraction
Following segmentation, we extracted high-dimensional morphological features from the generated 3D masks to quantify cellular structure. Feature extraction was implemented in Python using the `skimage`, `scipy`, and `skan` libraries.

*   **Volumetric Features**: Total Volume ($\mu m^3$) and Surface Area ($\mu m^2$) were calculated via voxel-wise analysis using `nibabel` to access physical voxel dimensions.
*   **Network Analysis**: Mitochondrial networks were skeletonized using `skimage.morphology.skeletonize`. The resulting 1-pixel-wide skeletons were analyzed using the `skan` library `Skeleton` object to compute topological metrics including branch count, junction density, and average branch length.
*   **Tortuosity Analysis**: We calculated the **tortuosity ratio** for every specific branch in the mitochondrial network. This was defined mathematically as:
    $$ \text{Tortuosity} = \frac{\text{Path Length along Skeleton}}{\text{Euclidean Distance between Endpoints}} $$
    A value of 1.0 indicates a perfectly straight line, while higher values indicate increasing curvature.

Features were aggregated at the cell level. Continuous metrics like tortuosity were summarized using mean, maximum, and standard deviation to capture the heterogeneity of the mitochondrial network within each cell. Label assignment (Healthy vs TAA) was performed by parsing the numeric ID from the filename and cross-referencing it with the project metadata (`Book1.xlsx`).

## 3.4 Statistical Analysis
Statistical analysis was performed using Python libraries `scipy.stats`, `statsmodels`, and `scikit_posthocs`. Data normality was assessed using the **Shapiro-Wilk test**. Since most morphological features followed a non-normal distribution, non-parametric tests were selected used.

A **Kruskal-Wallis H test** was employed as the omnibus test to detect significant differences across the four experimental groups:
1.  **Healthy** (No Collagen)
2.  **Healthy** (+ Collagen)
3.  **TAA** (No Collagen)
4.  **TAA** (+ Collagen)

For features showing significant differences (Omnibus $p < 0.05$), **Dunn’s post-hoc test** was performed. To rigorously control for the Family-Wise Error Rate (FWER) and False Discovery Rate (FDR) across the battery of extracted features:
*   **Bonferroni correction** was applied to pairwise p-values within Dunn's test.
*   **Benjamini-Hochberg FDR correction** was applied to the omnibus p-values across all tested features, with a significance threshold of $\alpha = 0.05$.

Collagen status was extracted from filenames using regex pattern matching (detecting `+coll`, `+col`, `plus coll` vs `-coll`, `nocoll`, `dmso`).

## 3.5 Machine Learning Classification
To validate the discriminative power of the features and quantify phenotypic shifts, we implemented machine learning models using `scikit-learn`.

**Binary Classification (Healthy vs. TAA):**
We trained a **Neural Network (MLPClassifier)** to distinguish between Healthy and TAA cells (untreated).
*   **Architecture**: Two hidden layers with 50 and 25 neurons respectively `(50, 25)`.
*   **Optimization**: Adam solver with an initial learning rate of 0.001 and L2 regularization (`alpha=0.1`) to prevent overfitting.
*   **Preprocessing**: All features were standardized using `StandardScaler` (zero mean, unit variance) prior to training.
*   **Evaluation**: Performance was assessed using 5-fold cross-validation and a stratified 30% holdout test set. Feature importance was computed using **Permutation Importance** (30 repeats), which measures the decrease in model accuracy when a single feature's values are randomly shuffled.

**Rescue Experiment Analysis:**
To quantify "phenotypic rescue," the Neural Network—trained strictly on untreated Healthy and TAA cells—was applied to the unseen **TAA + Collagen** group. The model’s output probability ($P(Healthy)$) was analyzed to determine if collagen treatment shifted the morphometric signature of TAA cells towards the healthy phenotype.
