# 4-Phase Thesis Analysis Pipeline

## Overview

This directory contains a complete, thesis-ready statistical and machine learning analysis pipeline for the aneurysm morphometry project.

## The 4 Phases

### **Phase 1: Statistical Discovery**
**Goal:** Prove that morphological features show real biological differences (not noise)

**Script:** `phase1_statistical_discovery.py`

**Outputs:**
- `phase1/statistical_results.csv` - All features with p-values, FDR correction
- `phase1/top_features_violin_plots.png` - Visualization of top 6 features
- Normality tests (Shapiro-Wilk)
- Mann-Whitney U / t-tests with FDR correction

**Thesis Section:** 3.1 - 3D Morphometric Analysis

---

### **Phase 2: Binary Classifier Validation**
**Goal:** Validate that features can classify Healthy vs TAA (No Collagen)

**Script:** `phase2_binary_classifier.py`

**Outputs:**
- `phase2/healthy_vs_taa_model.pkl` - Trained Random Forest (saved for Phase 3)
- `phase2/confusion_matrix.png`
- `phase2/feature_importance.png` - Top predictive features
- `phase2/roc_curve.png`
- `phase2/predictions.csv`

**Expected Accuracy:** >80% (validates Phase 1 statistics)

**Thesis Section:** 3.2 - Machine Learning Classification

---

### **Phase 3: Rescue Experiment**
**Goal:** Test if Collagen "rescues" TAA cells using the Phase 2 model

**Script:** `phase3_rescue_analysis.py`

**How it works:**
1. Load the Phase 2 model (trained on No-Collagen data)
2. Feed it TAA+Collagen cells (which it has NEVER seen)
3. Ask: "Does the AI think these are Healthy or TAA?"

**Interpretation:**
- If AI predicts **Healthy** → Collagen rescued the phenotype ✅
- If AI predicts **TAA** → Collagen did NOT rescue ❌

**Outputs:**
- `phase3/rescue_predictions.csv`
- `phase3/rescue_probability_distribution.png`
- `phase3/rescue_comparison.png`

**Thesis Section:** 3.3 - Collagen Treatment Induces Phenotypic Reversion

---

### **Phase 4: 4-Class Stress Test**
**Goal:** Can AI distinguish Disease AND Treatment simultaneously?

**Script:** `phase4_multiclass_classifier.py`

**Classes:**
1. Healthy - No Collagen
2. Healthy + Collagen
3. TAA - No Collagen
4. TAA + Collagen

**Expected Accuracy:** ~60-70% (lower is expected and okay!)

**Why lower?** Disease effect >> Collagen effect

**Outputs:**
- `phase4/confusion_matrix_4class.png`
- `phase4/feature_importance_4class.png`
- `phase4/per_class_accuracy.png`
- `phase4/predictions_4class.csv`

**Thesis Section:** 3.4 - Discussion: Subtle Treatment Effects vs. Strong Disease Signal

---

## How to Run

### Option 1: Run All Phases at Once (Recommended)
```bash
python src/analysis/run_all_phases.py
```

### Option 2: Run Phases Individually
```bash
# Phase 1
python src/analysis/phase1_statistical_discovery.py

# Phase 2 (requires Phase 1 to understand context)
python src/analysis/phase2_binary_classifier.py

# Phase 3 (requires Phase 2 model)
python src/analysis/phase3_rescue_analysis.py

# Phase 4
python src/analysis/phase4_multiclass_classifier.py
```

---

## Required Files

- `outputs/Advanced_Features_Raw.csv` - Feature data (from voxel-aware analysis)
- `data/Book1.xlsx` - Metadata (Healthy vs TAA labels)

---

## Output Directory Structure

```
src/analysis/outputs/
├── phase1/
│   ├── statistical_results.csv
│   └── top_features_violin_plots.png
├── phase2/
│   ├── healthy_vs_taa_model.pkl
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   ├── roc_curve.png
│   └── predictions.csv
├── phase3/
│   ├── rescue_predictions.csv
│   ├── rescue_probability_distribution.png
│   └── rescue_comparison.png
└── phase4/
    ├── confusion_matrix_4class.png
    ├── feature_importance_4class.png
    ├── per_class_accuracy.png
    └── predictions_4class.csv
```

---

## Statistical Methodology

**Phase 1:**
- Normality: Shapiro-Wilk test
- Comparison: Mann-Whitney U (non-parametric) or t-test (parametric)
- Multiple testing correction: FDR (Benjamini-Hochberg)
- Significance threshold: α = 0.05

**Phases 2-4:**
- Classifier: Random Forest (ensemble method)
- Validation: 5-fold cross-validation
- Test set: 30% holdout (stratified)

---

## Thesis Integration

Each phase directly corresponds to a thesis subsection:

| Phase | Thesis Section | Key Figure |
|-------|---------------|------------|
| 1 | 3.1: Morphometric Analysis | Violin plots of significant features |
| 2 | 3.2: ML Classification | Feature importance + confusion matrix |
| 3 | 3.3: Collagen Rescue | Rescue probability distribution |
| 4 | 3.4: Discussion | 4-class confusion matrix |

---

## Expected Results Summary

| Phase | Metric | Expected Value | Interpretation |
|-------|--------|----------------|----------------|
| 1 | Significant features (FDR < 0.05) | 10-20 | Strong biological signal |
| 2 | Binary accuracy | >80% | Validates statistical findings |
| 3 | Rescue rate | Variable | Quantifies collagen effect |
| 4 | 4-class accuracy | 60-70% | Disease >> Treatment effect |

---

## Troubleshooting

**Error: "No TAA+Collagen cells found" (Phase 3)**
- Check that your `outputs/Advanced_Features_Raw.csv` contains cells with "+coll" or "+Col" in the filename
- Verify collagen status extraction in the script

**Error: "Model not found" (Phase 3)**
- Run Phase 2 first to generate the model

**Low accuracy in Phase 2 (<70%)**
- Check data quality (missing values, inf values)
- Verify metadata labels are correct
- Review feature extraction output

---

## Author
Antigravity AI - Thesis-Ready Analysis Pipeline
