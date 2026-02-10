# Data Filtering Strategy - Comprehensive Classifier Comparison

## Your Experimental Design: 4 Groups

```
┌─────────────────────────────────────────────────────────────────┐
│                    ALL CELLS IN DATASET                         │
├─────────────────┬────────────────┬──────────────┬───────────────┤
│ Healthy-NoColl  │ Healthy+Coll   │ TAA-NoColl   │  TAA+Coll     │
│    (Group 0)    │   (Group 1)    │  (Group 2)   │  (Group 3)    │
│                 │                │              │               │
│  Control        │ Healthy        │  Disease     │  Treatment    │
│  Baseline       │ + Treatment    │  No Treat    │  + Treatment  │
└─────────────────┴────────────────┴──────────────┴───────────────┘
```

## Task-Specific Data Filtering

### ✅ Task 1: Diseased or Not
**Question**: Can we distinguish healthy from diseased cells using ONLY untreated samples?

```
Uses ONLY:
┌─────────────────┐                ┌──────────────┐
│ Healthy-NoColl  │                │ TAA-NoColl   │
│    (Label 0)    │       vs       │  (Label 1)   │
│                 │                │              │
└─────────────────┘                └──────────────┘

EXCLUDES: Healthy+Coll, TAA+Coll
WHY: To isolate pure disease effect without confounding treatment
```

### ✅ Task 2: Treated or Not  
**Question**: Does collagen treatment change TAA cells?

```
Uses ONLY TAA cells:
                                   ┌──────────────┬───────────────┐
                                   │ TAA-NoColl   │  TAA+Coll     │
                                   │  (Label 0)   │  (Label 1)    │
                                   │              │               │
                                   └──────────────┴───────────────┘

EXCLUDES: All Healthy cells
WHY: To isolate treatment effect on diseased cells only
```

### ✅ Task 3: Rescued or Not
**Question**: Does collagen make TAA cells more similar to healthy cells?

```
Step 1: Calculate Healthy Centroid
┌─────────────────┬────────────────┐
│ Healthy-NoColl  │ Healthy+Coll   │
│                 │                │  → Compute mean features → Healthy Centroid
└─────────────────┴────────────────┘

Step 2: Measure distance of each TAA+Coll cell to Healthy Centroid
                                                  ┌───────────────┐
                                                  │  TAA+Coll     │
                                                  │               │
                                                  └───────────────┘
                                                          │
                                             Calculate distance to Healthy Centroid
                                                          │
                                         ┌────────────────┴────────────────┐
                                         │                                 │
                                  Below Median                      Above Median
                                  (Label 0)                         (Label 1)
                                  "Rescued"                         "Not Rescued"

USES: Only TAA+Coll cells
CLASSIFICATION: Distance to Healthy centroid (median split)
WHY: Tests if collagen brings TAA cells phenotypically closer to healthy
```

## Comparison with OLD (Incorrect) Script

### ❌ OLD Script (WRONG)
```
Task 1: ALL Healthy vs ALL TAA
┌─────────────────┬────────────────┐   ┌──────────────┬───────────────┐
│ Healthy-NoColl  │ Healthy+Coll   │   │ TAA-NoColl   │  TAA+Coll     │
│                 │                │vs │              │               │
└─────────────────┴────────────────┘   └──────────────┴───────────────┘
         Mixed treatment status              Mixed treatment status
              ❌ CONFOUNDED                        ❌ CONFOUNDED
```

### ✅ NEW Script (CORRECT)
```
Task 1: ONLY untreated samples
┌─────────────────┐                    ┌──────────────┐
│ Healthy-NoColl  │          vs        │ TAA-NoColl   │
└─────────────────┘                    └──────────────┘
   Pure healthy                          Pure diseased
   ✅ CLEAN COMPARISON
```

## Code Implementation

### Task 1: Diseased or Not
```python
# Filter to only -NoColl samples (Groups 0 and 2)
mask = df_agg['FourClass'].isin([0, 2])  
X_task = X[mask]
y_task = (df_agg.loc[mask, 'FourClass'] == 2).astype(int)
# 0 = Healthy-NoColl, 1 = TAA-NoColl
```

### Task 2: Treated or Not
```python
# Filter to only TAA samples (Groups 2 and 3)
mask = df_agg['FourClass'].isin([2, 3])
X_task = X[mask]
y_task = (df_agg.loc[mask, 'FourClass'] == 3).astype(int)
# 0 = TAA-NoColl, 1 = TAA+Coll
```

### Task 3: Rescued or Not
```python
# Use all Healthy for centroid (Groups 0 and 1)
healthy_mask = df_agg['FourClass'].isin([0, 1])
healthy_centroid = X_scaled[healthy_mask].mean().values

# Measure distance for TAA+Coll only (Group 3)
taa_coll_mask = df_agg['FourClass'] == 3
distances = np.linalg.norm(X_scaled[taa_coll_mask] - healthy_centroid, axis=1)

# Split by median distance
y_task = (distances > median_distance).astype(int)
# 0 = Closer to healthy (Rescued), 1 = Further from healthy (Not Rescued)
```

## Why This Matthere

### Scientific Validity
- **Task 1**: Pure disease effect (no treatment confounding)
- **Task 2**: Pure treatment effect (no disease/healthy confounding)
- **Task 3**: Rescue effect (phenotypic normalization)

### Statistical Power
- Each comparison uses appropriate sample sizes
- Avoids diluting signal by mixing groups
- Reduces confounding variables

### Interpretability
- Clear biological questions
- Direct experimental comparisons
- Actionable results

## Running the Analysis

```bash
cd /home/luisa/aneurysm_project
python src/classification/comprehensive_classifier_comparison.py
```

The script will:
1. Load all cells and create 4-class labels
2. For each task, filter to appropriate groups
3. Compare Random Forest, XGBoost, MLP, Logistic Regression
4. Generate plots and summary tables
5. Save results to `classification_results/comprehensive_comparison/`
