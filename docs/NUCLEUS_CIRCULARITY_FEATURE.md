# Nucleus Circularity Feature Addition

## Summary

Added **`Nucleus_Circularity_ratio`** as a new feature to complement the existing `Nucleus_Sphericity_ratio`. This provides a more robust shape metric that isn't affected by 3D mesh resolution issues.

## Why Circularity?

### Problems with Sphericity (3D):
- ❌ Depends on marching cubes surface area estimation
- ❌ Coarse meshes → underestimated surface area → artificially high sphericity
- ❌ 75% of nuclei clamped to 1.0 (maximum), hiding real shape variation
- ❌ Cannot distinguish between shapes when at ceiling value

### Advantages of Circularity (2D):
- ✅ Uses 2D contour from maximum area slice
- ✅ Not affected by 3D mesh resolution
- ✅ Always properly bounded [0, 1] by mathematical definition
- ✅ Better suited for flattened/elongated nuclei
- ✅ Faster to compute
- ✅ More intuitive interpretation

## Implementation

### Method: Maximum Area Slice
```python
Circularity = 4π × Area / Perimeter²
```

**Process:**
1. Scan through all slices in all 3 axes (Z, Y, X)
2. Find the slice where nucleus has maximum cross-sectional area
3. Measure area and perimeter of that 2D contour
4. Calculate circularity using the formula above

**Interpretation:**
- **1.0** = Perfect circle (all radii equal)
- **< 1.0** = Elongated or irregular
- **0.0** = Extremely irregular/elongated

### Code Changes

**File:** `src/postprocessing/analyze_structures.py`

1. **Added `calculate_circularity()` function** (lines 65-127):
   - Finds maximum area slice across all axes
   - Uses `regionprops` to get area and perimeter
   - Calculates 4πA/P²
   - Returns value [0, 1]

2. **Modified `analyze_nucleus()` function**:
   - Calls `calculate_circularity(mask)`
   - Adds to return dictionary as `'Circularity'`

3. **Keeps sphericity** for backward compatibility

## Usage

### Rerun Feature Extraction

To add this feature to your dataset, rerun nucleus analysis:

```bash
python src/postprocessing/analyze_structures.py \
    --input_dir outputs/nnunet_inference/nucleus \
    --output_csv outputs/Nucleus_Features.csv \
    --structure nucleus \
    --raw_nrrd_dir raw_data
```

Then regenerate the combined features file.

### Expected Output

The `Nucleus_Features.csv` will now include:
- `Nucleus_Sphericity_ratio` (existing - 3D metric)
- `Nucleus_Circularity_ratio` (**new** - 2D metric)
- All other existing features

### Analysis Recommendations

**Use Circularity for:**
- Primary shape analysis (more reliable)
- Comparing to published work (common metric)
- Clinical biomarkers (robust and interpretable)

**Keep Sphericity for:**
- 3D shape assessment (when mesh quality is good)
- Comparing with previous results
- Completeness of feature set

**Compare both to:**
- Validate findings (should correlate but circularity will show more range)
- Identify artifacts (if sphericity = 1.0 but circularity < 1.0 → mesh issue)

## Expected Results

Based on observations that nuclei are "flat ovals":

| Metric | Expected Mean | Expected Range | Interpretation |
|--------|---------------|----------------|----------------|
| **Sphericity** | ~0.89 | [0.06, **1.00**] | Ceiling effect - 75% at max |
| **Circularity** | ~0.70-0.85 | [0.30, 0.95] | True variation visible |

Circularity should:
- Show **lower values** than sphericity (no ceiling effect)
- Have **more variation** between cells
- **Correlate with elongation** ratio (inverse relationship)
- Potentially show **disease differences** hidden by sphericity ceiling

## Benefits for Analysis

1. **More sensitive biomarker**: Can detect subtle shape changes
2. **Better statistics**: Not ceiling-bounded, normal distribution
3. **Biological relevance**: Nuclei flattening/elongation in disease
4. **Publication ready**: Standard metric in cell morphology papers

## Next Steps

1. ✅ Code implemented and tested
2. ⏳ **Rerun feature extraction** on all cells
3. ⏳ **Update combined features** CSV
4. ⏳ **Rerun statistical analyses** to compare with sphericity
5. ⏳ **Generate plots** comparing sphericity vs circularity

---

**Date:** 2026-02-04  
**Feature:** Nucleus_Circularity_ratio  
**Type:** Ratio [0, 1]  
**Method:** Maximum area slice, 4πA/P²
