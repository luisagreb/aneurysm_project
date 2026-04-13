# Thesis Figure Index
Maps every figure in `thesis_figures/` to the Python script that generated it.
Run all scripts from the **project root** (`/home/luisa/aneurysm_project/`).

---

## 00_pipeline

| Figure | Script | Run command |
|--------|--------|-------------|
| `pipeline_overview.png / .pdf` | `src/thesis_figures/plot_pipeline.py` | `python src/thesis_figures/plot_pipeline.py` |

---

## 01_nnunet_training

| Figure | Script | Run command |
|--------|--------|-------------|
| `actin_learning_curve.png` | `src/thesis_plot/plot_nnunet_curves.py` | `python src/thesis_plot/plot_nnunet_curves.py` |
| `mito_learning_curve.png` | `src/thesis_plot/plot_nnunet_curves.py` | same |
| `nucleus_learning_curve.png` | `src/thesis_plot/plot_nnunet_curves.py` | same |
| `all_learning_curves.png` | `src/thesis_plot/plot_nnunet_curves.py` | same |
| `mitochondria_learning_curve.png` | `src/thesis_plot/plot_nnunet_curves.py` | same |

---

## 02_segmentation_viz

These figures were generated manually (ImageJ / napari / notebook) or from segmentation
inspection scripts not tracked in `src/`. They are static reference figures.

| Figure | Origin |
|--------|--------|
| `comparison_nocoll.png` | Manual / napari export |
| `comparison_pluscoll.png` | Manual / napari export |
| `skeleton_visualization.png` | Manual / napari export |
| `skeleton_comparison_grid.png` | Manual / napari export |
| `skeleton_yz_comparison.png` | Manual / napari export |
| `skeleton_verification.png` | Manual / napari export |
| `skeleton_verification_TAA.png` | Manual / napari export |
| `cyclomatic_visualization.png` | Manual / napari export |
| `thresholding_summary.png` | Manual / napari export |

---

## 03_three_group_analysis
**Main script:** `src/analysis/three_group_analysis.py`
**Overview figures script:** `src/analysis/make_overview_plot.py`

Run:
```bash
python src/analysis/three_group_analysis.py
python src/analysis/make_overview_plot.py
```

### Core disease effect figures
| Figure | Script | Notes |
|--------|--------|-------|
| `graph1_omnibus_effect_size.png` | `three_group_analysis.py` | Kruskal-Wallis effect sizes, all features |
| `graph2_pairwise_heatmap.png` | `three_group_analysis.py` | FDR-corrected heatmap, 3 pairwise comparisons |
| `graph3_bav_signature_boxplots.png` | `three_group_analysis.py` | Boxplots of BAV-specific features |
| `graph4_shared_vs_distinct.png` | `three_group_analysis.py` | Venn-style shared vs group-specific effects |
| `heatmap_raw.png` | `three_group_analysis.py` | Raw Cohen's d heatmap |
| `heatmap_age_corrected.png` | `three_group_analysis.py` | Age-corrected Cohen's d heatmap |
| `forest_pairwise_raw.png` | `three_group_analysis.py` | Forest plot of all pairwise comparisons |
| `boxplots_significant.png` | `three_group_analysis.py` | Boxplots for FDR-significant features |
| `age_scatter_mito_branch.png` | `three_group_analysis.py` | Age vs Mito Branch Count scatter |

### Per-feature pairwise boxplots (21 features)
| Figures | Script | Notes |
|---------|--------|-------|
| `pairwise_boxplots/*.png` | `three_group_analysis.py` | One PNG per feature, TAV-NA / TAV-ATAA / BAV-ATAA |

### Overview bar charts
| Figure | Script | Notes |
|--------|--------|-------|
| `overview_pairwise.png` | `make_overview_plot.py` (Figure 1) | 3-panel, Cohen's d, NoCollagen cells |
| `overview_collagen.png` | `make_overview_plot.py` (Figure 2) | Paired collagen effect, specimen-level |
| `overview_collagen_celllevel.png` | `make_overview_plot.py` (Figure 3) | Collagen effect, cells treated as independent |
| `overview_healthy_vs_taa_collagen.png` | `make_overview_plot.py` (Figure 4) | TAV-NA −Col as reference, TAA ±Col |

### Layer 4 — Clinical covariates
| Figure | Script | Notes |
|--------|--------|-------|
| `layer4_clinical/diameter_correlation_forest.png` | `three_group_analysis.py` | Spearman ρ forest plot, feature vs aortic diameter |
| `layer4_clinical/diameter_scatter_top4.png` | `three_group_analysis.py` | Scatter plots, top 4 diameter-correlated features |
| `layer4_clinical/hypertension_forest.png` | `three_group_analysis.py` | Hypertension effect forest plot |

### Layer 5 — Sex effects
| Figure | Script | Notes |
|--------|--------|-------|
| `layer5_sex/sex_forest_overall.png` | `three_group_analysis.py` | Forest plot, Male vs Female overall |
| `layer5_sex/sex_heatmap_bygroup.png` | `three_group_analysis.py` | Sex effect heatmap per group |
| `layer5_sex/Actin_Fractional_Anisotropy_sex_boxplot.png` | `three_group_analysis.py` | Per-feature sex boxplot |
| `layer5_sex/Actin_Major_Axis_sex_boxplot.png` | `three_group_analysis.py` | Per-feature sex boxplot |
| `layer5_sex/Mito_Fragment_Count_sex_boxplot.png` | `three_group_analysis.py` | Per-feature sex boxplot |
| `layer5_sex/Nucleus_Elongation_sex_boxplot.png` | `three_group_analysis.py` | Per-feature sex boxplot |

---

## 04_taa_layered_analysis
**Main script:** `src/analysis/complete_stat_analysis.py`

Run:
```bash
python src/analysis/complete_stat_analysis.py
```

| Figure | Script (line ~) | Notes |
|--------|-----------------|-------|
| `layer1_disease_effect.png` | `complete_stat_analysis.py` (L194) | Layer 1: naive Healthy vs TAA effect sizes |
| `layer2_comparison.png` | `complete_stat_analysis.py` (L258) | Layer 2: naive vs age-corrected comparison |
| `layer2_age_slopes.png` | `complete_stat_analysis.py` (L271) | Layer 2: age regression slopes per feature |
| `layer2_comparison_l1_l2.png` | `complete_stat_analysis.py` | Layer 1 vs Layer 2 side-by-side |
| `layer3_lmm_forest.png` | `complete_stat_analysis.py` (L343) | Layer 3: LMM disease fixed effect forest plot |
| `layer3_lmm_heatmap.png` | `complete_stat_analysis.py` (L387) | Layer 3: LMM heatmap all features |
| `layer4_rescue_direction.png` | `complete_stat_analysis.py` (L479) | Layer 4: collagen rescue direction per feature |
| `4layer_summary_bar.png` | `complete_stat_analysis.py` | Summary bar: significant features per layer |
| `5layer_summary_bar.png` | `complete_stat_analysis.py` | Same + Layer 5 sex |
| `cross_layer_heatmap.png` | `complete_stat_analysis.py` (L705) | Heatmap: significance across all 5 layers |
| `sex_effect_overall.png` | `complete_stat_analysis.py` (L605) | Layer 5: sex effect forest plot |
| `sex_effect_heatmap.png` | `complete_stat_analysis.py` (L632) | Layer 5: sex effect heatmap |
| `sex_significant_boxplots.png` | `complete_stat_analysis.py` (L666) | Layer 5: boxplots for significant sex features |
| `pairwise_boxplots/*.png` | `sex_effect.py` | Per-feature Healthy vs TAA boxplots (cell-level) |

---

## 05_smc_analysis
**Main scripts:**
- `src/analysis/smc_statistical_analysis.py` — LMM, pairwise heatmap, violin plots, age/sex
- `src/analysis/make_layer1v2_comparison.py` — layer2_comparison_slide
- `src/analysis/make_age_scatter.py` — layer2_age_scatter

Run:
```bash
python src/analysis/smc_statistical_analysis.py
python src/analysis/make_layer1v2_comparison.py
python src/analysis/make_age_scatter.py
```

| Figure | Script (line ~) | Notes |
|--------|-----------------|-------|
| `layer1_disease_effect.png` | `complete_stat_analysis.py` (L194) | Same as 04 — Healthy vs TAA naive effect |
| `layer2_comparison.png` | `complete_stat_analysis.py` (L258) | Naive vs age-corrected |
| `layer2_comparison_slide.png` | `make_layer1v2_comparison.py` (L127) | Slide-ready version of layer2 comparison |
| `layer2_age_scatter.png` | `make_age_scatter.py` (L116) | Age scatter per feature |
| `layer2_age_slopes.png` | `complete_stat_analysis.py` (L271) | Age regression slopes |
| `layer3_lmm_forest.png` | `complete_stat_analysis.py` (L343) | LMM forest plot |
| `layer3_lmm_heatmap.png` | `complete_stat_analysis.py` (L387) | LMM heatmap |
| `layer4_3condition_bars.png` | `complete_stat_analysis.py` | 3-condition bar chart (Healthy / TAA NoCol / TAA +Col) |
| `layer4_rescue_direction.png` | `complete_stat_analysis.py` (L479) | Rescue direction arrows |
| `lmm_coefficient_heatmap.png` | `smc_statistical_analysis.py` (L189) | LMM coefficients heatmap |
| `lmm_forest_disease_effect.png` | `smc_statistical_analysis.py` (L212) | LMM forest, disease fixed effect |
| `pairwise_effect_heatmap.png` | `smc_statistical_analysis.py` (L283) | Pairwise effect size heatmap |
| `violin_top_features.png` | `smc_statistical_analysis.py` (L334) | Violin plots, top discriminating features |
| `violin_top_features_plots.png` | `smc_statistical_analysis.py` (L334) | Alternate version from plots/ subfolder |
| `age_scatter_plots.png` | `smc_statistical_analysis.py` (L417) | Age scatter, all features |
| `sex_boxplots.png` | `smc_statistical_analysis.py` (L457) | Sex boxplots |
| `sex_effect_overall.png` | `complete_stat_analysis.py` (L605) | Sex forest plot |
| `sex_effect_heatmap.png` | `complete_stat_analysis.py` (L632) | Sex heatmap |
| `sex_significant_boxplots.png` | `complete_stat_analysis.py` (L666) | Sex significant boxplots |
| `summary_significant_features.png` | `smc_statistical_analysis.py` (L503) | Summary bar chart |
| `layer1_vs_layer2.png` | `smc_statistical_analysis.py` | Layer 1 vs layer 2 comparison |
| `5layer_summary_bar.png` | `complete_stat_analysis.py` | 5-layer summary |
| `cross_layer_heatmap.png` | `complete_stat_analysis.py` (L705) | Cross-layer heatmap |
| `pairwise_boxplots/*.png` | `sex_effect.py` (L351) | Per-feature Healthy vs TAA boxplots |
| `lmm/lmm_comprehensive_table.png` | `LinearMixedModel.py` | Full LMM results table |
| `lmm/lmm_equation.png` | `LinearMixedModel.py` | LMM equation display |
| `lmm/lmm_summary_plot.png` | `LinearMixedModel.py` (L323) | LMM summary figure |
| `lmm/lmm_table_with_icc.png` | `LinearMixedModel.py` | LMM table with ICC values |

---

## 06_classification
**Main scripts:**
- `src/classification/regularized_classifier_comparison.py` — main comparison figures
- `src/classification/comprehensive_classifier_3groups.py` — 3-group tasks

Run:
```bash
python src/classification/regularized_classifier_comparison.py
python src/classification/comprehensive_classifier_3groups.py
```

| Figure | Script (line ~) | Notes |
|--------|-----------------|-------|
| `overall_comparison.png` | `regularized_classifier_comparison.py` (L397) | All 3 tasks accuracy comparison |
| `diseased_or_not_comparison.png` | `regularized_classifier_comparison.py` (L167) | Diseased vs Healthy classifier bars |
| `treated_or_not_comparison.png` | `regularized_classifier_comparison.py` | Treated vs Untreated classifier |
| `rescued_or_not_comparison.png` | `regularized_classifier_comparison.py` | Rescued vs Not classifier |
| `matrix_diseased_or_not.png` | `regularized_classifier_comparison.py` (L266) | Confusion matrix — diseased |
| `matrix_treated_or_not.png` | `regularized_classifier_comparison.py` (L266) | Confusion matrix — treated |
| `matrix_rescued_or_not.png` | `regularized_classifier_comparison.py` (L266) | Confusion matrix — rescued |
| `features_diseased_or_not.png` | `regularized_classifier_comparison.py` (L356) | Feature importance — diseased |
| `features_treated_or_not.png` | `regularized_classifier_comparison.py` (L356) | Feature importance — treated |
| `features_rescued_or_not.png` | `regularized_classifier_comparison.py` (L356) | Feature importance — rescued |

---

## 07_collagen
**Main script:** `src/analysis/three_group_analysis.py` (Layers 6)
**Overview figures:** `src/analysis/make_overview_plot.py`
**Violin figure:** `src/analysis/stats_3Groups_Col.py`

Run:
```bash
python src/analysis/three_group_analysis.py
python src/analysis/make_overview_plot.py
python src/analysis/stats_3Groups_Col.py
```

| Figure | Script (line ~) | Notes |
|--------|-----------------|-------|
| `overview_collagen.png` | `make_overview_plot.py` (Figure 2, L176) | Paired Cohen's d_z per group, specimen-level |
| `overview_collagen_celllevel.png` | `make_overview_plot.py` (Figure 3, L297) | Collagen effect, cells as independent |
| `overview_healthy_vs_taa_collagen.png` | `make_overview_plot.py` (Figure 4, L440) | TAV-NA −Col reference, TAA ±Col |
| `overview_grid.png` | `make_overview_plot.py` (Figure 5, L633) | 2×3 overview grid, all comparisons |
| `collagen_forest_overall.png` | `three_group_analysis.py` (L1794) | Paired collagen effect forest plot |
| `collagen_heatmap_bygroup.png` | `three_group_analysis.py` (L1844) | Collagen effect heatmap per group |
| `rescue_bubble.png` | `three_group_analysis.py` (L1896) | Rescue index bubble chart |
| `paired_lines/Actin_Solidity_paired_lines.png` | `three_group_analysis.py` (L1963) | Paired specimen lines — Actin Solidity |
| `paired_lines/Mito_Fragment_Count_paired_lines.png` | `three_group_analysis.py` (L1963) | Paired specimen lines — Mito Fragment Count |
| `paired_lines/Mito_Mean_Fragment_Volume_paired_lines.png` | `three_group_analysis.py` (L1963) | Paired specimen lines |
| `paired_lines/Mito_Mean_Tortuosity_paired_lines.png` | `three_group_analysis.py` (L1963) | Paired specimen lines |
| `top6_features_violin_6groups.png / .pdf` | `stats_3Groups_Col.py` (L160) | Violin plots, top 6 features × 6 conditions |

---

## Quick reference — script → figures

| Script | Figures generated |
|--------|------------------|
| `src/thesis_figures/plot_pipeline.py` | `00_pipeline/` |
| `src/thesis_plot/plot_nnunet_curves.py` | `01_nnunet_training/` |
| `src/analysis/three_group_analysis.py` | `03_three_group_analysis/` (core + layers 4–6) |
| `src/analysis/make_overview_plot.py` | `03_three_group_analysis/overview_*.png`, `07_collagen/overview_*.png` |
| `src/analysis/complete_stat_analysis.py` | `04_taa_layered_analysis/`, `05_smc_analysis/` (5layer subfolder) |
| `src/analysis/smc_statistical_analysis.py` | `05_smc_analysis/` (lmm, pairwise, violin, age_sex) |
| `src/analysis/make_layer1v2_comparison.py` | `05_smc_analysis/layer2_comparison_slide.png` |
| `src/analysis/make_age_scatter.py` | `05_smc_analysis/layer2_age_scatter.png` |
| `src/analysis/sex_effect.py` | `04_taa_layered_analysis/pairwise_boxplots/`, `05_smc_analysis/pairwise_boxplots/` |
| `src/analysis/LinearMixedModel.py` | `05_smc_analysis/lmm/` |
| `src/analysis/stats_3Groups_Col.py` | `07_collagen/top6_features_violin_6groups.png` |
| `src/classification/regularized_classifier_comparison.py` | `06_classification/` |
| `src/classification/comprehensive_classifier_3groups.py` | `06_classification/` (3-group tasks) |
