import nrrd
import numpy as np
import os
import glob
import pandas as pd
import re # Added for robust cell ID extraction
from scipy.stats import ttest_1samp
import matplotlib.pyplot as plt # Re-added for plotting
import seaborn as sns # Re-added for plotting

# Set global aesthetic style for plots
sns.set_theme(style="whitegrid")

# --- Configuration ---
# The folder where your automated masks are saved (output of segmentation_pipeline.py)
AUTO_MASKS_ROOT_DIR = "/Users/luisagrebici/Documents/Nezami_Lab/aneurysm_project/data/segmented_auto_nucleus/" 
# The folder where your manually segmented (ground truth) masks are stored
MANUAL_MASKS_ROOT_DIR = "/Users/luisagrebici/Documents/Nezami_Lab/aneurysm_project/data/segmented_training/" 

# --- Validation Filter ---
TARGET_SUBJECT_ID = "01Asc-180"

# --- Filename Suffixes (CRITICAL: NOW CORRECTED BASED ON YOUR EXAMPLE) ---
# Automated mask name structure (e.g., '01ASC-0180 ... cell1.nucleus.seg.nrrd')
AUTO_SUFFIX = ".nucleus.seg.nrrd"
# Manual mask name is a fixed file name within each cell's folder:
MANUAL_FILENAME = "Segmentation.nucleus.seg.nrrd" 

# --- Metrics Calculation ---
def calculate_dice_jaccard(auto_mask, manual_mask):
    """Calculates Dice and Jaccard indices (IoU) for two binary masks."""
    
    # Ensure both masks are binary (0/1) for clean calculation
    auto_mask_binary = (auto_mask > 0).astype(np.int8)
    manual_mask_binary = (manual_mask > 0).astype(np.int8)

    # Intersection (True Positives)
    intersection = np.sum(auto_mask_binary * manual_mask_binary)
    
    # Union (TP + FP + FN)
    union = np.sum(auto_mask_binary) + np.sum(manual_mask_binary) - intersection
    
    # Dice Coefficient (F1 Score): 2 * TP / (2*TP + FP + FN) = 2 * Intersection / (Sum of volumes)
    dice = (2.0 * intersection) / (np.sum(auto_mask_binary) + np.sum(manual_mask_binary))
    
    # Jaccard Index (IoU): TP / (TP + FP + FN) = Intersection / Union
    jaccard = intersection / union if union > 0 else 0.0

    return dice, jaccard

# --- Parsing Function (Fixed to be robust to directory depth) ---
def parse_experimental_factors(relative_path):
    """Extracts subject ID and collagen status from path components by searching."""
    
    parts = os.path.normpath(relative_path).split(os.sep)
    
    # Search for Subject ID (e.g., '01Asc-180')
    subject_id = next((p for p in parts if p.startswith("01Asc-")), "UNKNOWN")
    
    # Search for Collagen Status (e.g., '+coll' or 'no coll')
    collagen_status = next((p.lower() for p in parts if p.lower() in ('+coll', 'no coll')), "UNKNOWN")
    
    return subject_id, collagen_status

# --- Main Validation Logic ---
def run_validation(auto_root, manual_root, subject_filter):
    
    validation_results = []
    
    # Recursively find all automated NRRD files
    search_path = os.path.join(auto_root, "**", f"*{AUTO_SUFFIX}")
    auto_mask_paths = glob.glob(search_path, recursive=True)

    if not auto_mask_paths:
        print(f"Error: No automated NRRD masks found matching suffix '{AUTO_SUFFIX}' in {auto_root}. Check your path and file naming.")
        return pd.DataFrame()

    filtered_paths = [p for p in auto_mask_paths if subject_filter in p]
    print(f"Found {len(auto_mask_paths)} automated masks. Filtering for subject: {subject_filter}. Using {len(filtered_paths)} files.")

    for auto_path in filtered_paths:
        
        # 1. Get relative path (e.g., '01Asc-180/+coll/01ASC-0180 ... cell1.nucleus.seg.nrrd')
        relative_path = os.path.relpath(auto_path, auto_root)
        auto_dir = os.path.dirname(relative_path)
        
        # 2. Derive the manual mask path
        manual_path = os.path.join(manual_root, auto_dir, MANUAL_FILENAME)
        
        
        if not os.path.exists(manual_path):
            print(f"Skipping: Manual mask not found at expected location: {manual_path}")
            continue

        try:
            # 3. Load Masks
            auto_data, _ = nrrd.read(auto_path)
            manual_data, _ = nrrd.read(manual_path)
            
            # Ensure dimensions match
            if auto_data.shape != manual_data.shape:
                print(f"Skipping: Dimension mismatch for {relative_path}")
                continue
            
            # 4. Calculate Metrics
            dice, jaccard = calculate_dice_jaccard(auto_data, manual_data)
            
            # 5. Extract Factors 
            subject_id_base, collagen_status = parse_experimental_factors(relative_path)
            
            # Extract the unique cell identifier (e.g., 'cell1') from the folder name
            cell_folder_name = os.path.basename(auto_dir)
            cell_id_match = re.search(r'(cell\d+)', cell_folder_name, re.IGNORECASE)
            cell_id_suffix = cell_id_match.group(0).lower() if cell_id_match else "unknown_cell"
            
            # Create a unique ID suitable for plotting X-axis (e.g., '01Asc-180_cell1')
            unique_cell_plot_id = f"{subject_id_base}_{cell_id_suffix}"
            
            validation_results.append({
                'subject_id': unique_cell_plot_id, # Unique cell ID for plotting X-axis
                'collagen_status': collagen_status,
                'file_path': relative_path,
                'dice_coefficient': dice,
                'jaccard_index': jaccard,
                'auto_volume_voxels': np.sum(auto_data > 0),
                'manual_volume_voxels': np.sum(manual_data > 0)
            })
            print(f"Validated: {relative_path}")

        except Exception as e:
            print(f"Error processing {relative_path}: {e}")

    return pd.DataFrame(validation_results)

# --- Plotting Function (User's New Functions) ---
def make_plots(df: pd.DataFrame, subject_id: str, outdir: str = "."):
    """Generates four diagnostic plots and saves them as PNG files."""
    if df.empty:
        print("No data to plot.")
        return

    os.makedirs(outdir, exist_ok=True)
    
    print("\n--- Generating Plots ---")
    
    # Re-melt the DataFrame for the first summary plot (like the old plot_results)
    plot_df = pd.melt(df, 
                      id_vars=['subject_id', 'collagen_status'], 
                      value_vars=['dice_coefficient', 'jaccard_index'],
                      var_name='Metric', 
                      value_name='Score')

    # ---------- 1) Box Plot and Scatter Plot (Combined view for distribution and cell-by-cell) ----------
    fig1, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig1.suptitle(f'Segmentation Performance Metrics for Subject: {subject_id}', fontsize=16)

    # Box Plot
    sns.boxplot(ax=axes[0], 
                x='collagen_status', 
                y='Score', 
                hue='Metric', 
                data=plot_df, 
                palette={"dice_coefficient": "darkorange", "jaccard_index": "dodgerblue"})
    axes[0].set_title('Metric Distribution by Collagen Status')
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_ylabel('Segmentation Score (0.0 - 1.0)')
    axes[0].set_xlabel('Collagen Condition')
    axes[0].axhline(0.90, color='r', linestyle='--', linewidth=1, label='Target Dice (0.90)')
    axes[0].legend(loc='lower right')
    
    # Scatter Plot
    sns.scatterplot(ax=axes[1], 
                    x='subject_id', 
                    y='Score', 
                    hue='collagen_status',     
                    style='Metric',            
                    data=plot_df, 
                    s=100)

    axes[1].set_title('Cell-by-Cell Performance')
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_ylabel('Segmentation Score (0.0 - 1.0)')
    axes[1].set_xlabel('Individual Cell ID')
    axes[1].axhline(0.90, color='r', linestyle='--', linewidth=1)
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].legend(title='Condition & Metric', bbox_to_anchor=(1.05, 1), loc='upper left')

    fig1.tight_layout(rect=[0, 0, 1, 0.95])
    fig1.savefig(os.path.join(outdir, f"{subject_id}_performance_overview.png"), dpi=200)
    plt.close(fig1)

    # ---------- 2) Distributions (histograms) ----------
    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(10,4))
    ax2a.hist(df["dice_coefficient"], bins=12, color='darkorange', edgecolor='black')
    ax2a.set_title("Dice distribution"); ax2a.set_xlim(0,1.0); ax2a.set_xlabel("Dice"); ax2a.set_ylabel("Count")
    ax2b.hist(df["jaccard_index"], bins=12, color='dodgerblue', edgecolor='black')
    ax2b.set_title("Jaccard distribution"); ax2b.set_xlim(0,1.0); ax2b.set_xlabel("Jaccard")
    fig2.suptitle(f"{subject_id} – Score Distributions")
    fig2.tight_layout()
    fig2.savefig(os.path.join(outdir, f"{subject_id}_histograms.png"), dpi=200)
    plt.close(fig2)

    # ---------- 3) Dice vs manual volume ----------
    fig3, ax3 = plt.subplots(figsize=(6,4))
    sns.scatterplot(x="manual_volume_voxels", y="dice_coefficient", hue="collagen_status", data=df, ax=ax3, s=80)
    ax3.set_xlabel("Manual Volume (voxels)")
    ax3.set_ylabel("Dice Coefficient")
    ax3.set_title(f"{subject_id} – Dice vs Manual Volume")
    ax3.set_ylim(0,1.05)
    fig3.tight_layout()
    fig3.savefig(os.path.join(outdir, f"{subject_id}_dice_vs_manual_volume.png"), dpi=200)
    plt.close(fig3)

    # ---------- 4) Auto vs manual volume ----------
    fig4, ax4 = plt.subplots(figsize=(6,6))
    sns.scatterplot(x="manual_volume_voxels", y="auto_volume_voxels", hue="collagen_status", data=df, ax=ax4, s=80)
    
    lo = 0
    hi = max(df["manual_volume_voxels"].max(), df["auto_volume_voxels"].max()) * 1.05
    ax4.plot([lo, hi], [lo, hi], 'r--', label='Ideal Agreement')
    
    ax4.set_xlim(lo, hi); ax4.set_ylim(lo, hi)
    ax4.set_xlabel("Manual Volume (voxels)")
    ax4.set_ylabel("Auto Volume (voxels)")
    ax4.set_title(f"{subject_id} – Volume Agreement")
    ax4.legend()
    fig4.tight_layout()
    fig4.savefig(os.path.join(outdir, f"{subject_id}_auto_vs_manual_volume.png"), dpi=200)
    plt.close(fig4)

    print("Saved plots:",
          f"\n - {subject_id}_performance_overview.png (Replaced the old plot)",
          f"\n - {subject_id}_histograms.png",
          f"\n - {subject_id}_dice_vs_manual_volume.png",
          f"\n - {subject_id}_auto_vs_manual_volume.png")


# --- Statistical Reporting ---
def generate_report(df, subject_id):
    if df.empty:
        print("\nNo data found for reporting.")
        return

    print(f"\n############################################################")
    print(f"### Statistical Validation Report for Subject: {subject_id} ###")
    print(f"############################################################")

    # Averages
    summary = df[['dice_coefficient', 'jaccard_index']].agg(['mean', 'std', 'min', 'max']).T
    summary['cv'] = summary['std'] / summary['mean'] * 100
    
    print("\n--- Summary Statistics (Mean, Std Dev, Min, Max, CV%) ---")
    print(summary.round(4))

    # T-Test (Comparing Dice against an ideal standard, e.g., Dice=0.90)
    IDEAL_DICE_THRESHOLD = 0.90
    
    # Only run if enough samples exist
    if len(df) >= 2:
        # T-test checks if the sample mean is statistically different from the ideal threshold
        t_stat, p_value = ttest_1samp(df['dice_coefficient'], IDEAL_DICE_THRESHOLD)
        
        print(f"\n--- One-Sample T-Test (against Dice = {IDEAL_DICE_THRESHOLD}) ---")
        print(f"Hypothesis: Automated segmentation is significantly different from ideal 90% overlap.")
        print(f"T-Statistic: {t_stat:.4f}")
        print(f"P-Value: {p_value:.4f}")

        if p_value < 0.05:
            print(f"\nConclusion: The average Dice score ({summary.loc['dice_coefficient', 'mean']:.4f}) is statistically different from {IDEAL_DICE_THRESHOLD} (p={p_value:.4f}).")
            if summary.loc['dice_coefficient', 'mean'] < IDEAL_DICE_THRESHOLD:
                print("Your segmentation tends to be less accurate than the threshold.")
            else:
                print("Your segmentation tends to be more accurate than the threshold (though typically we aim for p > 0.05 to show no significant difference).")
        else:
            print(f"\nConclusion: The average Dice score is NOT statistically different from {IDEAL_DICE_THRESHOLD} (p={p_value:.4f}). This indicates the automated method performs comparably to the desired standard.")
            
    # Save the detailed results for later plotting
    output_filename = f"validation_report_{subject_id}.csv"
    df.to_csv(output_filename, index=False)
    print(f"\nDetailed results saved to {output_filename}")


if __name__ == "__main__":
    
    # 1. Run the comparison on the filtered data
    results_df = run_validation(
        auto_root=AUTO_MASKS_ROOT_DIR,
        manual_root=MANUAL_MASKS_ROOT_DIR,
        subject_filter=TARGET_SUBJECT_ID
    )

    # 2. Generate and print the statistical report (NOW CORRECTLY PLACED)
    generate_report(results_df, TARGET_SUBJECT_ID) 

    # 3. Generate and save the new set of visualizations
    make_plots(results_df, TARGET_SUBJECT_ID)