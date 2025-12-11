import nrrd
import numpy as np
import os
import glob
import pandas as pd
from scipy.stats import ttest_1samp
import matplotlib.pyplot as plt
import seaborn as sns

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

# --- Parsing Function (Specific to your Subject/Collagen structure) ---
def parse_experimental_factors(relative_path):
    """Extracts experimental factors from a file path."""
    
    parts = os.path.normpath(relative_path).split(os.sep)
    
    subject_id = "UNKNOWN"
    collagen_status = "UNKNOWN"

    for part in parts:
        if part.startswith("01Asc-"): # Crude way to identify Subject ID
            subject_id = part
        elif part.lower() == '+coll':
            collagen_status = "+coll"
        elif part.lower() == 'no coll':
            collagen_status = "no coll"
        # Extract cell ID from the filename or deepest folder path
        if "cell" in part.lower():
            # Extract the actual cell name (e.g., 'cell1')
            cell_id = part.split('cell')[-1].split('.')[0].strip()
            subject_id = f"{subject_id}_cell{cell_id}"

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
        
        # 1. Get relative path 
        relative_path = os.path.relpath(auto_path, auto_root)
            
        # 2. Derive the directory path in the MANUAL_MASKS_ROOT_DIR
        auto_dir = os.path.dirname(relative_path)
        
        # Construct the manual filename and path using the new fixed filename
        manual_path = os.path.join(manual_root, auto_dir, MANUAL_FILENAME)
        
        
        if not os.path.exists(manual_path):
            print(f"Skipping: Manual mask not found at expected location: {manual_path}")
            print(f"         Ensure the manual file is named '{MANUAL_FILENAME}' and is in the correct cell folder.")
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
            
            # 5. Extract Factors (using the directory structure which contains the subject and collagen status)
            subject_id_with_cell, collagen_status = parse_experimental_factors(relative_path)
            
            validation_results.append({
                'subject_id': subject_id_with_cell, # Now contains cell ID for unique plotting
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


# --- Plotting Function (FIXED) ---
def plot_results(df, subject_id):
    """Generates and saves visual plots of the validation metrics."""
    if df.empty:
        print("\nNo data to plot.")
        return

    # Melt the DataFrame for easier plotting of multiple metrics (Dice and Jaccard)
    plot_df = pd.melt(df, 
                      id_vars=['subject_id', 'collagen_status'], 
                      value_vars=['dice_coefficient', 'jaccard_index'],
                      var_name='Metric', 
                      value_name='Score')

    # Create a single figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'Segmentation Performance Metrics for Subject: {subject_id}', fontsize=16)

    # --- Subplot 1: Box Plot (Distribution by Collagen Status) ---
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


    # --- Subplot 2: Scatter Plot (Individual Cell Performance using HUE and STYLE) ---
    # This single call correctly maps the two categorical variables to aesthetics, generating the combined legend.
    sns.scatterplot(ax=axes[1], 
                    x='subject_id', 
                    y='Score', 
                    hue='collagen_status',     # Color distinguishes condition
                    style='Metric',            # Shape distinguishes metric (Dice vs Jaccard)
                    data=plot_df, 
                    s=100)


    axes[1].set_title('Cell-by-Cell Performance')
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_ylabel('Segmentation Score (0.0 - 1.0)')
    axes[1].set_xlabel('Individual Cell ID')
    axes[1].axhline(0.90, color='r', linestyle='--', linewidth=1)
    
    # Rotate x-axis labels for readability if many cells
    axes[1].tick_params(axis='x', rotation=45)
    # Seaborn automatically creates the combined legend 
    axes[1].legend(title='Condition & Metric', bbox_to_anchor=(1.05, 1), loc='upper left')

    
    plt.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust layout to prevent title overlap
    
    # Save the plot
    plot_filename = f"validation_plot_{subject_id}.png"
    plt.savefig(plot_filename)
    print(f"\nVisual report saved to {plot_filename}")


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
    # Ensure you have the necessary libraries installed: pip install pandas numpy nrrd scipy matplotlib seaborn
    
    # 1. Run the comparison on the filtered data
    results_df = run_validation(
        auto_root=AUTO_MASKS_ROOT_DIR,
        manual_root=MANUAL_MASKS_ROOT_DIR,
        subject_filter=TARGET_SUBJECT_ID
    )

    # 2. Generate and print the statistical report
    generate_report(results_df, TARGET_SUBJECT_ID) 
    
    # 3. Generate and save the visualization
    plot_results(results_df, TARGET_SUBJECT_ID)