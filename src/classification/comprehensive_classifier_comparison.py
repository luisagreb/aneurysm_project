#!/usr/bin/env python3
"""
Comprehensive Classifier Comparison Script (CORRECTED DATA FILTERING)
Compares Random Forest, XGBoost, MLP Classifier, and Linear Regression
across three classification tasks:

1. Diseased or not: Healthy-NoColl vs TAA-NoColl (pure disease effect)
2. Treated or not: TAA-NoColl vs TAA+Coll (treatment effect on diseased cells)
3. Rescued or not: TAA+Coll phenotypically closer vs further from healthy

Data Filtering Strategy:
- Task 1: ONLY Healthy-NoColl and TAA-NoColl (no collagen samples)
- Task 2: ONLY TAA samples (with and without collagen)
- Task 3: ONLY TAA+Coll samples (classified by rescue status)

Outputs: Detailed comparison plots, metrics tables, and performance analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import warnings
from pathlib import Path
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, 
    roc_auc_score, 
    f1_score, 
    precision_score, 
    recall_score
)

# Try to import XGBoost (might not be installed)
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("WARNING: XGBoost not installed. Skipping XGBoost classifier.")

warnings.filterwarnings('ignore')

# Configuration
FEATURES_FILE = 'outputs/Advanced_Features_Raw.csv'
METADATA_FILE = 'data/Book1.xlsx'
OUTPUT_DIR = Path('classification_results/comprehensive_comparison')


def extract_subject_id(filename):
    """Extract numeric subject ID from filename."""
    if pd.isna(filename):
        return None
    text = str(filename).strip()
    first_token = text.split(' ')[0].strip('-')
    parts = first_token.split('-')
    if len(parts) > 1:
        last_part = parts[-1]
        if last_part.isdigit():
            return int(last_part)
        sub_digits = re.findall(r'\d+', last_part)
        if sub_digits:
            return int(sub_digits[0])
    digits = re.findall(r'\d+', first_token)
    if digits:
        return int(digits[-1])
    return None


def extract_collagen_status(filename):
    """Extract collagen status from filename: 1=+Coll, 0=-Coll/NoColl."""
    if pd.isna(filename):
        return None
    filename = str(filename).lower()
    if '+coll' in filename or '+col' in filename:
        return 1  # Collagen
    elif '-coll' in filename or 'nocoll' in filename or 'no coll' in filename:
        return 0  # No Collagen
    return None


def normalize_filename(f):
    """Normalize filename for cell-level grouping."""
    s = str(f).replace('.nii.gz', '').replace('.tif', '')
    s = s.replace('_segmentation', '').replace('_visible', '')
    return s.strip()


def load_metadata():
    """Load metadata: Healthy vs TAA subject IDs."""
    print("Loading metadata from Excel file...")
    
    df_meta = pd.read_excel(METADATA_FILE, header=None, skiprows=18)
    
    healthy_ids = set()
    taa_ids = set()
    
    for x in df_meta[0].dropna():
        nid = extract_subject_id(str(x))
        if nid:
            healthy_ids.add(nid)
    
    for x in df_meta[1].dropna():
        nid = extract_subject_id(str(x))
        if nid:
            taa_ids.add(nid)
    
    print(f"  Healthy IDs: {sorted(healthy_ids)}")
    print(f"  TAA IDs: {sorted(taa_ids)}")
    
    return healthy_ids, taa_ids


def load_and_aggregate_features(healthy_ids, taa_ids):
    """
    Load features and create 4-class labels:
    0 = Healthy-NoColl
    1 = Healthy+Coll
    2 = TAA-NoColl
    3 = TAA+Coll
    """
    print(f"\n{'='*60}")
    print("Loading and Processing Features")
    print('='*60)
    
    # Load features
    df = pd.read_csv(FEATURES_FILE)
    
    # Extract IDs and collagen status
    df['Numeric_ID'] = df['CellName'].apply(extract_subject_id)
    df['Collagen'] = df['CellName'].apply(extract_collagen_status)
    
    # Drop rows where we couldn't determine ID or collagen status
    df = df.dropna(subset=['Numeric_ID', 'Collagen'])
    
    # Assign disease status: 0=Healthy, 1=TAA
    def get_disease_status(nid):
        if nid in healthy_ids:
            return 0  # Healthy
        if nid in taa_ids:
            return 1  # TAA
        return None
    
    df['Disease'] = df['Numeric_ID'].apply(get_disease_status)
    df = df.dropna(subset=['Disease'])
    
    # Create 4-class label: Disease*2 + Collagen
    # 0*2+0=0 (Healthy-NoColl), 0*2+1=1 (Healthy+Coll)
    # 1*2+0=2 (TAA-NoColl), 1*2+1=3 (TAA+Coll)
    df['FourClass'] = (df['Disease'] * 2 + df['Collagen']).astype(int)
    
    # Cell-level aggregation
    df['Cell_ID'] = df['CellName'].apply(normalize_filename)
    
    feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in ['Numeric_ID', 'Collagen', 'Disease', 'FourClass']:
        if col in feature_cols:
            feature_cols.remove(col)
    
    agg_dict = {col: 'max' for col in feature_cols}
    agg_dict.update({'FourClass': 'first', 'Disease': 'first', 'Collagen': 'first'})
    
    df_agg = df.groupby('Cell_ID').agg(agg_dict)
    
    # Prepare features
    X = df_agg[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    print(f"\nTotal cells: {len(X)}")
    print(f"  Healthy-NoColl: {sum(df_agg['FourClass'] == 0)}")
    print(f"  Healthy+Coll:   {sum(df_agg['FourClass'] == 1)}")
    print(f"  TAA-NoColl:     {sum(df_agg['FourClass'] == 2)}")
    print(f"  TAA+Coll:       {sum(df_agg['FourClass'] == 3)}")
    
    return X, df_agg


def prepare_task_data(X, df_agg, task_type):
    """
    Prepare data for specific classification task with proper filtering.
    
    Args:
        X: Feature dataframe (all cells)
        df_agg: Metadata with FourClass labels
        task_type: 'diseased', 'treated', or 'rescued'
    
    Returns:
        X_filtered, y_filtered
    """
    print(f"\n{'='*60}")
    print(f"Preparing data for: {task_type.upper()}")
    print('='*60)
    
    if task_type == 'diseased':
        # Task 1: Healthy-NoColl (0) vs TAA-NoColl (2)
        # ONLY untreated samples to isolate disease effect
        mask = df_agg['FourClass'].isin([0, 2])
        X_task = X[mask]
        y_task = (df_agg.loc[mask, 'FourClass'] == 2).astype(int)  # 0=Healthy, 1=TAA
        
        print(f"  Comparing: Healthy-NoColl vs TAA-NoColl")
        print(f"  Filters: Only cells with NO collagen treatment")
        print(f"  Classes: 0=Healthy (n={sum(y_task==0)}), 1=TAA (n={sum(y_task==1)})")
    
    elif task_type == 'treated':
        # Task 2: TAA-NoColl (2) vs TAA+Coll (3)
        # ONLY TAA samples to isolate treatment effect
        mask = df_agg['FourClass'].isin([2, 3])
        X_task = X[mask]
        y_task = (df_agg.loc[mask, 'FourClass'] == 3).astype(int)  # 0=NoColl, 1=+Coll
        
        print(f"  Comparing: TAA-NoColl vs TAA+Coll")
        print(f"  Filters: Only TAA (diseased) cells")
        print(f"  Classes: 0=No Treatment (n={sum(y_task==0)}), 1=+Collagen (n={sum(y_task==1)})")
    
    elif task_type == 'rescued':
        # Task 3: Rescued (TAA+Coll closer to Healthy) vs Not Rescued
        # Calculate distance of each TAA+Coll cell to Healthy centroid
        
        healthy_mask = df_agg['FourClass'].isin([0, 1])  # Both healthy groups
        taa_coll_mask = df_agg['FourClass'] == 3  # TAA+Coll
        
        # Standardize features for distance calculation
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)
        
        # Compute healthy centroid
        healthy_centroid = X_scaled_df[healthy_mask].mean().values
        
        # Compute distance for each TAA+Coll cell
        taa_coll_cells = X_scaled_df[taa_coll_mask]
        distances = np.linalg.norm(taa_coll_cells.values - healthy_centroid, axis=1)
        
        # Classify by median: below median = rescued (0), above = not rescued (1)
        median_distance = np.median(distances)
        y_task = (distances > median_distance).astype(int)
        X_task = X[taa_coll_mask]
        
        print(f"  Comparing: TAA+Coll cells by rescue status")
        print(f"  Filters: Only TAA+Collagen cells")
        print(f"  Classification: Distance to Healthy centroid")
        print(f"  Classes: 0=Rescued/Closer (n={sum(y_task==0)}), 1=Not Rescued/Further (n={sum(y_task==1)})")
        print(f"  Median distance threshold: {median_distance:.3f}")
    
    else:
        raise ValueError(f"Unknown task_type: {task_type}")
    
    # Ensure we have both classes
    if len(np.unique(y_task)) < 2:
        print(f"  WARNING: Only one class present! Cannot train classifier.")
        return None, None
    
    return X_task, y_task


def get_classifiers():
    """Return dictionary of classifiers to compare."""
    classifiers = {
        'Random Forest': RandomForestClassifier(
            n_estimators=100, 
            random_state=42,
            max_depth=10
        ),
        'Logistic Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('lr', LogisticRegression(max_iter=1000, random_state=42))
        ]),
        'MLP Classifier': Pipeline([
            ('scaler', StandardScaler()),
            ('mlp', MLPClassifier(
                hidden_layer_sizes=(50, 25), 
                max_iter=1000,
                solver='adam', 
                learning_rate_init=0.001,
                alpha=0.1, 
                early_stopping=False, 
                random_state=42
            ))
        ]),
    }
    
    if XGBOOST_AVAILABLE:
        classifiers['XGBoost'] = XGBClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=6,
            learning_rate=0.1,
            eval_metric='logloss'
        )
    
    return classifiers


def evaluate_single_task(X, y, task_name):
    """Evaluate all classifiers on a single classification task."""
    print(f"\n{'='*60}")
    print(f"EVALUATING: {task_name}")
    print('='*60)
    
    if X is None or y is None:
        print("  Skipping - insufficient data")
        return pd.DataFrame()
    
    classifiers = get_classifiers()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    results = []
    
    for name, clf in classifiers.items():
        print(f"\n  Training {name}...")
        
        try:
            # Use cross_validate to get BOTH training and test scores
            # This allows us to check for OVERFITTING
            from sklearn.model_selection import cross_validate
            
            scoring = ['accuracy', 'f1', 'roc_auc']
            cv_results = cross_validate(clf, X, y, cv=cv, scoring=scoring, 
                                      return_train_score=True)
            
            # Extract scores
            test_acc = cv_results['test_accuracy'].mean()
            train_acc = cv_results['train_accuracy'].mean()
            test_acc_std = cv_results['test_accuracy'].std()
            
            test_f1 = cv_results['test_f1'].mean()
            train_f1 = cv_results['train_f1'].mean()
            test_f1_std = cv_results['test_f1'].std()
            
            test_auc = cv_results['test_roc_auc'].mean()
            train_auc = cv_results['train_roc_auc'].mean()
            test_auc_std = cv_results['test_roc_auc'].std()
            
            # Calculate Overfitting Gap (Train - Test)
            overfit_gap = train_acc - test_acc
            
            results.append({
                'Task': task_name,
                'Classifier': name,
                'Accuracy': test_acc,
                'Accuracy_Std': test_acc_std,
                'Train_Accuracy': train_acc, # Extracted for checking
                'Overfit_Gap': overfit_gap,
                'F1_Score': test_f1,
                'F1_Std': test_f1_std,
                'AUC_ROC': test_auc,
                'AUC_Std': test_auc_std,
                # Placeholders for simple bar plots to work without modification
                'Precision': 0, 'Precision_Std': 0, 'Recall': 0, 'Recall_Std': 0
            })
            
            print(f"    Test Accuracy:  {test_acc:.1%} ± {test_acc_std:.1%}")
            print(f"    Train Accuracy: {train_acc:.1%}")
            print(f"    Overfit Gap:    {overfit_gap:.1%}", end="")
            
            if overfit_gap > 0.15: # If gap > 15%, likely overfitting
                print(" 🚨 (HIGH OVERFITTING)")
            elif overfit_gap > 0.05:
                print(" ⚠️ (Mild Overfitting)")
            else:
                print(" ✅ (Good Generalization)")
                
            print(f"    AUC-ROC:        {test_auc:.3f}")
            
        except Exception as e:
            print(f"    ERROR: {str(e)}")
            continue
    
    return pd.DataFrame(results)


def plot_task_comparison(results_df, task_name, output_dir):
    """Create comparison visualization for a single task."""
    if results_df.empty:
        return
    
    # Sort by accuracy
    results_df = results_df.sort_values('Accuracy', ascending=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Classifier Comparison: {task_name}\n(5-Fold Cross-Validation)', 
                 fontsize=18, fontweight='bold', y=0.995)
    
    # Color palette (Pastel)
    # Blue, Orange, Green, Red
    colors = ["#AEC6CF", "#FFB347", "#77DD77", "#FF6961"]
    
    # Ensure we have enough colors if there are more classifiers
    if len(results_df) > len(colors):
        # Fallback to extending the palette if needed
        colors = colors * (len(results_df) // len(colors) + 1)
    colors = colors[:len(results_df)]
    
    metrics = [
        ('Accuracy', 'Accuracy_Std', axes[0, 0]),
        ('F1_Score', 'F1_Std', axes[0, 1]),
        ('Precision', 'Precision_Std', axes[0, 2]),
        ('Recall', 'Recall_Std', axes[1, 0]),
        ('AUC_ROC', 'AUC_Std', axes[1, 1])
    ]
    
    for metric, std, ax in metrics:
        ax.barh(results_df['Classifier'], results_df[metric], 
                xerr=results_df[std], color=colors, capsize=3)
        ax.set_xlabel(metric.replace('_', ' '), fontsize=12)
        ax.set_title(f'{metric.replace("_", " ")} Comparison', fontsize=13, fontweight='bold')
        ax.set_xlim(0, 1.0)
        
        if metric == 'Accuracy':
            ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Random')
            ax.legend()
        
        # Add value labels
        for i, (val, std_val) in enumerate(zip(results_df[metric], results_df[std])):
            ax.text(val + std_val + 0.02, i, f'{val:.1%}', va='center', fontsize=9)
    
    # Summary table in the last subplot
    ax_table = axes[1, 2]
    ax_table.axis('off')
    
    table_data = []
    for _, row in results_df.iterrows():
        table_data.append([
            row['Classifier'],
            f"{row['Accuracy']:.1%}",
            f"{row['F1_Score']:.1%}",
            f"{row['AUC_ROC']:.1%}"
        ])
    
    table = ax_table.table(
        cellText=table_data,
        colLabels=['Classifier', 'Accuracy', 'F1', 'AUC'],
        cellLoc='center',
        loc='center',
        colWidths=[0.35, 0.2, 0.2, 0.2]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style table header
    for i in range(4):
        table[(0, i)].set_facecolor('#3498DB')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.tight_layout()
    
    output_path = output_dir / f'{task_name.lower().replace(" ", "_")}_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nSaved plot: {output_path}")
    plt.close()


def plot_overall_comparison(all_results_df, output_dir):
    """Create overall comparison across all tasks."""
    if all_results_df.empty:
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('Overall Classifier Performance Across All Tasks', 
                 fontsize=18, fontweight='bold')
    
    tasks = all_results_df['Task'].unique()
    classifiers = all_results_df['Classifier'].unique()
    
    metrics = ['Accuracy', 'F1_Score', 'AUC_ROC']
    titles = ['Accuracy', 'F1 Score', 'AUC-ROC']
    
    # Custom Pastel Palette
    # Aligns with: Random Forest (Blue), Logistic Regression (Orange), MLP (Green), XGBoost (Red)
    # Note: The order depends on 'classifiers' array order. 
    # To Ensure consistency, we map specific names if possible, but for now we iterate.
    palette = ["#AEC6CF", "#FFB347", "#77DD77", "#FF6961"]
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx]
        
        x = np.arange(len(tasks))
        width = 0.8 / len(classifiers)
        
        for i, clf in enumerate(classifiers):
            color = palette[i % len(palette)]
            
            clf_data = all_results_df[all_results_df['Classifier'] == clf]
            values = [clf_data[clf_data['Task'] == task][metric].values[0] 
                     if len(clf_data[clf_data['Task'] == task]) > 0 else 0 
                     for task in tasks]
            
            ax.bar(x + i * width, values, width, label=clf, alpha=0.9, color=color)
        
        ax.set_xlabel('Classification Task', fontsize=12)
        ax.set_ylabel(title, fontsize=12)
        ax.set_title(f'{title} Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * (len(classifiers) - 1) / 2)
        ax.set_xticklabels(tasks, rotation=15, ha='right')
        ax.legend()
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    output_path = output_dir / 'overall_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nSaved overall comparison: {output_path}")
    plt.close()


def create_summary_table(all_results_df, output_dir):
    """Create and save summary table."""
    if all_results_df.empty:
        return
    
    # Pivot table for easy viewing
    pivot_accuracy = all_results_df.pivot(index='Classifier', columns='Task', values='Accuracy')
    pivot_f1 = all_results_df.pivot(index='Classifier', columns='Task', values='F1_Score')
    pivot_auc = all_results_df.pivot(index='Classifier', columns='Task', values='AUC_ROC')
    
    # Save to Excel with multiple sheets
    excel_path = output_dir / 'comprehensive_results.xlsx'
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        all_results_df.to_excel(writer, sheet_name='All Results', index=False)
        pivot_accuracy.to_excel(writer, sheet_name='Accuracy')
        pivot_f1.to_excel(writer, sheet_name='F1 Score')
        pivot_auc.to_excel(writer, sheet_name='AUC-ROC')
    
    print(f"\nSaved comprehensive results: {excel_path}")
    
    # Also save as CSV
    csv_path = output_dir / 'comprehensive_results.csv'
    all_results_df.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")


def print_summary(all_results_df):
    """Print summary of best performers."""
    if all_results_df.empty:
        return
    
    print("\n" + "="*60)
    print("SUMMARY: BEST PERFORMERS")
    print("="*60)
    
    for task in all_results_df['Task'].unique():
        task_data = all_results_df[all_results_df['Task'] == task]
        best = task_data.loc[task_data['Accuracy'].idxmax()]
        
        print(f"\n{task}:")
        print(f"  Best: {best['Classifier']}")
        print(f"    Accuracy: {best['Accuracy']:.1%} ± {best['Accuracy_Std']:.1%}")
        print(f"    F1 Score: {best['F1_Score']:.1%} ± {best['F1_Std']:.1%}")
        print(f"    AUC-ROC:  {best['AUC_ROC']:.1%} ± {best['AUC_Std']:.1%}")


def main():
    print("="*60)
    print("COMPREHENSIVE CLASSIFIER COMPARISON")
    print("Random Forest | XGBoost | MLP | Logistic Regression")
    print("="*60)
    print("\nData Filtering Strategy:")
    print("  Task 1 (Diseased): Healthy-NoColl vs TAA-NoColl")
    print("  Task 2 (Treated):  TAA-NoColl vs TAA+Coll")
    print("  Task 3 (Rescued):  TAA+Coll by distance to Healthy")
    print("="*60)
    
    if not XGBOOST_AVAILABLE:
        print("\nNOTE: Install XGBoost with: pip install xgboost")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load metadata
    healthy_ids, taa_ids = load_metadata()
    
    # Load and aggregate features
    X, df_agg = load_and_aggregate_features(healthy_ids, taa_ids)
    
    # Define classification tasks
    tasks = {
        'Diseased or Not': 'diseased',
        'Treated or Not': 'treated',
        'Rescued or Not': 'rescued'
    }
    
    all_results = []
    
    # Evaluate each task
    for task_name, task_type in tasks.items():
        X_task, y_task = prepare_task_data(X, df_agg, task_type)
        
        if X_task is None or y_task is None:
            print(f"\nSkipping {task_name} - insufficient data")
            continue
        
        # Evaluate classifiers
        task_results = evaluate_single_task(X_task, y_task, task_name)
        
        if not task_results.empty:
            all_results.append(task_results)
            
            # Create task-specific plot
            plot_task_comparison(task_results, task_name, OUTPUT_DIR)
    
    # Combine all results
    if all_results:
        all_results_df = pd.concat(all_results, ignore_index=True)
        
        # Create overall comparison plot
        plot_overall_comparison(all_results_df, OUTPUT_DIR)
        
        # Create summary tables
        create_summary_table(all_results_df, OUTPUT_DIR)
        
        # Print summary
        print_summary(all_results_df)
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print(f"Results saved to: {OUTPUT_DIR}")
        print("="*60)
    else:
        print("\nNo results generated - check your data!")


if __name__ == '__main__':
    main()
