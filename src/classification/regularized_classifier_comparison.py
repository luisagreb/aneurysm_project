#!/usr/bin/env python3
"""
Regularized Classifier Comparison Script (SUPPLEMENTARY)
------------------------------------------------------
Purpose: To fix OVERFITTING seen in the Treated/Rescued tasks.
Strategy: Apply strict constraints (Regularization) to models to force them 
          to learn simple general rules instead of memorizing data.

Changes from original:
1. Random Forest: max_depth=3 (was 10/None), min_samples_leaf=3
2. XGBoost: max_depth=3, reg_alpha=1.0 (L1 regularization added)
3. MLP: hidden_layers=(10,) (simpler brain), high alpha (0.1 -> 1.0)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# Import helpers from the main script to avoid code duplication
# Assumes the original script is in the same folder
import sys
sys.path.append(str(Path(__file__).parent))
try:
    from comprehensive_classifier_comparison import (
        load_metadata, load_and_aggregate_features, prepare_task_data, 
        plot_task_comparison, plot_overall_comparison, create_summary_table,
        METADATA_FILE, FEATURES_FILE
    )
except ImportError:
    print("Could not import helpers. Please ensure comprehensive_classifier_comparison.py is in this folder.")
    sys.exit(1)

OUTPUT_DIR = Path('classification_results/regularized')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Explicitly override in case import failing to propogate
FEATURES_FILE = 'outputs/Advanced_Features_Raw_Final.csv'

def get_regularized_classifiers():
    """Return dictionary of STRICTLY REGULARIZED classifiers."""
    classifiers = {
        # Random Forest: Constrained depth and leaf size
        'Random Forest (Reg)': RandomForestClassifier(
            n_estimators=100, 
            max_depth=3,           # LIMIT: Can't grow deep trees
            min_samples_leaf=4,    # LIMIT: Leaves must have >=4 samples
            max_features='sqrt',
            random_state=42
        ),
        
        # Logistic Regression: High L2 regularization (C is small)
        'Logistic Regression (Reg)': Pipeline([
            ('scaler', StandardScaler()),
            ('lr', LogisticRegression(
                C=0.1,             # LIMIT: Strong Regularization
                max_iter=1000, 
                random_state=42
            ))
        ]),
        
        # MLP: Very simple network ("Small Brain")
        'MLP Classifier (Reg)': Pipeline([
            ('scaler', StandardScaler()),
            ('mlp', MLPClassifier(
                hidden_layer_sizes=(10,), # LIMIT: Only 10 neurons (was 50,25)
                alpha=2.0,                # LIMIT: Very high weight penalty
                max_iter=1000,
                random_state=42
            ))
        ]),
    }
    
    if XGBOOST_AVAILABLE:
        # XGBoost: Shallow trees + L1/L2 Regularization
        classifiers['XGBoost (Reg)'] = XGBClassifier(
            n_estimators=50,       # LIMIT: Fewer trees
            max_depth=2,           # LIMIT: Very shallow trees
            learning_rate=0.05,    # LIMIT: Slow learning
            reg_alpha=1.0,         # LIMIT: L1 Regularization (Lasso)
            reg_lambda=1.0,        # LIMIT: L2 Regularization (Ridge)
            subsample=0.7,         # LIMIT: Use only 70% of data per tree
            eval_metric='logloss',
            random_state=42,
            use_label_encoder=False
        )
    
    return classifiers

def evaluate_regularized_task(X, y, task_name):
    """Evaluate regularized classifiers and check for overfitting."""
    print(f"\n{'='*60}")
    print(f"EVALUATING (REGULARIZED): {task_name}")
    print('='*60)
    
    classifiers = get_regularized_classifiers()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    
    for name, clf in classifiers.items():
        print(f"  Training {name}...")
        try:
            # We need ALL metrics to satisfy the standard plot function
            scoring = ['accuracy', 'roc_auc', 'f1', 'precision', 'recall']
            cv_results = cross_validate(clf, X, y, cv=cv, scoring=scoring, 
                                      return_train_score=True)
            
            test_acc = cv_results['test_accuracy'].mean()
            train_acc = cv_results['train_accuracy'].mean()
            gap = train_acc - test_acc
            
            print(f"    Test Acc:  {test_acc:.1%} (Train: {train_acc:.1%})")
            print(f"    Gap:       {gap:.1%}", end="")
            
            if gap < 0.10: print(" ✅ (Generalized)")
            else: print(" ⚠️ (Still Overfitting)")
            
            results.append({
                'Task': task_name, 'Classifier': name,
                'Accuracy': test_acc, 
                'Accuracy_Std': cv_results['test_accuracy'].std(),
                'Train_Accuracy': train_acc, 
                'Overfit_Gap': gap,
                'AUC_ROC': cv_results['test_roc_auc'].mean(),
                'AUC_Std': cv_results['test_roc_auc'].std(),
                'F1_Score': cv_results['test_f1'].mean(),
                'F1_Std': cv_results['test_f1'].std(),
                'Precision': cv_results['test_precision'].mean(),
                'Precision_Std': cv_results['test_precision'].std(),
                'Recall': cv_results['test_recall'].mean(),
                'Recall_Std': cv_results['test_recall'].std(),
            })
            
        except Exception as e:
            print(f"    Error: {e}")
            continue
            
    return pd.DataFrame(results)

def main():
    print("="*60)
    print("REGULARIZED CLASSIFIER COMPARISON")
    print("Goal: Fix Overfitting in Treated/Rescued tasks")
    print("="*60)
    
    healthy_ids, taa_ids = load_metadata()
    X, df_agg = load_and_aggregate_features(healthy_ids, taa_ids)
    
    tasks = {
        'Diseased or Not': 'diseased', # Added back for complete comparison
        'Treated or Not': 'treated', 
        'Rescued or Not': 'rescued'
    }
    
    all_results = []
    
    for task_name, task_type in tasks.items():
        X_task, y_task = prepare_task_data(X, df_agg, task_type)
        if X_task is not None:
            res_df = evaluate_regularized_task(X_task, y_task, task_name)
            
            # Clean names for plotting (remove " (Reg)")
            res_df_clean = res_df.copy()
            res_df_clean['Classifier'] = res_df_clean['Classifier'].str.replace(r' \(Reg\)', '', regex=True)
            
            # Generate Standard Plot for this task
            plot_task_comparison(res_df_clean, task_name, OUTPUT_DIR)
            
            all_results.append(res_df_clean)
    
    if all_results:
        final_df = pd.concat(all_results)
        csv_path = OUTPUT_DIR / 'regularized_results.csv'
        final_df.to_csv(csv_path, index=False)
        print(f"\nSaved regularized results to: {csv_path}")
        
        # Plotting Results using STANDARD functions
        plot_overall_comparison(final_df, OUTPUT_DIR)
        create_summary_table(final_df, OUTPUT_DIR)

def plot_results_pastel(df):
    """Create beautiful pastel bar charts for the results."""
    print("\nGenerating Pastel Plots...")
    
    # Set style
    sns.set_theme(style="whitegrid")
    
    # Define nice pastel colors
    # distinct pastels: Blue, Orange, Green, Red
    colors = ["#AEC6CF", "#FFB347", "#77DD77", "#FF6961"]
    
    # Clean classifier names
    df_clean = df.copy()
    df_clean['Classifier'] = df_clean['Classifier'].str.replace(' \(Reg\)', '', regex=True)
    
    tasks = df_clean['Task'].unique()
    
    for task in tasks:
        task_data = df_clean[df_clean['Task'] == task]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot Accuracy
        sns.barplot(data=task_data, x='Classifier', y='Accuracy', ax=axes[0], palette=colors)
        axes[0].set_title(f'{task}: Accuracy', fontsize=14, fontweight='bold')
        axes[0].set_ylim(0, 1.05)
        for i, val in enumerate(task_data['Accuracy']):
            axes[0].text(i, val + 0.01, f'{val:.1%}', ha='center', fontsize=11)
            
        # Plot AUC
        if 'AUC' in task_data.columns:
            sns.barplot(data=task_data, x='Classifier', y='AUC', ax=axes[1], palette=colors)
            axes[1].set_title(f'{task}: AUC-ROC', fontsize=14, fontweight='bold')
            axes[1].set_ylim(0, 1.05)
            for i, val in enumerate(task_data['AUC']):
                axes[1].text(i, val + 0.01, f'{val:.3f}', ha='center', fontsize=11)
        
        plt.suptitle(f'Regularized Models Performance - {task}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save
        safe_name = task.replace(' ', '_').lower()
        save_path = OUTPUT_DIR / f'plot_{safe_name}_pastel.png'
        plt.savefig(save_path, dpi=300)
        print(f"  Saved plot: {save_path}")

    # Create Overall Summary Plot
    plot_overall_pastel(df)

def plot_overall_pastel(df):
    """Create a single combined summary plot."""
    print("\nGenerating Overall Pastel Summary...")
    sns.set_theme(style="whitegrid")
    
    tasks = df['Task'].unique()
    classifiers = df['Classifier'].unique()
    
    # Clean classifier names for plotting (remove ' (Reg)')
    df_clean = df.copy()
    df_clean['Classifier'] = df_clean['Classifier'].str.replace(' \(Reg\)', '', regex=True)
    
    # Custom Pastel Palette
    colors = ["#AEC6CF", "#FFB347", "#77DD77", "#FF6961"]
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    plt.suptitle("Overall Regularized Model Performance", fontsize=20, fontweight='bold', y=1.05)
    
    for i, task in enumerate(tasks):
        ax = axes[i]
        task_data = df_clean[df_clean['Task'] == task]
        
        sns.barplot(data=task_data, x='Classifier', y='Accuracy', ax=ax, palette=colors)
        
        ax.set_title(task, fontsize=16, fontweight='bold')
        ax.set_ylim(0, 1.05)
        ax.set_xlabel('')
        ax.tick_params(axis='x', rotation=45)
        
        # Add labels
        for j, val in enumerate(task_data['Accuracy']):
            ax.text(j, val + 0.02, f'{val:.1%}', ha='center', fontweight='bold')
            
    plt.tight_layout()
    output_path = OUTPUT_DIR / 'plot_overall_summary_pastel.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved overall summary: {output_path}")


if __name__ == "__main__":
    main()
