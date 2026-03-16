#!/usr/bin/env python3
"""
Comprehensive Classifier Comparison - 3 Groups (Healthy, TAA, BAV)
Compares Random Forest, XGBoost, MLP Classifier, and Logistic Regression
across the updated classification tasks.

Data Filtering Strategy:
- Task 1: TAA vs Healthy (No Collagen)
- Task 2: BAV vs Healthy (No Collagen)
- Task 3: TAA vs BAV (No Collagen)
- Task 4: TAA Treatment Effect (NoCollagen vs Collagen)
- Task 5: BAV Treatment Effect (NoCollagen vs Collagen)
- Task 6: Healthy Treatment Effect (NoCollagen vs Collagen)

Outputs: Detailed comparison plots, metrics tables, and performance analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.metrics import confusion_matrix
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
    print("WARNING: XGBoost not installed. Skipping XGBoost classifier.")

warnings.filterwarnings('ignore')

FEATURES_FILE = 'outputs/Advanced_Features_Raw_Final.csv'
OUTPUT_DIR = Path('classification_results/comparison_3groups')

def get_classifiers():
    classifiers = {
        'Random Forest': RandomForestClassifier(
            n_estimators=100, max_depth=5, 
            min_samples_split=4, min_samples_leaf=2,
            random_state=42, class_weight='balanced'
        ),
        'Logistic Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('lr', LogisticRegression(C=0.1, class_weight='balanced', max_iter=1000, random_state=42))
        ]),
        'MLP (Neural Net)': Pipeline([
            ('scaler', StandardScaler()),
            ('mlp', MLPClassifier(
                hidden_layer_sizes=(32, 16),
                alpha=0.01,
                learning_rate_init=0.001,
                max_iter=1000,
                early_stopping=True,
                validation_fraction=0.2,
                random_state=42
            ))
        ])
    }
    
    if XGBOOST_AVAILABLE:
        classifiers['XGBoost'] = XGBClassifier(
            n_estimators=100, random_state=42,
            max_depth=6, learning_rate=0.1, eval_metric='logloss'
        )
    return classifiers

def load_and_aggregate_features():
    print(f"\n{'='*60}")
    print("Loading and Processing Features")
    print('='*60)
    
    df = pd.read_csv(FEATURES_FILE)
    df = df.dropna(subset=['Disease', 'Collagen_Status'])
    df['Disease'] = df['Disease'].replace({'TAA': 'TAV_TAA', 'BAV': 'BAV_TAA'})
    
    def normalize_filename(f):
        s = str(f).replace('.nii.gz', '').replace('.tif', '')
        s = s.replace('_segmentation', '').replace('_visible', '')
        return s.strip()
        
    df['Cell_ID'] = df['CellName'].apply(normalize_filename)
    
    # Feature columns
    feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns 
                    if c not in ['Numeric_ID', 'FourClass', 'Age']]
    
    agg_dict = {col: 'max' for col in feature_cols}
    agg_dict.update({'Disease': 'first', 'Collagen_Status': 'first'})
    
    df_agg = df.groupby('Cell_ID').agg(agg_dict)
    
    X = df_agg[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    print(f"\nTotal aggregated cells: {len(X)}")
    print(df_agg.groupby(['Disease', 'Collagen_Status']).size().unstack(fill_value=0))
    
    return X, df_agg

def prepare_task_data(X, df_agg, task):
    """Filter data based on task configuration."""
    print(f"\nPreparing data for: {task['name']}")
    
    # Filter conditions
    group1_disease = task['group1']['disease']
    group1_collagen = task['group1']['collagen']
    group2_disease = task['group2']['disease']
    group2_collagen = task['group2']['collagen']
    
    mask1 = (df_agg['Disease'] == group1_disease) & (df_agg['Collagen_Status'] == group1_collagen)
    mask2 = (df_agg['Disease'] == group2_disease) & (df_agg['Collagen_Status'] == group2_collagen)
    
    final_mask = mask1 | mask2
    
    if sum(final_mask) < 10: # Minimum records
        print("  Skipping - insufficient data")
        return None, None
        
    X_task = X[final_mask]
    
    # Label: 0 for Group 1, 1 for Group 2
    y_task = np.where(df_agg.loc[final_mask, 'Disease'] == group2_disease, 1, 0)
    if group1_disease == group2_disease: # Treatment task
        y_task = np.where(df_agg.loc[final_mask, 'Collagen_Status'] == group2_collagen, 1, 0)
        
    print(f"  Class 0 ({group1_disease} {group1_collagen}): {sum(y_task==0)}")
    print(f"  Class 1 ({group2_disease} {group2_collagen}): {sum(y_task==1)}")
    
    return X_task, y_task

def evaluate_single_task(X, y, task_name):
    print(f"\n{'='*60}\nEVALUATING: {task_name}\n{'='*60}")
    
    if X is None or y is None:
        return pd.DataFrame()
        
    classifiers = get_classifiers()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    results = []
    
    
    for name, clf in classifiers.items():
        print(f"  Training {name}...")
        try:
            cv_results = cross_validate(clf, X, y, cv=cv, scoring=['accuracy', 'f1', 'roc_auc'], return_train_score=True)
            y_pred_cv = cross_val_predict(clf, X, y, cv=cv)
            cm = confusion_matrix(y, y_pred_cv)
            
            test_acc = cv_results['test_accuracy'].mean()
            train_acc = cv_results['train_accuracy'].mean()
            test_acc_std = cv_results['test_accuracy'].std()
            overfit_gap = train_acc - test_acc
            
            results.append({
                'Task': task_name,
                'Classifier': name,
                'Accuracy': test_acc, 'Accuracy_Std': test_acc_std,
                'F1_Score': cv_results['test_f1'].mean(), 'F1_Std': cv_results['test_f1'].std(),
                'AUC_ROC': cv_results['test_roc_auc'].mean(), 'AUC_Std': cv_results['test_roc_auc'].std(),
                'Overfit_Gap': overfit_gap,
                'Confusion_Matrix': cm
            })
            print(f"    Accuracy: {test_acc:.1%} +/- {test_acc_std:.1%} (Gap: {overfit_gap:.1%})")
        except Exception as e:
            print(f"    ERROR: {str(e)}")
            
    return pd.DataFrame(results)

def plot_task_comparison(results_df, task_name, output_dir):
    if results_df.empty: return
    results_df = results_df.sort_values('Accuracy', ascending=True)
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    fig.suptitle(f'Classifier Comparison: {task_name}', fontweight='bold', y=1.05)
    colors = ["#AEC6CF", "#FFB347", "#77DD77", "#FF6961"]
    
    metrics = [('Accuracy', 'Accuracy_Std', axes[0]), ('F1_Score', 'F1_Std', axes[1]), ('AUC_ROC', 'AUC_Std', axes[2])]
    
    for metric, std, ax in metrics:
        ax.barh(results_df['Classifier'], results_df[metric], xerr=results_df[std], color=colors[:len(results_df)], capsize=3)
        ax.set_title(metric.replace("_", " "))
        ax.set_xlim(0, 1.0)
        if metric == 'Accuracy': ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.5)
            
    # Table subplot
    ax_table = axes[3]
    ax_table.axis('off')
    table_data = [[r['Classifier'], f"{r['Accuracy']:.1%}", f"{r['F1_Score']:.1%}", f"{r['AUC_ROC']:.1%}"] for _, r in results_df.iterrows()]
    table = ax_table.table(cellText=table_data, colLabels=["Model", "Acc", "F1", "AUC"], loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    safe_name = task_name.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '-')
    plt.tight_layout()
    plt.savefig(output_dir / f'comparison_{safe_name}.png', bbox_inches='tight', dpi=300)
    plt.close()

def plot_confusion_matrix_best_model(results_df, task_name, output_dir, group1_name, group2_name):
    if results_df.empty: return
    
    # Identify best model
    best_row = results_df.loc[results_df['Accuracy'].idxmax()]
    best_model_name = best_row['Classifier']
    cm = best_row['Confusion_Matrix']
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=[group1_name, group2_name], 
                yticklabels=[group1_name, group2_name])
    
    plt.title(f"{task_name}\nBest Model: {best_model_name} (Acc: {best_row['Accuracy']:.1%})")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    safe_name = task_name.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '-')
    plt.tight_layout()
    plt.savefig(output_dir / f'confusion_matrix_{safe_name}.png', dpi=300)
    plt.close()

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    X, df_agg = load_and_aggregate_features()
    
    tasks = [
        {'name': 'Diseased Task: TAV vs Healthy (Baseline)', 'group1': {'disease': 'Healthy', 'collagen': 'NoCollagen'}, 'group2': {'disease': 'TAV_TAA', 'collagen': 'NoCollagen'}},
        {'name': 'Diseased Task: BAV vs Healthy (Baseline)', 'group1': {'disease': 'Healthy', 'collagen': 'NoCollagen'}, 'group2': {'disease': 'BAV_TAA', 'collagen': 'NoCollagen'}},
        {'name': 'Diseased Task: TAV vs BAV (Baseline)', 'group1': {'disease': 'BAV_TAA', 'collagen': 'NoCollagen'}, 'group2': {'disease': 'TAV_TAA', 'collagen': 'NoCollagen'}},
        {'name': 'Treatment Task: TAV Rescue', 'group1': {'disease': 'TAV_TAA', 'collagen': 'NoCollagen'}, 'group2': {'disease': 'TAV_TAA', 'collagen': 'Collagen'}},
        {'name': 'Treatment Task: BAV Rescue', 'group1': {'disease': 'BAV_TAA', 'collagen': 'NoCollagen'}, 'group2': {'disease': 'BAV_TAA', 'collagen': 'Collagen'}},
        {'name': 'Treatment Task: Healthy Effect', 'group1': {'disease': 'Healthy', 'collagen': 'NoCollagen'}, 'group2': {'disease': 'Healthy', 'collagen': 'Collagen'}}
    ]
    
    all_results = []
    
    for task in tasks:
        X_task, y_task = prepare_task_data(X, df_agg, task)
        task_results = evaluate_single_task(X_task, y_task, task['name'])
        
        if not task_results.empty:
            all_results.append(task_results)
            plot_task_comparison(task_results, task['name'], OUTPUT_DIR)
            
            group1_label = f"{task['group1']['disease']} {task['group1']['collagen']}"
            group2_label = f"{task['group2']['disease']} {task['group2']['collagen']}"
            plot_confusion_matrix_best_model(task_results, task['name'], OUTPUT_DIR, group1_label, group2_label)
            
    if all_results:
        all_results_df = pd.concat(all_results, ignore_index=True)
        all_results_df.to_csv(OUTPUT_DIR / 'all_tasks_metrics.csv', index=False)
        
        print("\n" + "="*60)
        print("SUMMARY: BEST PERFORMERS PER TASK")
        print("="*60)
        for task in all_results_df['Task'].unique():
            td = all_results_df[all_results_df['Task'] == task]
            best = td.loc[td['Accuracy'].idxmax()]
            print(f"\n{task}:")
            print(f"  Best: {best['Classifier']} -> Accuracy: {best['Accuracy']:.1%} | AUC: {best['AUC_ROC']:.1%}")
            
if __name__ == '__main__':
    main()
