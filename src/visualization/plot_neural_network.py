#!/usr/bin/env python3
"""
Generate presentation-quality Neural Network visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
                            roc_curve, auc)

# Configuration
FEATURES_FILE = 'Advanced_Features_Raw.csv'
METADATA_FILE = 'data/Book1.xlsx'
OUTPUT_DIR = Path('classification_results')


def load_data():
    """Load and prepare dataset."""
    df = pd.read_csv(FEATURES_FILE)
    df_meta = pd.read_excel(METADATA_FILE, header=None, skiprows=18)
    
    def extract_id(t):
        if pd.isna(t): return None
        first = str(t).strip().split(' ')[0].strip('-')
        parts = first.split('-')
        if len(parts) > 1 and parts[-1].isdigit():
            return int(parts[-1])
        d = re.findall(r'\d+', first)
        return int(d[-1]) if d else None
    
    healthy = set(extract_id(x) for x in df_meta[0].dropna() if extract_id(x))
    taa = set(extract_id(x) for x in df_meta[1].dropna() if extract_id(x))
    
    df['ID'] = df['Filename'].apply(extract_id)
    df = df.dropna(subset=['ID'])
    df['Label'] = df['ID'].apply(lambda x: 0 if x in healthy else (1 if x in taa else None))
    df = df.dropna(subset=['Label'])
    
    df['Cell'] = df['Filename'].str.replace('.nii.gz','').str.replace('_segmentation','').str.replace('_visible','')
    feat_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in ['Label','ID']]
    agg = {c: 'max' for c in feat_cols}
    agg['Label'] = 'first'
    df_agg = df.groupby('Cell').agg(agg)
    
    X = df_agg.drop(columns=['Label']).replace([np.inf, -np.inf], np.nan).fillna(0)
    y = df_agg['Label']
    
    return X, y


def main():
    print("Loading data...")
    X, y = load_data()
    print(f"Dataset: {len(X)} cells")
    
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Split and scale
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Neural Network
    print("Training Neural Network...")
    nn = MLPClassifier(
        hidden_layer_sizes=(50, 25),
        max_iter=1000,
        random_state=42,
        solver='adam',
        learning_rate_init=0.001,
        early_stopping=False  # Train for full iterations
    )
    nn.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = nn.predict(X_test_scaled)
    y_prob = nn.predict_proba(X_test_scaled)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"Accuracy: {acc:.1%}")
    
    # ===========================================
    # CREATE VISUALIZATIONS
    # ===========================================
    
    fig = plt.figure(figsize=(16, 12))
    
    # --- Panel 1: Neural Network Architecture ---
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    
    # Draw layers
    layer_sizes = [X.shape[1], 50, 25, 2]  # Input, Hidden1, Hidden2, Output
    layer_names = ['Input\n(23 features)', 'Hidden 1\n(50 neurons)', 'Hidden 2\n(25 neurons)', 'Output\n(2 classes)']
    layer_colors = ['#3498DB', '#9B59B6', '#9B59B6', '#E74C3C']
    
    x_positions = [1.5, 4, 6.5, 9]
    
    for i, (size, name, color, x) in enumerate(zip(layer_sizes, layer_names, layer_colors, x_positions)):
        # Draw neurons (max 10 shown)
        n_show = min(size, 8)
        y_positions = np.linspace(2, 8, n_show)
        
        for y in y_positions:
            circle = plt.Circle((x, y), 0.25, color=color, alpha=0.7)
            ax1.add_patch(circle)
        
        if size > 8:
            ax1.text(x, 1.5, f'...({size})', ha='center', fontsize=10)
        
        ax1.text(x, 9.2, name, ha='center', fontsize=11, fontweight='bold')
        
        # Draw connections to next layer
        if i < len(layer_sizes) - 1:
            next_n = min(layer_sizes[i+1], 8)
            next_y = np.linspace(2, 8, next_n)
            for y1 in y_positions[::2]:  # Draw every other connection for clarity
                for y2 in next_y[::2]:
                    ax1.plot([x+0.25, x_positions[i+1]-0.25], [y1, y2], 
                            'gray', alpha=0.2, linewidth=0.5)
    
    ax1.set_title('Neural Network Architecture', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # --- Panel 2: Training Loss Curve ---
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(nn.loss_curve_, linewidth=2, color='#2980B9')
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('Training Loss Curve', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # --- Panel 3: Confusion Matrix ---
    ax3 = fig.add_subplot(2, 2, 3)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Healthy', 'TAA'],
                yticklabels=['Healthy', 'TAA'],
                annot_kws={'size': 20}, ax=ax3)
    ax3.set_xlabel('Predicted', fontsize=12)
    ax3.set_ylabel('Actual', fontsize=12)
    ax3.set_title(f'Confusion Matrix\nAccuracy: {acc:.1%}', fontsize=14, fontweight='bold')
    
    # --- Panel 4: ROC Curve ---
    ax4 = fig.add_subplot(2, 2, 4)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    ax4.plot(fpr, tpr, color='#2980B9', lw=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
    ax4.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1)
    ax4.fill_between(fpr, tpr, alpha=0.2, color='#3498DB')
    ax4.set_xlim([0.0, 1.0])
    ax4.set_ylim([0.0, 1.05])
    ax4.set_xlabel('False Positive Rate', fontsize=12)
    ax4.set_ylabel('True Positive Rate', fontsize=12)
    ax4.set_title('ROC Curve', fontsize=14, fontweight='bold')
    ax4.legend(loc='lower right', fontsize=11)
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Neural Network Classification: Healthy vs TAA', 
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    output_path = OUTPUT_DIR / 'neural_network_results.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nSaved: {output_path}")
    
    # Print summary
    print("\n" + "="*50)
    print("NEURAL NETWORK SUMMARY")
    print("="*50)
    print(f"Architecture: 23 → 50 → 25 → 2")
    print(f"Training iterations: {len(nn.loss_curve_)}")
    print(f"Final loss: {nn.loss_curve_[-1]:.4f}")
    print(f"Test Accuracy: {acc:.1%}")
    print(f"AUC-ROC: {roc_auc:.3f}")


if __name__ == '__main__':
    main()
