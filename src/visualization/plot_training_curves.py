#!/usr/bin/env python3
"""
Generate publication-quality training curves from nnU-Net v2 logs.
Plots Dice Score and Loss curves with smoothing for Actin, Mito, and Nucleus.
"""

import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Configuration
RESULTS_DIR = Path('/home/luisa/aneurysm_project/data/nnUNet/nnUNet_results')
OUTPUT_FILE = '/home/luisa/aneurysm_project/training_curves.png'

DATASETS = {
    'Actin': 'Dataset001_Actin',
    'Mitochondria': 'Dataset002_Mito',
    'Nucleus': 'Dataset003_Nucleus'
}

# Smoothing factor for EMA (higher = more smoothing)
SMOOTHING_WEIGHT = 0.9


def find_log_file(dataset_dir):
    """Find the training log file in a dataset results folder."""
    fold_dir = dataset_dir / 'nnUNetTrainer__nnUNetPlans__3d_fullres' / 'fold_0'
    log_files = list(fold_dir.glob('training_log_*.txt'))
    if log_files:
        return log_files[0]
    return None


def parse_log(log_path):
    """Extract training metrics from nnU-Net log file."""
    epochs = []
    train_losses = []
    val_losses = []
    dice_scores = []
    
    with open(log_path, 'r') as f:
        content = f.read()
    
    # Parse epoch blocks
    epoch_pattern = r'Epoch (\d+).*?train_loss (-?[\d.]+).*?val_loss (-?[\d.]+).*?Pseudo dice \[np\.float32\(([\d.]+)\)\]'
    matches = re.findall(epoch_pattern, content, re.DOTALL)
    
    for match in matches:
        epochs.append(int(match[0]))
        train_losses.append(float(match[1]))
        val_losses.append(float(match[2]))
        dice_scores.append(float(match[3]))
    
    return {
        'epochs': np.array(epochs),
        'train_loss': np.array(train_losses),
        'val_loss': np.array(val_losses),
        'dice': np.array(dice_scores)
    }


def smooth_ema(data, weight=0.9):
    """Apply exponential moving average smoothing."""
    smoothed = np.zeros_like(data)
    smoothed[0] = data[0]
    for i in range(1, len(data)):
        smoothed[i] = weight * smoothed[i-1] + (1 - weight) * data[i]
    return smoothed


def main():
    print("Parsing training logs...")
    
    # Parse all logs
    all_data = {}
    for name, dataset_id in DATASETS.items():
        log_path = find_log_file(RESULTS_DIR / dataset_id)
        if log_path:
            print(f"  {name}: {log_path.name}")
            all_data[name] = parse_log(log_path)
        else:
            print(f"  {name}: Log not found!")
    
    if not all_data:
        print("No logs found!")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # Color scheme
    colors = {
        'Actin': '#E74C3C',      # Red
        'Mitochondria': '#27AE60', # Green
        'Nucleus': '#3498DB'      # Blue
    }
    
    # Plot settings
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 10
    
    for idx, (name, data) in enumerate(all_data.items()):
        color = colors.get(name, '#333333')
        epochs = data['epochs']
        
        # Top row: Dice Score
        ax_dice = axes[0, idx]
        ax_dice.plot(epochs, data['dice'], alpha=0.3, color=color, linewidth=0.8)
        ax_dice.plot(epochs, smooth_ema(data['dice'], SMOOTHING_WEIGHT), 
                    color=color, linewidth=2, label=f'{name} (smoothed)')
        ax_dice.set_title(f'{name}', fontsize=14, fontweight='bold', color=color)
        ax_dice.set_xlabel('Epoch')
        ax_dice.set_ylabel('Dice Score')
        ax_dice.set_ylim(0.5, 1.0)
        ax_dice.grid(True, alpha=0.3)
        ax_dice.axhline(y=data['dice'].max(), linestyle='--', color=color, alpha=0.5, linewidth=1)
        ax_dice.text(epochs[-1] * 0.95, data['dice'].max() + 0.01, 
                    f'Max: {data["dice"].max():.3f}', ha='right', fontsize=9, color=color)
        
        # Bottom row: Loss
        ax_loss = axes[1, idx]
        ax_loss.plot(epochs, data['train_loss'], alpha=0.3, color='#3498DB', linewidth=0.8)
        ax_loss.plot(epochs, smooth_ema(data['train_loss'], SMOOTHING_WEIGHT), 
                    color='#3498DB', linewidth=2, label='Train Loss')
        ax_loss.plot(epochs, data['val_loss'], alpha=0.3, color='#E74C3C', linewidth=0.8)
        ax_loss.plot(epochs, smooth_ema(data['val_loss'], SMOOTHING_WEIGHT), 
                    color='#E74C3C', linewidth=2, label='Val Loss')
        ax_loss.set_xlabel('Epoch')
        ax_loss.set_ylabel('Loss')
        ax_loss.grid(True, alpha=0.3)
        ax_loss.legend(loc='upper right', fontsize=8)
    
    # Overall title
    fig.suptitle('nnU-Net v2 Training Curves', fontsize=16, fontweight='bold', y=0.98)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save
    plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nSaved plot to: {OUTPUT_FILE}")
    
    # Print summary stats
    print("\n=== Training Summary ===")
    for name, data in all_data.items():
        print(f"{name}:")
        print(f"  Epochs: {len(data['epochs'])}")
        print(f"  Final Dice: {data['dice'][-1]:.4f}")
        print(f"  Best Dice:  {data['dice'].max():.4f}")


if __name__ == '__main__':
    main()
