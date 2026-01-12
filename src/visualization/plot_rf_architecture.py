#!/usr/bin/env python3
"""
Create a Random Forest architecture diagram customized for our project.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path('classification_results')

def draw_random_forest():
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Colors
    tree_color = '#9B59B6'
    input_color = '#3498DB'
    output_color = '#F1C40F'
    healthy_color = '#27AE60'
    taa_color = '#E74C3C'
    
    # Title
    ax.text(7, 9.5, 'Random Forest Classifier', ha='center', fontsize=18, fontweight='bold')
    ax.text(7, 9.0, '100 Decision Trees for Healthy vs TAA Classification', ha='center', fontsize=12, color='#7F8C8D')
    
    # --- INPUT DATA BOX ---
    input_box = mpatches.FancyBboxPatch((0.5, 7), 3, 1.5, boxstyle="round,pad=0.1",
                                         facecolor='#EBF5FB', edgecolor=input_color, lw=2)
    ax.add_patch(input_box)
    ax.text(2, 8.2, '200 Cells', ha='center', fontsize=12, fontweight='bold', color=input_color)
    ax.text(2, 7.6, '23 Features each', ha='center', fontsize=10, color='#2C3E50')
    ax.text(2, 7.2, '(Actin, Mito, Nucleus)', ha='center', fontsize=9, style='italic', color='#7F8C8D')
    
    # --- BOOTSTRAP SAMPLING ---
    ax.annotate('', xy=(2, 6.8), xytext=(2, 7),
                arrowprops=dict(arrowstyle='->', color='#2C3E50', lw=2))
    
    # Bootstrap samples box
    boot_box = mpatches.FancyBboxPatch((0.3, 5.8), 3.4, 0.8, boxstyle="round,pad=0.05",
                                        facecolor='#FDEBD0', edgecolor='#F39C12', lw=1)
    ax.add_patch(boot_box)
    ax.text(2, 6.2, 'Bootstrap Sampling', ha='center', fontsize=10, fontweight='bold', color='#F39C12')
    ax.text(2, 5.95, '(Random subsets with replacement)', ha='center', fontsize=8, color='#7F8C8D')
    
    # --- DECISION TREES ---
    tree_positions = [1, 3.5, 6, 8.5, 11]
    tree_labels = ['Tree 1', 'Tree 2', 'Tree 3', '...', 'Tree 100']
    
    for i, (x, label) in enumerate(zip(tree_positions, tree_labels)):
        if label == '...':
            ax.text(x + 0.5, 4, '...', ha='center', fontsize=24, color='#7F8C8D')
            continue
            
        # Draw tree structure
        # Root node
        root = plt.Circle((x + 0.5, 5), 0.25, color=tree_color, ec='white', lw=2, zorder=3)
        ax.add_patch(root)
        
        # Branches
        ax.plot([x + 0.5, x], [4.75, 4.25], color='#2C3E50', lw=2)
        ax.plot([x + 0.5, x + 1], [4.75, 4.25], color='#2C3E50', lw=2)
        
        # Level 2 nodes
        node_l = plt.Circle((x, 4), 0.2, color=tree_color, ec='white', lw=2, zorder=3)
        node_r = plt.Circle((x + 1, 4), 0.2, color=tree_color, ec='white', lw=2, zorder=3)
        ax.add_patch(node_l)
        ax.add_patch(node_r)
        
        # More branches
        ax.plot([x, x - 0.3], [3.8, 3.3], color='#2C3E50', lw=1.5)
        ax.plot([x, x + 0.3], [3.8, 3.3], color='#2C3E50', lw=1.5)
        ax.plot([x + 1, x + 0.7], [3.8, 3.3], color='#2C3E50', lw=1.5)
        ax.plot([x + 1, x + 1.3], [3.8, 3.3], color='#2C3E50', lw=1.5)
        
        # Leaf nodes (predictions)
        leaves = [(x - 0.3, healthy_color), (x + 0.3, taa_color), 
                  (x + 0.7, healthy_color), (x + 1.3, taa_color)]
        for lx, lcolor in leaves:
            leaf = plt.Circle((lx, 3.1), 0.15, color=lcolor, ec='white', lw=1, zorder=3)
            ax.add_patch(leaf)
        
        # Tree label
        ax.text(x + 0.5, 2.5, label, ha='center', fontsize=10, fontweight='bold', color=tree_color)
        
        # Arrow from bootstrap to tree
        if label != '...':
            ax.annotate('', xy=(x + 0.5, 5.3), xytext=(2, 5.8),
                       arrowprops=dict(arrowstyle='->', color='#BDC3C7', lw=1, 
                                      connectionstyle="arc3,rad=0.1"))
    
    # --- VOTING BOX ---
    vote_box = mpatches.FancyBboxPatch((4, 1.3), 6, 0.8, boxstyle="round,pad=0.1",
                                        facecolor='#FADBD8', edgecolor='#E74C3C', lw=2)
    ax.add_patch(vote_box)
    ax.text(7, 1.7, 'Majority Voting', ha='center', fontsize=12, fontweight='bold', color='#E74C3C')
    
    # Arrows from trees to voting
    for x in [1.5, 4, 9, 11.5]:
        ax.annotate('', xy=(7, 2.1), xytext=(x, 2.5),
                   arrowprops=dict(arrowstyle='->', color='#BDC3C7', lw=1))
    
    # --- OUTPUT ---
    ax.annotate('', xy=(7, 0.9), xytext=(7, 1.3),
                arrowprops=dict(arrowstyle='->', color='#2C3E50', lw=2))
    
    # Output predictions
    out_box = mpatches.FancyBboxPatch((4.5, 0.1), 5, 0.7, boxstyle="round,pad=0.1",
                                       facecolor='#D5F5E3', edgecolor='#27AE60', lw=2)
    ax.add_patch(out_box)
    ax.text(7, 0.45, 'Final Prediction: Healthy or TAA', ha='center', fontsize=11, 
           fontweight='bold', color='#27AE60')
    
    # --- LEGEND ---
    legend_y = 4.5
    legend_x = 12.5
    
    ax.text(legend_x, legend_y + 1.5, 'Legend', fontsize=11, fontweight='bold')
    
    # Healthy leaf
    h_leaf = plt.Circle((legend_x, legend_y + 0.8), 0.15, color=healthy_color)
    ax.add_patch(h_leaf)
    ax.text(legend_x + 0.4, legend_y + 0.8, 'Healthy', va='center', fontsize=9)
    
    # TAA leaf  
    t_leaf = plt.Circle((legend_x, legend_y + 0.3), 0.15, color=taa_color)
    ax.add_patch(t_leaf)
    ax.text(legend_x + 0.4, legend_y + 0.3, 'TAA', va='center', fontsize=9)
    
    # Decision node
    d_node = plt.Circle((legend_x, legend_y - 0.2), 0.15, color=tree_color)
    ax.add_patch(d_node)
    ax.text(legend_x + 0.4, legend_y - 0.2, 'Decision node', va='center', fontsize=9)
    
    # --- INFO BOX ---
    info_text = "Configuration:\n• 100 decision trees\n• Bootstrap sampling\n• Random feature subsets\n\nResults:\n• Accuracy: 82.5%\n• AUC-ROC: 0.90"
    props = dict(boxstyle='round', facecolor='#ECF0F1', alpha=0.9)
    ax.text(12.5, 8, info_text, fontsize=9, verticalalignment='top',
           bbox=props, fontfamily='monospace')
    
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.axis('off')
    
    plt.tight_layout()
    
    output_path = OUTPUT_DIR / 'rf_architecture.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")

if __name__ == '__main__':
    OUTPUT_DIR.mkdir(exist_ok=True)
    draw_random_forest()
