#!/usr/bin/env python3
"""
Create a neural network architecture diagram customized for our project.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path('classification_results')

def draw_neural_network():
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Layer configuration for our project
    layers = [
        {'name': 'Input\nlayer', 'neurons': 23, 'show': 6, 'color': '#3498DB', 'label': '23 features'},
        {'name': 'Hidden\nlayer 1', 'neurons': 50, 'show': 7, 'color': '#9B59B6', 'label': '50 neurons'},
        {'name': 'Hidden\nlayer 2', 'neurons': 25, 'show': 6, 'color': '#9B59B6', 'label': '25 neurons'},
        {'name': 'Output\nlayer', 'neurons': 2, 'show': 2, 'color': '#F1C40F', 'label': '2 classes'}
    ]
    
    x_positions = [1.5, 4.5, 7.5, 10.5]
    neuron_radius = 0.25
    
    # Store neuron positions for connections
    all_positions = []
    
    for layer_idx, (layer, x) in enumerate(zip(layers, x_positions)):
        n_show = layer['show']
        color = layer['color']
        
        # Calculate y positions
        if layer['neurons'] > n_show:
            # Show some neurons with "..." in middle
            top_neurons = n_show // 2
            bottom_neurons = n_show - top_neurons
            
            y_top = np.linspace(6.5, 5, top_neurons)
            y_bottom = np.linspace(3, 1.5, bottom_neurons)
            y_positions = list(y_top) + list(y_bottom)
            has_dots = True
        else:
            # Show all neurons
            y_positions = np.linspace(5.5, 2.5, n_show)
            has_dots = False
        
        layer_positions = []
        
        for i, y in enumerate(y_positions):
            circle = plt.Circle((x, y), neuron_radius, color=color, ec='white', lw=2, zorder=3)
            ax.add_patch(circle)
            layer_positions.append((x, y))
        
        # Add dots to indicate more neurons
        if has_dots:
            for dot_y in [4.3, 4.0, 3.7]:
                ax.plot(x, dot_y, 'o', color='#7F8C8D', markersize=3, zorder=3)
        
        all_positions.append(layer_positions)
        
        # Layer label at bottom
        ax.text(x, 0.3, layer['name'], ha='center', va='top', fontsize=12, fontweight='bold')
        
        # Neuron count label
        ax.text(x, 7.2, layer['label'], ha='center', va='bottom', fontsize=10, 
                color=color, fontweight='bold')
    
    # Draw connections between layers
    for layer_idx in range(len(all_positions) - 1):
        current_layer = all_positions[layer_idx]
        next_layer = all_positions[layer_idx + 1]
        
        for (x1, y1) in current_layer:
            for (x2, y2) in next_layer:
                ax.plot([x1 + neuron_radius, x2 - neuron_radius], [y1, y2], 
                       color='#BDC3C7', linewidth=0.5, alpha=0.6, zorder=1)
    
    # Add feature labels for input layer
    input_features = [
        'Actin_Volume',
        'Actin_Skeleton',
        'Mito_Fragments',
        '...',
        'Nucleus_Volume',
        'Nucleus_Solidity'
    ]
    
    for i, (label, (x, y)) in enumerate(zip(input_features, all_positions[0])):
        if label == '...':
            continue
        ax.text(x - 0.5, y, label, ha='right', va='center', fontsize=9, 
               color='#2C3E50', style='italic')
    
    # Add output labels
    output_labels = ['Healthy (0)', 'TAA (1)']
    for label, (x, y) in zip(output_labels, all_positions[-1]):
        ax.text(x + 0.5, y, label, ha='left', va='center', fontsize=11, 
               fontweight='bold', color='#2C3E50')
    
    # Title
    ax.text(6, 8.2, 'Neural Network Architecture', ha='center', fontsize=16, fontweight='bold')
    ax.text(6, 7.7, 'MLP Classifier for Healthy vs TAA Classification', ha='center', fontsize=12, color='#7F8C8D')
    
    # Architecture summary box
    box_text = "Architecture: 23 → 50 → 25 → 2\nActivation: ReLU\nOptimizer: Adam\nAccuracy: 85%"
    props = dict(boxstyle='round', facecolor='#ECF0F1', alpha=0.8)
    ax.text(12, 7, box_text, fontsize=10, verticalalignment='top', 
           bbox=props, fontfamily='monospace')
    
    # Set limits and remove axes
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 8.5)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Add background boxes for hidden layers
    rect = mpatches.FancyBboxPatch((3.8, 1), 4.4, 6.2, 
                                    boxstyle="round,pad=0.1", 
                                    facecolor='#F5EEF8', 
                                    edgecolor='#9B59B6',
                                    alpha=0.3, zorder=0)
    ax.add_patch(rect)
    ax.text(6, 0.7, 'Hidden layers', ha='center', fontsize=11, color='#9B59B6')
    
    plt.tight_layout()
    
    output_path = OUTPUT_DIR / 'nn_architecture.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    
    return output_path

if __name__ == '__main__':
    OUTPUT_DIR.mkdir(exist_ok=True)
    draw_neural_network()
