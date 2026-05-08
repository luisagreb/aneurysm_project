import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Config
RESULTS_CSV = 'classification_results/regularized/regularized_results.csv'
OUTPUT_DIR = Path('classification_results/regularized/tables')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def render_mpl_table(data, col_width=3.0, row_height=0.625, font_size=12,
                     header_color='#2c3e50', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')

    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in mpl_table._cells.items():
        cell.set_edgecolor(edge_color)
        if k[0] == 0:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors) ])
    return ax

def main():
    if not Path(RESULTS_CSV).exists():
        print(f"Error: {RESULTS_CSV} not found.")
        return

    df = pd.read_csv(RESULTS_CSV)
    
    # Filter columns
    cols_to_keep = ['Task', 'Classifier', 'Accuracy', 'F1_Score', 'AUC_ROC']
    df = df[cols_to_keep]
    
    # Format percentages
    for col in ['Accuracy', 'F1_Score', 'AUC_ROC']:
        df[col] = (df[col] * 100).apply(lambda x: f"{x:.1f}%")
    
    # Rename for pretty printing
    df.columns = ['Task', 'Classifier', 'Accuracy', 'F1 Score', 'AUC']

    tasks = df['Task'].unique()
    
    for task in tasks:
        print(f"Generating table for: {task}")
        task_df = df[df['Task'] == task].drop(columns=['Task'])
        
        # Sort by Accuracy descending
        task_df = task_df.sort_values(by='Accuracy', ascending=False)
        
        # Plot
        import numpy as np
        fig_width = 8
        fig_height = len(task_df) * 0.8 + 1
        
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.axis('off')
        ax.set_title(f"Model Performance: {task}", fontsize=16, weight='bold', pad=20)
        
        # Define pretty colors (Blue header like in your image)
        render_mpl_table(task_df, header_color='#007acc', row_colors=['#e6f2ff', 'white'], ax=ax)
        
        safe_name = task.replace(' ', '_').lower()
        save_path = OUTPUT_DIR / f'table_{safe_name}.png'
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"  Saved: {save_path}")
        plt.close()

if __name__ == "__main__":
    main()
