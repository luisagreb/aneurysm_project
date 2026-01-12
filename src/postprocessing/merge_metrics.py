import pandas as pd
import os
import sys

def merge_csvs(file1, file2, source1, source2, output_file):
    if not os.path.exists(file1):
        print(f"Warning: {file1} not found. Skipping merge for this pair.")
        return
    if not os.path.exists(file2):
        print(f"Warning: {file2} not found. Skipping merge for this pair.")
        return
        
    df1 = pd.read_csv(file1)
    df1['Source'] = source1
    
    df2 = pd.read_csv(file2)
    df2['Source'] = source2
    
    combined = pd.concat([df1, df2], ignore_index=True)
    
    # Move 'Source' to the second column
    cols = ['Filename', 'Source'] + [c for c in combined.columns if c not in ['Filename', 'Source']]
    combined = combined[cols]
    
    combined.to_csv(output_file, index=False)
    print(f"Merged {file1} and {file2} into {output_file}")

def main():
    # Merge Actin
    merge_csvs(
        'actin_training_metrics.csv', 'actin_inference_metrics.csv',
        'Training', 'Inference',
        'actin_combined_metrics.csv'
    )
    
    
    # Merge Mito
    merge_csvs(
        'mito_training_metrics.csv', 'mito_inference_metrics.csv',
        'Training', 'Inference',
        'mito_combined_metrics.csv'
    )

    # Merge Nucleus
    merge_csvs(
        'nucleus_training_metrics.csv', 'nucleus_inference_metrics.csv',
        'Training', 'Inference',
        'nucleus_combined_metrics.csv'
    )

if __name__ == "__main__":
    main()
