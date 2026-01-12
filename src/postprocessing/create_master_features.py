import pandas as pd
import glob
import os

# List of all feature CSVs we just generated
feature_files = [
    'actin_training_metrics_adv.csv',
    'actin_inference_metrics_adv.csv',
    'mito_training_metrics_adv.csv',
    'mito_inference_metrics_adv.csv',
    # Nucleus metrics were moved to experiments/V2/analysis_results, let's grab them
    'experiments/V2/analysis_results/nucleus_training_metrics.csv',
    'experiments/V2/analysis_results/nucleus_inference_metrics.csv'
]

dfs = []

for f in feature_files:
    if os.path.exists(f):
        df = pd.read_csv(f)
        
        # Determine structure type from filename
        if 'actin' in f.lower():
            structure = 'Actin'
        elif 'mito' in f.lower():
            structure = 'Mito'
        elif 'nucleus' in f.lower():
            structure = 'Nucleus'
        else:
            structure = 'Unknown'
            
        # Determine Source (Training vs Inference)
        if 'training' in f.lower():
            source = 'Training'
        elif 'inference' in f.lower():
            source = 'Inference'
        else:
            source = 'Unknown'
            
        # Prefix columns with Structure name to avoid collision (Volume -> Actin_Volume)
        # except Filename
        new_cols = []
        for c in df.columns:
            if c == 'Filename':
                new_cols.append(c)
            else:
                new_cols.append(f"{structure}_{c}")
        df.columns = new_cols
        
        # Add metadata columns
        df['Structure'] = structure
        df['Dataset_Source'] = source
        
        # We need to reshape or merge?
        # The user wants "Advanced_Features.csv" with "Subject_ID".
        # IMPORTANT: A single subject has MANY cells.
        # AND a single subject has Actin, Mito, AND Nucleus data.
        # Are the filenames consistent across structures? 
        # e.g. "Subject1_Cell1.tif" for Actin and "Subject1_Cell1.tif" for Mito?
        # If so, we can merge rows. 
        # If not (random cells), we treat them as independent samples or aggregate per subject.
        # User prompt: "Create a Subject_ID... from Filename... Merge... Target: Group".
        # This implies we will classify *rows* of this extracted file.
        # If we validly strictly merge, we'd need Actin_Volume, Mito_Volume on the SAME row.
        # That requires the filenames to match exactly or have a common ID.
        # Let's assume we simply concatenate for now (Wide format might be impossible if cell counts differ).
        # Actually, "Random Forest... merge feature data with metadata" -> 
        # If we have rows for Actin, rows for Mito, rows for Nucleus... 
        # A Random Forest usually takes a fixed feature vector X.
        # If row 1 is Actin (has Actin_Vol, Mito_Vol=NaN), row 2 is Mito (Actin_Vol=NaN, Mito_Vol=XX).
        # This sparse matrix is bad for RF. 
        # The user likely wants to classify *Cells* that have ALL channels? 
        # OR classify Subjects based on aggregated stats?
        # OR classify single structure images?
        # Given "Super-Feature List", it seems they want to classify the *Subject* or *Cell* using available features.
        # Let's stack them? Or maybe the user implies we analyze *Images* that have all 3 channels.
        # But our inputs were separate folders.
        # Let's create a "Long" format where we have "Subject_ID" and then all possible columns.
        # If we cannot link Actin Cell 1 to Mito Cell 1, we treat them as separate samples?
        # No, that makes no sense for "Actin_Connectivity" AND "Mito_Fragmentation" features in one classifier.
        # Unless we Classify SUBJECTS.
        # Let's aggregate by Subject_ID? 
        # User instruction: "Create Subject_ID... Merge... Target... Run Random Forest".
        # It doesn't explicitly say "Aggregate".
        # However, "Subject_01.tif" -> Subject_01.
        # If we have multiple cells per subject, we will have multiple rows per subject.
        # If we leave it as is, and we have columns "Actin_Volume" and "Mito_Volume"...
        # If a row is from an Actin file, "Mito_Volume" is null.
        # Standard approach: Fillna(0)? Or impute? 
        # OR: The user assumes we have matched samples.
        # Let's look at filenames in the CSVs.
        
        dfs.append(df)
    else:
        print(f"Skipping missing file: {f}")

if dfs:
    # Concatenate all (Long format with sparse columns)
    # This is risky if we intend to use ALL features for ONE prediction.
    # But let's build the CSV first.
    full_df = pd.concat(dfs, ignore_index=True)
    
    # Save
    full_df.to_csv('Advanced_Features_Raw.csv', index=False)
    print("Created Advanced_Features_Raw.csv with sparse columns.")
else:
    print("No data found.")
