import os
import shutil

def copy_oir_files_recursive(source_root_folder):

    parent_dir = os.path.dirname(source_root_folder)
    if not parent_dir:

        parent_dir = os.getcwd()
        
    destination_folder = os.path.join(parent_dir, "NewData")
    
    # 2. Create the destination folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
        print(f" Created destination folder: '{destination_folder}'")
    else:
        print(f" Destination folder already exists: '{destination_folder}'")

    oir_count = 0
    print(f"\n Starting recursive scan from: '{os.path.abspath(source_root_folder)}'")
    

    for root, dirs, files in os.walk(source_root_folder):
        
        # We can skip the 'NewData' folder if it exists within the source_root_folder
        if root == destination_folder:
            continue

        # Iterate through all files found in the current directory ('root')
        for file_name in files:
            
            # Check if the file ends with '.oir'
            if file_name.endswith(".oir"):
                
                # Construct the full path to the file
                source_path = os.path.join(root, file_name)
                
                try:
                    destination_path = os.path.join(destination_folder, file_name)
                    
                    # Copy the file
                    # We use shutil.copy2 to preserve metadata (like timestamps)
                    shutil.copy2(source_path, destination_path)
                    
                    # Print the path of the original file found
                    print(f"   [Found in: {root}] --> Copied: {file_name}")
                    oir_count += 1
                    
                except Exception as e:
                    print(f"🛑 Error copying file {source_path}: {e}")
    
    # 4. Print the final summary
    if oir_count > 0:
        print(f"\n✨ Operation complete! Recursively copied **{oir_count}** .oir files to '{destination_folder}'.")
    else:
        print("\n😔 No .oir files were found recursively in the directory structure.")

# --- Execution ---

source_directory_path = "C:/Users/YourName/Documents/MyProjectData" 

copy_oir_files_recursive(source_directory_path)