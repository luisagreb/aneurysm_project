import os 
import shutil 

source_folder = "D:\SMCs_Zstacks"
def copy_oir_files(source_folder):
    destination_folder = "D:\converted_files"

    oir_count = 0

    for item in os.listdir(source_folder): 
        source_path = os.path.join(source_folder, item)
        if os.path.isfile(source_path) and item.endswith('.oir'):
            try:
                destination_path = os.path.join(destination_folder, item)

                shutil.copy2(source_path,destination_path)
                print(f" copied: {item}")
                oir_count += 1
            except Exception as e:
                print(f"Error copying file: {item}")

    if oir_count > 0:
        print(f"Operation complete, {oir_count} files copied")
    else:
        print (f"Not working")

copy_oir_files(source_folder)