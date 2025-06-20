import os
import shutil
import argparse

FOLDERS_TO_KEEP = [
                   "clip-vit-large_batch20_5e-6lr",
                   "clip-vit-large_batch20_5e-6lr_1gcn_xuinit_mfb",
                   "custom_multimodal_xlm-roberta-large_vit-large-patch32-384_batch20_1gcn_xuinit_promptB",
                   ]

def get_folders_in_path(directory):
    folders = []
    for entry in os.scandir(directory):
        if entry.is_dir():
            folders.append(entry.name)
    return folders

def remove_file(file_path):
    try:
        os.remove(file_path)
        print("File removed successfully.")
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except PermissionError:
        print(f"Permission denied: {file_path}")
    except Exception as e:
        print(f"Error occurred while removing file: {str(e)}")

def remove_folder(folder_path):
    try:
        shutil.rmtree(folder_path)
        print(f"Folder '{folder_path}' removed successfully.")
    except OSError as e:
        print(f"Error: {folder_path} : {e.strerror}")


def main():
    
    folder_list = get_folders_in_path(args.output_dir)

    for folder in folder_list:
            
            if folder in FOLDERS_TO_KEEP:
                continue

            # Specify the file path you want to delete
            file_path = os.path.join(args.output_dir, folder, "training_args.bin")
            remove_file(file_path)

            file_path = os.path.join(args.output_dir, folder, "model.safetensors")
            remove_file(file_path)

            file_path = os.path.join(args.output_dir, folder, "tokenizer.json")
            remove_file(file_path)

            file_path = os.path.join(args.output_dir, folder, "sentencepiece.bpe.model")
            remove_file(file_path)
            
            folder_path = os.path.join(args.output_dir, folder)
            try:
                folders = get_folders_in_path(folder_path)
                for folder in folders:
                    final_folder_path = os.path.join(folder_path, folder)
                    if "checkpoint" in final_folder_path:
                        remove_folder(final_folder_path)
            except:
                print("No checkpoint folder found")
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="output_mami")

    args = parser.parse_args()
    main()