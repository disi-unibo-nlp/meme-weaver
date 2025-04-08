import os
import shutil
import argparse

FOLDERS_TO_KEEP = ["xlm-roberta-large_texten_qwen25vl_caption_128batch_10eps_mami128_1gcn",
                   "xlm-roberta-large_texten_batch128_10eps_1gcn",
                   "xlm-roberta-large_batch140_10eps_1gcn"]

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
    
    folder_list = get_folders_in_path(args.folder)

    for folder in folder_list:
            
            if folder in FOLDERS_TO_KEEP:
                continue
            # Specify the file path you want to delete
            file_path = os.path.join(args.folder, folder, "training_args.bin")
            remove_file(file_path)

            file_path = os.path.join(args.folder, folder, "model.safetensors")
            remove_file(file_path)

            file_path = os.path.join(args.folder, folder, "sentencepiece.bpe.model")
            remove_file(file_path)
            
            folder_path = os.path.join(args.folder, folder)
            try:
                final_folder_path = os.path.join(folder_path, get_folders_in_path(folder_path)[0])
                if "checkpoint" in final_folder_path:
                    remove_folder(final_folder_path)
            except:
                print("No checkpoint folder found")
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", default="output_hard_label_task4")

    args = parser.parse_args()
    main()