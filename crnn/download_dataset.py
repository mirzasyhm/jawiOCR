# download_dataset.py
import subprocess
import os

def download_and_extract_dataset():
    """
    Downloads the Jawi LMDB dataset archive from Hugging Face and extracts it.
    """
    dataset_url = "https://huggingface.co/datasets/mirzasyhm/synthetic_jawi_images/resolve/main/data_jawi_color_2_lmdb.tar"
    archive_name = "data_jawi_color_2_lmdb.tar"
    
    # We'll extract into a relative './content/' directory to somewhat mimic Colab's structure.
    # The tarball is expected to contain a 'data_jawi_color_2_lmdb' folder.
    extraction_parent_dir = "content" 
    expected_dataset_path = os.path.join(extraction_parent_dir, "data_jawi_color_2_lmdb")

    print(f"Downloading dataset from {dataset_url}...")
    try:
        subprocess.run(["wget", "-q", dataset_url, "-O", archive_name], check=True)
    except FileNotFoundError:
        print("Error: wget command not found. Please ensure wget is installed and in your PATH.")
        print("Alternatively, you can download the file manually and place it as " + archive_name)
        return
    except subprocess.CalledProcessError as e:
        print(f"Error during download: {e}")
        return

    print(f"Extracting {archive_name} into ./{extraction_parent_dir}/ ...")
    os.makedirs(extraction_parent_dir, exist_ok=True)
    try:
        subprocess.run(["tar", "-xf", archive_name, "-C", extraction_parent_dir], check=True)
    except FileNotFoundError:
        print("Error: tar command not found. Please ensure tar is installed and in your PATH.")
        return
    except subprocess.CalledProcessError as e:
        print(f"Error during extraction: {e}")
        return
    
    print(f"Dataset extracted. Expected at: ./{expected_dataset_path}")
    
    # Optional: Clean up the archive file
    # try:
    #     os.remove(archive_name)
    #     print(f"Removed {archive_name}.")
    # except OSError as e:
    #     print(f"Error removing {archive_name}: {e}")

if __name__ == "__main__":
    download_and_extract_dataset()
