# build_alphabet.py
import os
import lmdb
from tqdm import tqdm
import json

# Default path to the training LMDB dataset directory
DEFAULT_TRAIN_LMDB_DIR = os.path.join("content", "data_jawi_color_2_lmdb", "train")
DEFAULT_ALPHABET_OUTPUT_FILE = "alphabet.json"

def build_and_save_alphabet(lmdb_path=DEFAULT_TRAIN_LMDB_DIR, output_file=DEFAULT_ALPHABET_OUTPUT_FILE):
    """
    Scans labels in an LMDB dataset to build a unique character alphabet and saves it to a file.

    Args:
        lmdb_path (str): Path to the LMDB directory (e.g., the 'train' split).
        output_file (str): Path to save the generated alphabet (JSON format).

    Returns:
        list: The sorted list of unique characters (alphabet), or None if an error occurs.
    """
    print(f"Building alphabet from LMDB dataset at: {lmdb_path}")
    
    if not os.path.exists(lmdb_path):
        print(f"Error: LMDB path '{lmdb_path}' does not exist.")
        print("Please ensure you have downloaded and extracted the dataset correctly.")
        print("You might need to run the 'download_dataset.py' script first.")
        return None

    try:
        env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
    except lmdb.Error as e:
        print(f"Error opening LMDB environment at '{lmdb_path}': {e}")
        return None
        
    unique_chars = set()
    
    with env.begin(write=False) as txn:
        try:
            num_samples_bytes = txn.get(b'num-samples')
            if num_samples_bytes is None:
                print(f"Error: 'num-samples' key not found in LMDB at '{lmdb_path}'. "
                      "The dataset might be corrupted or incomplete.")
                env.close()
                return None
            n_samples = int(num_samples_bytes.decode())
        except Exception as e:
            print(f"Error reading 'num-samples' from LMDB: {e}")
            env.close()
            return None
            
        print(f"Scanning {n_samples} samples to build alphabet...")
        for i in tqdm(range(1, n_samples + 1), desc='Scanning labels'):
            lbl_key = f'label-{i:09d}'.encode()
            lbl_bytes = txn.get(lbl_key)
            if lbl_bytes is None:
                print(f"Warning: Label for key '{lbl_key.decode()}' not found (sample index {i-1}). Skipping.")
                continue
            try:
                label_str = lbl_bytes.decode('utf-8')
                unique_chars.update(label_str)
            except UnicodeDecodeError:
                print(f"Warning: Could not decode label for key '{lbl_key.decode()}' as UTF-8. Skipping.")

    env.close()
    
    alphabet_list = sorted(list(unique_chars))
    num_classes = len(alphabet_list) + 1  # +1 for the CTC blank token
    
    print(f"\nAlphabet built successfully.")
    print(f"Found {len(alphabet_list)} unique Jawi characters.")
    # print("Alphabet characters:", alphabet_list) # This can be very long
    print(f"Total number of classes (including CTC blank): {num_classes}")
    
    # Save the alphabet to a JSON file
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(alphabet_list, f, ensure_ascii=False, indent=2)
        print(f"Alphabet saved to '{output_file}'")
    except IOError as e:
        print(f"Error saving alphabet to '{output_file}': {e}")
        return None # Indicate failure to save
        
    return alphabet_list

if __name__ == "__main__":
    print("--- Building Jawi Character Alphabet ---")
    # Ensure dataset is available (e.g., by running download_dataset.py first)
    # Example: python download_dataset.py
    
    generated_alphabet = build_and_save_alphabet()
    
    if generated_alphabet:
        print("\nScript finished. The alphabet can now be loaded from 'alphabet.json'.")
    else:
        print("\nScript finished with errors. Alphabet generation may have failed.")
