# evaluate.py
import os
import json
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import editdistance # For CER calculation if not using utils.calculate_cer

from dataset import LMDBOCRDataset, DEFAULT_LMDB_BASE_PATH
from model import CRNN
# from utils import calculate_cer # You can use this or the direct editdistance.eval

# --- Configuration ---
# These should match the parameters used during training for the model architecture
IMG_HEIGHT = 32
IMG_WIDTH = 128 # Not directly used by CRNN init, but by dataset
NUM_HIDDEN_RNN = 256
NUM_INPUT_CHANNELS = 1 # Grayscale

# Paths and Files
DEFAULT_CHECKPOINT_PATH = os.path.join("checkpoints", "best_crnn.pth")
LMDB_DATA_BASE_PATH = DEFAULT_LMDB_BASE_PATH
BATCH_SIZE = 64 # Can be adjusted based on available memory for evaluation
MAX_SAMPLES_TO_PRINT = 30

def main(checkpoint_path=DEFAULT_CHECKPOINT_PATH):
    # --- 1. Setup Device ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- 2. Load Checkpoint and Alphabet ---
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file '{checkpoint_path}' not found.")
        print("Please ensure 'train.py' has been run and a checkpoint is saved.")
        return

    try:
        print(f"Loading checkpoint from '{checkpoint_path}'...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    # Extract alphabet and model state
    alphabet = checkpoint.get('alphabet')
    model_state_dict = checkpoint.get('model_state_dict')
    # Optionally, load training args if saved in checkpoint, to ensure consistency
    # train_img_height = checkpoint.get('imgH', IMG_HEIGHT) 
    # train_nc = checkpoint.get('nc', NUM_INPUT_CHANNELS)
    # train_nh = checkpoint.get('nh', NUM_HIDDEN_RNN)
    # epoch_trained = checkpoint.get('epoch', 'N/A')
    # best_val_cer_from_train = checkpoint.get('best_val_cer', 'N/A')

    if not alphabet or not model_state_dict:
        print("Error: Checkpoint is missing 'alphabet' or 'model_state_dict'.")
        return
        
    n_class = len(alphabet) + 1  # +1 for CTC blank
    print(f"Alphabet loaded from checkpoint: {len(alphabet)} characters, {n_class} classes.")
    # print(f"Model trained for {epoch_trained} epochs with best validation CER: {best_val_cer_from_train}")


    # --- 3. Initialize Model ---
    # Ensure these parameters match those used when the model was saved
    model = CRNN(imgH=IMG_HEIGHT, nc=NUM_INPUT_CHANNELS, nclass=n_class, nh=NUM_HIDDEN_RNN)
    model.load_state_dict(model_state_dict)
    model = model.to(device)
    model.eval()
    print("Model loaded and set to evaluation mode.")

    # --- 4. Prepare Test DataLoader ---
    dataset_args = dict(
        alphabet=alphabet,
        imgH=IMG_HEIGHT,
        imgW=IMG_WIDTH, # Used by dataset for transforms
        base_path=LMDB_DATA_BASE_PATH
    )

    try:
        test_ds = LMDBOCRDataset(lmdb_path_suffix="test", **dataset_args)
    except FileNotFoundError as e:
        print(f"Error initializing test dataset: {e}")
        print(f"Ensure the 'test' split exists at '{os.path.join(LMDB_DATA_BASE_PATH, 'test')}'.")
        return
    except ValueError as e: # For alphabet issues or missing num-samples
        print(f"Error initializing test dataset: {e}")
        return

    if len(test_ds) == 0:
        print("Test dataset is empty. Cannot perform evaluation.")
        return

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=LMDBOCRDataset.collate_fn, # Static method from LMDBOCRDataset
        num_workers=2, # Adjust as needed
        pin_memory=True
    )
    print(f"Test DataLoader created with {len(test_ds)} samples in {len(test_loader)} batches.")

    # --- 5. Evaluation Loop (Cell 12 logic) ---
    total_cer_val, total_wer_val = 0.0, 0.0
    num_evaluated_samples = 0
    printed_samples = []

    print(f"\n--- Starting Evaluation on Test Set ---")
    test_pbar = tqdm(test_loader, desc='Evaluating on Test Set')

    with torch.no_grad():
        for imgs, labels_concat, target_lengths in test_pbar:
            imgs = imgs.to(device)
            # labels_concat and target_lengths remain on CPU for decoding reference, 
            # but could be moved to device if used in loss calculation (not here)

            preds_model = model(imgs)  # Output: (seq_len, batch_size, n_class)
            # No log_softmax needed if just taking argmax for decoding
            
            # Greedy decode
            _, pred_indices_batch = preds_model.cpu().max(2) # (T, N) -> sequence of max prob indices
            
            current_labels_idx_in_concat = 0
            for i in range(imgs.size(0)): # Iterate through batch
                pred_indices_sample = pred_indices_batch[:, i].tolist() # List of T indices for one sample
                
                # Decode predicted text
                decoded_pred_text = ""
                last_char_idx = 0 # CTC blank
                for char_idx in pred_indices_sample:
                    if char_idx == 0: # CTC Blank
                        last_char_idx = 0
                        continue
                    if char_idx == last_char_idx: # Repeated character (already handled by ctc blank logic above)
                        pass # this check is more for non-blank repeats
                    
                    # char_idx is 1-based for actual characters from model output
                    # alphabet is 0-indexed
                    if 1 <= char_idx <= len(alphabet): # Ensure char_idx is valid
                        decoded_pred_text += alphabet[char_idx - 1]
                    else:
                        print(f"Warning: Invalid character index {char_idx} encountered in prediction.")
                    last_char_idx = char_idx
                
                # Decode target text
                current_target_len = target_lengths[i].item()
                target_indices_sample = labels_concat[current_labels_idx_in_concat : current_labels_idx_in_concat + current_target_len].tolist()
                decoded_target_text = "".join([alphabet[idx - 1] for idx in target_indices_sample if 1 <= idx <= len(alphabet)])
                current_labels_idx_in_concat += current_target_len

                if not decoded_target_text: # Should not happen with good data
                    # print("Warning: Empty target text encountered. Skipping sample.")
                    continue

                # Collect samples for printing
                if len(printed_samples) < MAX_SAMPLES_TO_PRINT:
                    printed_samples.append((decoded_target_text, decoded_pred_text))

                # Calculate CER for this sample
                # cer_sample = calculate_cer(decoded_pred_text, decoded_target_text) # Using utils
                cer_sample = editdistance.eval(decoded_pred_text, decoded_target_text) / len(decoded_target_text)
                total_cer_val += cer_sample
                
                # Calculate WER (simple version: 1 if different, 0 if same word lists)
                wer_sample = 1.0 if decoded_pred_text.split() != decoded_target_text.split() else 0.0
                total_wer_val += wer_sample
                
                num_evaluated_samples += 1

    # --- 6. Report Results ---
    print(f"\n--- Evaluation Finished ---")
    if num_evaluated_samples > 0:
        avg_test_cer = total_cer_val / num_evaluated_samples
        avg_test_wer = total_wer_val / num_evaluated_samples
        
        print(f"\nTotal samples evaluated: {num_evaluated_samples}")
        print(f"Test CER: {avg_test_cer:.4f}")
        print(f"Test WER (sentence-level exact match for words): {avg_test_wer:.4f}")

        print(f"\nFirst {min(len(printed_samples), MAX_SAMPLES_TO_PRINT)} (Ground Truth → Prediction) pairs:")
        for idx, (gt, pred) in enumerate(printed_samples, 1):
            # Using repr() for gt and pred to make spaces and special characters visible
            print(f"{idx:2d}. GT: {gt!r:<50} → Pred: {pred!r}")
    else:
        print("No samples were evaluated from the test set.")

if __name__ == "__main__":
    # You can pass a different checkpoint path if needed:
    # main(checkpoint_path="path/to/your/checkpoint.pth")
    main()
