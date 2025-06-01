# evaluate.py
import os
import json
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import editdistance # For CER calculation as in Cell 12

from dataset import LMDBOCRDataset, DEFAULT_LMDB_BASE_PATH
from model import CRNN
from build_alphabet import DEFAULT_ALPHABET_OUTPUT_FILE # To load alphabet

# --- Configuration ---
IMG_HEIGHT = 32
IMG_WIDTH = 128
NUM_HIDDEN_RNN = 256
NUM_INPUT_CHANNELS = 1 # Grayscale

DEFAULT_CHECKPOINT_PATH = os.path.join("checkpoints", "best_crnn.pth")
LMDB_DATA_BASE_PATH = DEFAULT_LMDB_BASE_PATH
ALPHABET_FILE = DEFAULT_ALPHABET_OUTPUT_FILE
BATCH_SIZE = 64
MAX_SAMPLES_TO_PRINT = 30

def main(checkpoint_path=DEFAULT_CHECKPOINT_PATH):
    # --- 1. Setup Device ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- 2. Load Alphabet (Essential for decoding) ---
    if not os.path.exists(ALPHABET_FILE):
        print(f"Error: Alphabet file '{ALPHABET_FILE}' not found.")
        print("Please run 'build_alphabet.py' first to generate the alphabet.")
        return
    try:
        with open(ALPHABET_FILE, 'r', encoding='utf-8') as f:
            alphabet = json.load(f)
    except Exception as e:
        print(f"Error loading alphabet from '{ALPHABET_FILE}': {e}")
        return
    
    n_class = len(alphabet) + 1
    print(f"Alphabet loaded from '{ALPHABET_FILE}': {len(alphabet)} characters, {n_class} classes.")

    # --- 3. Load Checkpoint (Model State Dictionary ONLY) ---
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file '{checkpoint_path}' not found.")
        print("Please ensure 'train.py' has been run and a checkpoint is saved, or provide correct path.")
        return

    print(f"Loading model state_dict from '{checkpoint_path}'...")
    try:
        # Load the state_dict directly
        model_state_dict = torch.load(checkpoint_path, map_location=device)
    except Exception as e:
        print(f"Error loading model state_dict: {e}")
        return

    # --- 4. Initialize Model and Load State ---
    model = CRNN(imgH=IMG_HEIGHT, nc=NUM_INPUT_CHANNELS, nclass=n_class, nh=NUM_HIDDEN_RNN)
    try:
        model.load_state_dict(model_state_dict)
    except RuntimeError as e:
        print(f"Error loading state_dict into model: {e}")
        print("Ensure model architecture in `model.py` matches the saved checkpoint.")
        print(f"Expected architecture params: imgH={IMG_HEIGHT}, nc={NUM_INPUT_CHANNELS}, nclass={n_class}, nh={NUM_HIDDEN_RNN}")
        return
        
    model = model.to(device)
    model.eval()
    print("Model initialized and state loaded. Set to evaluation mode.")

    # --- 5. Prepare Test DataLoader ---
    dataset_args = dict(
        alphabet=alphabet,
        imgH=IMG_HEIGHT,
        imgW=IMG_WIDTH,
        base_path=LMDB_DATA_BASE_PATH
    )
    try:
        test_ds = LMDBOCRDataset(lmdb_path_suffix="test", **dataset_args)
    except FileNotFoundError as e:
        print(f"Error initializing test dataset: {e}")
        print(f"Ensure the 'test' split exists at '{os.path.join(LMDB_DATA_BASE_PATH, 'test')}'.")
        return
    except ValueError as e:
        print(f"Error initializing test dataset: {e}")
        return

    if len(test_ds) == 0:
        print("Test dataset is empty. Cannot perform evaluation.")
        return

    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=LMDBOCRDataset.collate_fn, num_workers=2, pin_memory=True
    )
    print(f"Test DataLoader created with {len(test_ds)} samples in {len(test_loader)} batches.")

    # --- 6. Evaluation Loop (Directly from Cell 12) ---
    total_cer_metric, total_wer_metric = 0.0, 0.0 # Renamed from total_cer, total_wer
    num_evaluated_samples = 0 # Renamed from n_samples
    printed_samples_list = [] # Renamed from samples

    print(f"\n--- Starting Evaluation on Test Set ---")
    test_pbar = tqdm(test_loader, desc='Evaluating on Test Set')

    with torch.no_grad():
        for imgs, labels_concat, target_lengths in test_pbar:
            imgs = imgs.to(device)
            # labels_concat and target_lengths are on CPU via collate_fn, keep them there for decoding

            preds_model = model(imgs)
            # preds_model.log_softmax(2) # log_softmax is in Cell 12, but not strictly needed for .max(2)
            # Let's keep it for consistency with Cell 12, though .max(2) on raw logits is same as on log_softmax
            preds_log_softmax = preds_model.log_softmax(2) 
            _, pred_indices_batch = preds_log_softmax.cpu().max(2)
            
            # Similar to validation loop, this start_idx is for current batch's concatenated labels
            batch_label_start_idx = 0
            for i in range(imgs.size(0)):
                pred_indices_sample = pred_indices_batch[:, i].tolist()
                
                pred_text = ''.join(
                    alphabet[c-1] for j, c in enumerate(pred_indices_sample)
                    if c != 0 and (j == 0 or c != pred_indices_sample[j-1])
                )
                
                current_target_len = target_lengths[i].item() # target_lengths is already on CPU
                target_indices_sample = labels_concat[batch_label_start_idx : batch_label_start_idx + current_target_len].tolist()
                tgt_text = ''.join(
                    alphabet[c-1] for c in target_indices_sample # Original was 'c', ensuring it's general
                )
                batch_label_start_idx += current_target_len

                if not tgt_text:
                    continue

                if len(printed_samples_list) < MAX_SAMPLES_TO_PRINT:
                    printed_samples_list.append((tgt_text, pred_text))

                # CER (using editdistance directly as in Cell 12)
                total_cer_metric += editdistance.eval(pred_text, tgt_text) / len(tgt_text)
                # WER (as in Cell 12)
                total_wer_metric += float(pred_text.split() != tgt_text.split())
                num_evaluated_samples += 1

    # --- 7. Report Results ---
    print(f"\n--- Evaluation Finished ---")
    if num_evaluated_samples > 0:
        avg_test_cer = total_cer_metric / num_evaluated_samples
        avg_test_wer = total_wer_metric / num_evaluated_samples
        
        print(f"\nTotal samples evaluated: {num_evaluated_samples}")
        print(f"Test CER: {avg_test_cer:.4f}")
        print(f"Test WER: {avg_test_wer:.4f}")

        print(f"\nFirst {min(len(printed_samples_list), MAX_SAMPLES_TO_PRINT)} (GT → Pred):")
        for idx, (gt, pr) in enumerate(printed_samples_list, 1):
            print(f"{idx:2d}. {gt!r:<50}  →  {pr!r}")
    else:
        print("No samples were evaluated from the test set.")

if __name__ == "__main__":
    main()
