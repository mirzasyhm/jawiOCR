# train.py
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from dataset import LMDBOCRDataset, DEFAULT_LMDB_BASE_PATH
from model import CRNN
from utils import cer # Now importing 'cer' directly from utils.py
from build_alphabet import DEFAULT_ALPHABET_OUTPUT_FILE

# --- Configuration (from Cell 10 and implied by Cell 8/11) ---
IMG_HEIGHT = 32
IMG_WIDTH = 128 # Dataset param
NUM_INPUT_CHANNELS = 1 # Grayscale from dataset transform
NUM_HIDDEN_RNN = 256
# BATCH_SIZE = 512 # From Cell 8, adjust if memory issues
BATCH_SIZE = 64 # A more common default, user can change
LEARNING_RATE = 1e-3
EPOCHS = 25 # From Cell 11
CHECKPOINT_DIR = "checkpoints"
PATIENCE_LR_SCHEDULER = 2 # From Cell 10 setup
ALPHABET_FILE = DEFAULT_ALPHABET_OUTPUT_FILE
LMDB_DATA_BASE_PATH = DEFAULT_LMDB_BASE_PATH # From dataset.py

def main():
    # --- 1. Setup Device (from Cell 10) ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- 2. Load Alphabet (needed for nclass and validation decoding) ---
    if not os.path.exists(ALPHABET_FILE):
        print(f"Error: Alphabet file '{ALPHABET_FILE}' not found.")
        print("Please run 'build_alphabet.py' first to generate the alphabet.")
        return
    try:
        with open(ALPHABET_FILE, 'r', encoding='utf-8') as f:
            alphabet = json.load(f) # This 'alphabet' is the list of characters
    except Exception as e:
        print(f"Error loading alphabet from '{ALPHABET_FILE}': {e}")
        return
    
    n_class = len(alphabet) + 1  # +1 for CTC blank (from Cell 7 logic)
    print(f"Alphabet loaded: {len(alphabet)} characters, {n_class} classes (incl. blank).")

    # --- 3. Prepare DataLoaders (from Cell 8) ---
    # Note: In Cell 8, dataset_args uses 'lmdb_path' directly.
    # Here, we use 'base_path' and 'lmdb_path_suffix' for LMDBOCRDataset.
    # This keeps paths relative and configurable.
    # The paths in Cell 8:
    # '/content/data_jawi_color_2_lmdb/train'
    # '/content/data_jawi_color_2_lmdb/val'
    # '/content/data_jawi_color_2_lmdb/test'
    # These map to base_path = 'content/data_jawi_color_2_lmdb' and suffix = 'train'/'val'/'test'

    common_dataset_params = dict(
        alphabet=alphabet, # The loaded list of characters
        imgH=IMG_HEIGHT,
        imgW=IMG_WIDTH,
        base_path=LMDB_DATA_BASE_PATH
    )

    try:
        train_ds = LMDBOCRDataset(lmdb_path_suffix="train", **common_dataset_params)
        val_ds = LMDBOCRDataset(lmdb_path_suffix="val", **common_dataset_params)
        # test_ds for evaluation is handled in evaluate.py
    except FileNotFoundError as e:
        print(f"Error initializing dataset: {e}")
        print(f"Ensure the dataset exists at '{LMDB_DATA_BASE_PATH}' and its subdirectories (train, val).")
        print("You might need to run 'download_dataset.py'.")
        return
    except ValueError as e:
        print(f"Error initializing dataset: {e}")
        return

    collate_fn = LMDBOCRDataset.collate_fn

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=4, pin_memory=True
    )
    print(f"DataLoaders created. Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # --- 4. Define Model, Loss, Optimizer (from Cell 9 & 10) ---
    model = CRNN(imgH=IMG_HEIGHT, nc=NUM_INPUT_CHANNELS, nclass=n_class, nh=NUM_HIDDEN_RNN).to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    scaler = GradScaler(enabled=(device.type == 'cuda')) # From Cell 10, uses 'cuda' by default under torch.amp
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=PATIENCE_LR_SCHEDULER) # From Cell 10
    
    print(f"Model, Criterion, Optimizer, Scaler, Scheduler initialized.")

    # --- 5. Training Loop (Directly from Cell 11) ---
    os.makedirs(CHECKPOINT_DIR, exist_ok=True) # From Cell 11
    best_cer_val = float('inf') # Renamed from best_cer to avoid conflict if cer is also a var name

    print(f"\n--- Starting Training for {EPOCHS} epochs ---")
    for epoch in range(1, EPOCHS + 1):
        # --- Training ---
        model.train()
        total_train_loss = 0.0 # Renamed from total_loss to be specific
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{EPOCHS} [Train]')
        
        for imgs, labels_concat, target_lengths in train_pbar: # Variable names from LMDBOCRDataset.collate_fn
            imgs = imgs.to(device)
            labels_concat = labels_concat.to(device)
            target_lengths = target_lengths.to(device) # Ensure target_lengths is also on device
            
            optimizer.zero_grad()
            
            # Mixed precision training context
            with autocast(device_type=device.type, enabled=(device.type == 'cuda')): # Pass device.type
                preds = model(imgs) # Output: (seq_len, batch_size, n_class)
                preds_log_softmax = preds.log_softmax(2) # CTCLoss expects log_softmax
                
                preds_seq_len = preds_log_softmax.size(0)
                input_lengths = torch.full(size=(imgs.size(0),), fill_value=preds_seq_len, dtype=torch.long).to(device)
                
                loss = criterion(preds_log_softmax, labels_concat, input_lengths, target_lengths)

            if torch.isinf(loss) or torch.isnan(loss):
                print(f"Warning: Encountered inf or nan loss at epoch {epoch}. Skipping update.")
                del loss, preds, preds_log_softmax
                torch.cuda.empty_cache() if device.type == 'cuda' else None
                continue

            if device.type == 'cuda':
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else: # CPU
                loss.backward()
                optimizer.step()
                
            total_train_loss += loss.item()
            train_pbar.set_postfix(loss=loss.item())

        avg_train_loss = total_train_loss / len(train_loader) # Renamed from avg_train
        print(f"Epoch {epoch}/{EPOCHS} - Train Loss: {avg_train_loss:.4f}")

        # --- Validation ---
        model.eval()
        current_val_loss = 0.0 # Renamed from val_loss
        total_val_cer_metric = 0.0 # Renamed from total_cer
        num_val_samples = 0 # Renamed from n_samples
        
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch}/{EPOCHS} [Validate]')
        
        with torch.no_grad():
            # start_idx for manually slicing concatenated labels
            # This is correct if labels_concat is on CPU and target_lengths is on CPU for indexing
            # If labels_concat and target_lengths are on GPU, .cpu() will be needed before slicing or using as Python int
            current_labels_idx_in_concat = 0 
            for imgs, labels_concat_val, target_lengths_val in val_pbar:
                imgs = imgs.to(device)
                labels_concat_val = labels_concat_val.to(device) # For loss calculation
                target_lengths_val = target_lengths_val.to(device) # For loss calculation

                preds = model(imgs)
                preds_log_softmax = preds.log_softmax(2)
                
                preds_seq_len = preds_log_softmax.size(0)
                input_lengths_val = torch.full(size=(imgs.size(0),), fill_value=preds_seq_len, dtype=torch.long).to(device)
                
                batch_val_loss = criterion(preds_log_softmax, labels_concat_val, input_lengths_val, target_lengths_val)
                current_val_loss += batch_val_loss.item()

                # Greedy decode + CER (as in Cell 11)
                _, max_indices_batch = preds.cpu().max(2) # (T, N)
                
                # For decoding target text, we need labels_concat_val and target_lengths_val on CPU
                labels_concat_val_cpu = labels_concat_val.cpu()
                target_lengths_val_cpu = target_lengths_val.cpu()

                # Reset current_labels_idx_in_concat for each batch if labels are batch-wise concatenated
                # The collate_fn concatenates all labels in a batch.
                # So start_idx logic from Cell 11 is correct if labels_concat is the *entire validation set's* labels concatenated.
                # However, DataLoader gives batch-wise concatenated labels.
                # Thus, start_idx needs to be reset for each batch for target decoding.

                batch_label_start_idx = 0
                for i in range(imgs.size(0)): # Iterate through batch
                    pred_indices_sample = max_indices_batch[:, i].tolist()
                    
                    pred_text = ''.join(
                        alphabet[c-1] for j, c in enumerate(pred_indices_sample)
                        if c != 0 and (j == 0 or c != pred_indices_sample[j-1]) # Ensure c != seq[j-1] logic
                    )
                    
                    current_target_len = target_lengths_val_cpu[i].item()
                    target_indices_sample = labels_concat_val_cpu[batch_label_start_idx : batch_label_start_idx + current_target_len].tolist()
                    tgt_text = ''.join(
                        alphabet[idx - 1] for idx in target_indices_sample # c is idx here
                    )
                    batch_label_start_idx += current_target_len

                    if not tgt_text:
                        continue
                    total_val_cer_metric += cer(pred_text, tgt_text) # Using imported cer
                    num_val_samples += 1
                
                val_pbar.set_postfix(val_loss=batch_val_loss.item())

        avg_val_loss = current_val_loss / len(val_loader) if len(val_loader) > 0 else 0
        avg_val_cer = total_val_cer_metric / num_val_samples if num_val_samples > 0 else float('inf')
        
        print(f"Epoch {epoch}/{EPOCHS} - Val Loss: {avg_val_loss:.4f}, Val CER: {avg_val_cer:.4f}")

        # --- Scheduler & Checkpoint (as in Cell 11) ---
        scheduler.step(avg_val_loss) # Original cell uses avg_val_loss
        if avg_val_cer < best_cer_val:
            best_cer_val = avg_val_cer
            # THIS IS THE CRUCIAL LINE FOR CHECKPOINT COMPATIBILITY
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, 'best_crnn.pth'))
            print(f"→ New best CER: {best_cer_val:.4f}, checkpoint saved.")

    print(f"\n--- Training Finished ---")
    print(f"Best Validation CER achieved: {best_cer_val:.4f}")

if __name__ == "__main__":
    main()
