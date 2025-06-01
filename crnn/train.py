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
from utils import calculate_cer # Renamed from cer to calculate_cer in utils.py
from build_alphabet import DEFAULT_ALPHABET_OUTPUT_FILE

# --- Configuration ---
BATCH_SIZE = 64 # Adjusted from 512 for potentially wider accessibility
IMG_HEIGHT = 32
IMG_WIDTH = 128
NUM_HIDDEN_RNN = 256
EPOCHS = 25
LEARNING_RATE = 1e-3
CHECKPOINT_DIR = "checkpoints"
PATIENCE_LR_SCHEDULER = 2
ALPHABET_FILE = DEFAULT_ALPHABET_OUTPUT_FILE
LMDB_DATA_BASE_PATH = DEFAULT_LMDB_BASE_PATH # From dataset.py

def main():
    # --- 1. Setup Device ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- 2. Load Alphabet ---
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
    
    n_class = len(alphabet) + 1  # +1 for CTC blank
    print(f"Alphabet loaded: {len(alphabet)} characters, {n_class} classes (incl. blank).")

    # --- 3. Prepare DataLoaders (Cell 8) ---
    dataset_args = dict(
        alphabet=alphabet,
        imgH=IMG_HEIGHT,
        imgW=IMG_WIDTH,
        base_path=LMDB_DATA_BASE_PATH
    )

    try:
        train_ds = LMDBOCRDataset(lmdb_path_suffix="train", **dataset_args)
        val_ds = LMDBOCRDataset(lmdb_path_suffix="val", **dataset_args)
        # test_ds = LMDBOCRDataset(lmdb_path_suffix="test", **dataset_args) # Test set can be loaded if needed later
    except FileNotFoundError as e:
        print(f"Error initializing dataset: {e}")
        print(f"Ensure the dataset exists at '{LMDB_DATA_BASE_PATH}' and its subdirectories (train, val).")
        print("You might need to run 'download_dataset.py'.")
        return
    except ValueError as e: # For alphabet issues or missing num-samples
        print(f"Error initializing dataset: {e}")
        return

    collate_fn = LMDBOCRDataset.collate_fn # Use the static method

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=4, pin_memory=True
    )
    # test_loader = DataLoader(
    #     test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=4, pin_memory=True
    # )
    print(f"DataLoaders created. Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # --- 4. Define CRNN Model, Loss, Optimizer (Cell 10) ---
    # nc = 1 for grayscale (as per LMDBOCRDataset transform)
    model = CRNN(imgH=IMG_HEIGHT, nc=1, nclass=n_class, nh=NUM_HIDDEN_RNN).to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True) # blank=0 corresponds to the CTC blank token
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # AMP Scaler and LR Scheduler
    # GradScaler is only enabled if device is CUDA
    scaler = GradScaler(enabled=(device.type == 'cuda'))
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=PATIENCE_LR_SCHEDULER)
    
    print(f"Model, Criterion, Optimizer, Scaler, Scheduler initialized.")

    # --- 5. Training Loop (Cell 11) ---
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    best_val_cer = float('inf')
    best_val_loss = float('inf')

    print(f"\n--- Starting Training for {EPOCHS} epochs ---")
    for epoch in range(1, EPOCHS + 1):
        # --- Training Phase ---
        model.train()
        total_train_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{EPOCHS} [Train]')
        
        for imgs, labels_concat, target_lengths in train_pbar:
            imgs = imgs.to(device)
            labels_concat = labels_concat.to(device)
            target_lengths = target_lengths.to(device) # Ensure target_lengths is also on device
            
            optimizer.zero_grad()
            
            # Mixed precision training if CUDA
            with autocast(enabled=(device.type == 'cuda')):
                preds = model(imgs) # Output: (seq_len, batch_size, n_class)
                preds_log_softmax = preds.log_softmax(2) # CTCLoss expects log_softmax
                
                # CTCLoss expects:
                # preds: (T, N, C) where T=input_seq_len, N=batch_size, C=n_class
                # labels_concat: (sum of target_lengths)
                # input_lengths: (N,) -> sequence lengths of preds for each item in batch
                # target_lengths: (N,) -> actual lengths of targets
                
                preds_seq_len = preds_log_softmax.size(0)
                input_lengths = torch.full(size=(imgs.size(0),), fill_value=preds_seq_len, dtype=torch.long).to(device)
                
                loss = criterion(preds_log_softmax, labels_concat, input_lengths, target_lengths)

            if torch.isinf(loss) or torch.isnan(loss):
                print(f"Warning: Encountered inf or nan loss at epoch {epoch}, batch. Skipping update.")
                # Potentially log more details or stop training
                # preds_for_debug = preds.detach().cpu().numpy()
                # labels_for_debug = labels_concat.detach().cpu().numpy()
                # input_len_debug = input_lengths.detach().cpu().numpy()
                # target_len_debug = target_lengths.detach().cpu().numpy()
                # print(f"Preds shape: {preds_for_debug.shape}, Labels: {labels_for_debug}, Input Lens: {input_len_debug}, Target Lens: {target_len_debug}")
                # This often happens with CTCLoss if input_lengths are too short for target_lengths.
                # Or if some targets are empty.
                del loss, preds, preds_log_softmax # Free memory
                torch.cuda.empty_cache() if device.type == 'cuda' else None
                continue

            if device.type == 'cuda':
                scaler.scale(loss).backward()
                # Optional: Gradient clipping (can be helpful for RNNs)
                # scaler.unscale_(optimizer) # Unscale before clipping
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
            else: # CPU
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0) # if clipping
                optimizer.step()
                
            total_train_loss += loss.item()
            train_pbar.set_postfix(loss=loss.item())

        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Epoch {epoch}/{EPOCHS} - Avg Train Loss: {avg_train_loss:.4f}")

        # --- Validation Phase ---
        model.eval()
        total_val_loss = 0.0
        total_val_cer = 0.0
        num_val_samples = 0
        
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch}/{EPOCHS} [Validate]')
        
        with torch.no_grad():
            for imgs, labels_concat, target_lengths in val_pbar:
                imgs = imgs.to(device)
                labels_concat = labels_concat.to(device)
                target_lengths = target_lengths.to(device)

                # No autocast for validation usually, unless it was critical for speed AND showed no perf drop
                preds = model(imgs) # (seq_len, batch_size, n_class)
                preds_log_softmax = preds.log_softmax(2)
                
                preds_seq_len = preds_log_softmax.size(0)
                input_lengths = torch.full(size=(imgs.size(0),), fill_value=preds_seq_len, dtype=torch.long).to(device)
                
                val_loss = criterion(preds_log_softmax, labels_concat, input_lengths, target_lengths)
                total_val_loss += val_loss.item()

                # Greedy decode for CER calculation
                # preds shape: (T, N, C)
                _, pred_indices_batch = preds.cpu().max(2) # (T, N) -> sequence of max prob indices per timestep
                
                current_labels_idx = 0
                for i in range(imgs.size(0)): # Iterate through batch
                    pred_indices_sample = pred_indices_batch[:, i].tolist() # List of T indices for one sample
                    
                    # Decode predicted text
                    decoded_pred_text = ""
                    last_char_idx = 0 # CTC blank
                    for char_idx in pred_indices_sample:
                        if char_idx == 0: # CTC Blank
                            last_char_idx = 0
                            continue
                        if char_idx == last_char_idx: # Repeated character
                            continue
                        # char_idx is 1-based for actual characters from model output
                        decoded_pred_text += alphabet[char_idx - 1]
                        last_char_idx = char_idx
                    
                    # Decode target text
                    current_target_len = target_lengths[i].item()
                    target_indices_sample = labels_concat[current_labels_idx : current_labels_idx + current_target_len].cpu().tolist()
                    decoded_target_text = "".join([alphabet[idx - 1] for idx in target_indices_sample])
                    current_labels_idx += current_target_len

                    if not decoded_target_text: # Skip if target is empty (though should not happen with good data)
                        continue
                        
                    total_val_cer += calculate_cer(decoded_pred_text, decoded_target_text)
                    num_val_samples += 1
                
                val_pbar.set_postfix(val_loss=val_loss.item())

        avg_val_loss = total_val_loss / len(val_loader) if len(val_loader) > 0 else 0
        avg_val_cer = total_val_cer / num_val_samples if num_val_samples > 0 else float('inf')
        
        print(f"Epoch {epoch}/{EPOCHS} - Avg Val Loss: {avg_val_loss:.4f}, Avg Val CER: {avg_val_cer:.4f}")

        # --- Scheduler Step & Checkpointing ---
        scheduler.step(avg_val_loss) # Step scheduler based on validation loss
        
        # Save checkpoint if validation CER improved
        if avg_val_cer < best_val_cer:
            best_val_cer = avg_val_cer
            best_val_loss_at_best_cer = avg_val_loss # Store loss when best CER was found
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f'best_crnn_cer.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_cer': best_val_cer,
                'best_val_loss': best_val_loss_at_best_cer,
                'alphabet': alphabet # Save alphabet for easy inference later
            }, checkpoint_path)
            print(f"→ New best CER: {best_val_cer:.4f}. Checkpoint saved to '{checkpoint_path}'")
        
        # Optionally, also save based on best validation loss if desired
        # if avg_val_loss < best_val_loss:
        #     best_val_loss = avg_val_loss
        #     # ... save another checkpoint or update a 'best_loss_model.pth'

    print(f"\n--- Training Finished ---")
    print(f"Best Validation CER achieved: {best_val_cer:.4f}")

if __name__ == "__main__":
    main()
