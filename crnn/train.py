# train.py
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.amp import GradScaler, autocast # Using torch.amp for consistency
from tqdm import tqdm

from dataset import LMDBOCRDataset, DEFAULT_LMDB_BASE_PATH
from model import CRNN
from utils import cer
from build_alphabet import DEFAULT_ALPHABET_OUTPUT_FILE

# --- Configuration ---
IMG_HEIGHT = 32
IMG_WIDTH = 128
NUM_INPUT_CHANNELS = 1
NUM_HIDDEN_RNN = 256
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 3
CHECKPOINT_DIR = "checkpoints"
PATIENCE_LR_SCHEDULER = 2
ALPHABET_FILE = DEFAULT_ALPHABET_OUTPUT_FILE
LMDB_DATA_BASE_PATH = DEFAULT_LMDB_BASE_PATH

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if not os.path.exists(ALPHABET_FILE):
        print(f"Error: Alphabet file '{ALPHABET_FILE}' not found. Run 'build_alphabet.py'.")
        return
    try:
        with open(ALPHABET_FILE, 'r', encoding='utf-8') as f:
            alphabet = json.load(f)
    except Exception as e:
        print(f"Error loading alphabet from '{ALPHABET_FILE}': {e}")
        return
    
    n_class = len(alphabet) + 1
    print(f"Alphabet loaded: {len(alphabet)} characters, {n_class} classes (incl. blank).")

    common_dataset_params = dict(
        alphabet=alphabet, imgH=IMG_HEIGHT, imgW=IMG_WIDTH, base_path=LMDB_DATA_BASE_PATH
    )
    try:
        train_ds = LMDBOCRDataset(lmdb_path_suffix="train", **common_dataset_params)
        val_ds = LMDBOCRDataset(lmdb_path_suffix="val", **common_dataset_params)
    except Exception as e:
        print(f"Error initializing dataset: {e}")
        print(f"Ensure dataset exists at '{LMDB_DATA_BASE_PATH}' and alphabet at '{ALPHABET_FILE}'.")
        return

    collate_fn = LMDBOCRDataset.collate_fn
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=4, pin_memory=True
    )
    print(f"DataLoaders created. Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    model = CRNN(imgH=IMG_HEIGHT, nc=NUM_INPUT_CHANNELS, nclass=n_class, nh=NUM_HIDDEN_RNN).to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # CORRECTED GradScaler initialization (reverted based on error and search result [3])
    scaler = GradScaler(enabled=(device.type == 'cuda'))
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=PATIENCE_LR_SCHEDULER)
    
    print(f"Model, Criterion, Optimizer, Scaler, Scheduler initialized.")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    best_cer_val = float('inf')

    print(f"\n--- Starting Training for {EPOCHS} epochs ---")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_train_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{EPOCHS} [Train]')
        
        for imgs, labels_concat, target_lengths in train_pbar:
            imgs = imgs.to(device)
            labels_concat = labels_concat.to(device)
            target_lengths = target_lengths.to(device)
            
            optimizer.zero_grad(set_to_none=True)
            
            # autocast still uses device_type
            with autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                preds = model(imgs)
                preds_log_softmax = preds.log_softmax(2)
                preds_seq_len = preds_log_softmax.size(0)
                input_lengths = torch.full(size=(imgs.size(0),), fill_value=preds_seq_len, dtype=torch.long, device=device)
                loss = criterion(preds_log_softmax, labels_concat, input_lengths, target_lengths)

            if torch.isinf(loss) or torch.isnan(loss):
                print(f"Warning: Inf/NaN loss at epoch {epoch}. Skipping update.")
                del loss, preds, preds_log_softmax
                if device.type == 'cuda': torch.cuda.empty_cache()
                continue

            if device.type == 'cuda' and scaler.is_enabled(): # Check if scaler is enabled
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else: # CPU or scaler not enabled
                loss.backward()
                optimizer.step()
                
            total_train_loss += loss.item()
            train_pbar.set_postfix(loss=loss.item())

        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Epoch {epoch}/{EPOCHS} - Train Loss: {avg_train_loss:.4f}")

        model.eval()
        current_val_loss = 0.0
        total_val_cer_metric = 0.0
        num_val_samples = 0
        
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch}/{EPOCHS} [Validate]')
        
        with torch.no_grad():
            for imgs, labels_concat_val, target_lengths_val in val_pbar:
                imgs = imgs.to(device)
                labels_concat_val_dev = labels_concat_val.to(device)
                target_lengths_val_dev = target_lengths_val.to(device)

                # For validation, autocast usage should match training if desired,
                # or be omitted for full precision.
                # Assuming validation without autocast here for simplicity unless specified.
                preds = model(imgs)
                preds_log_softmax = preds.log_softmax(2)
                
                preds_seq_len = preds_log_softmax.size(0)
                input_lengths_val = torch.full(size=(imgs.size(0),), fill_value=preds_seq_len, dtype=torch.long, device=device)
                
                batch_val_loss = criterion(preds_log_softmax, labels_concat_val_dev, input_lengths_val, target_lengths_val_dev)
                current_val_loss += batch_val_loss.item()

                _, max_indices_batch = preds.cpu().max(2)
                
                labels_concat_val_cpu = labels_concat_val.cpu()
                target_lengths_val_cpu = target_lengths_val.cpu()

                batch_label_start_idx = 0
                for i in range(imgs.size(0)):
                    pred_indices_sample = max_indices_batch[:, i].tolist()
                    pred_text = ''.join(
                        alphabet[c-1] for j, c in enumerate(pred_indices_sample)
                        if c != 0 and (j == 0 or c != pred_indices_sample[j-1])
                    )
                    current_target_len = target_lengths_val_cpu[i].item()
                    target_indices_sample = labels_concat_val_cpu[batch_label_start_idx : batch_label_start_idx + current_target_len].tolist()
                    tgt_text = ''.join(alphabet[idx - 1] for idx in target_indices_sample)
                    batch_label_start_idx += current_target_len

                    if not tgt_text: continue
                    total_val_cer_metric += cer(pred_text, tgt_text)
                    num_val_samples += 1
                
                val_pbar.set_postfix(val_loss=batch_val_loss.item())

        avg_val_loss = current_val_loss / len(val_loader) if len(val_loader) > 0 else 0
        avg_val_cer = total_val_cer_metric / num_val_samples if num_val_samples > 0 else float('inf')
        
        print(f"Epoch {epoch}/{EPOCHS} - Val Loss: {avg_val_loss:.4f}, Val CER: {avg_val_cer:.4f}")

        scheduler.step(avg_val_loss)
        if avg_val_cer < best_cer_val:
            best_cer_val = avg_val_cer
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, 'best_crnn.pth'))
            print(f"→ New best CER: {best_cer_val:.4f}, checkpoint saved.")

    print(f"\n--- Training Finished ---")
    print(f"Best Validation CER achieved: {best_cer_val:.4f}")

if __name__ == "__main__":
    main()

