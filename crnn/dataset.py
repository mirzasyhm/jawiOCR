# dataset.py
import os
import lmdb
import numpy as np
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

# Default base path for the dataset, assuming extraction into './content/'
DEFAULT_LMDB_BASE_PATH = os.path.join("content", "data_jawi_color_2_lmdb")

class LMDBOCRDataset(Dataset):
    """
    Dataset class to read images and labels from an LMDB database for OCR.
    """
    def __init__(self, lmdb_path_suffix="train", alphabet=None, imgH=32, imgW=128, base_path=DEFAULT_LMDB_BASE_PATH):
        """
        Args:
            lmdb_path_suffix (str): Suffix for the LMDB environment (e.g., "train", "test").
                                    This will be appended to base_path.
            alphabet (list or str): A list or string of unique characters in the dataset.
                                    Required for encoding labels.
            imgH (int): Target image height for resizing.
            imgW (int): Target image width for resizing.
            base_path (str): The base directory where the LMDB dataset (e.g., 'data_jawi_color_2_lmdb') is located.
        """
        full_lmdb_path = os.path.join(base_path, lmdb_path_suffix)
        
        if not os.path.exists(full_lmdb_path):
            raise FileNotFoundError(f"LMDB path {full_lmdb_path} does not exist. "
                                    "Ensure the dataset is downloaded and extracted correctly, "
                                    "and paths are set up as expected.")

        # Open LMDB environment for reading
        self.env = lmdb.open(full_lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            num_samples_bytes = txn.get(b'num-samples')
            if num_samples_bytes is None:
                raise ValueError(f"'num-samples' key not found in LMDB at {full_lmdb_path}. "
                                 "Dataset might be corrupted or not in the expected format.")
            self.nSamples = int(num_samples_bytes.decode())
        
        if alphabet is None:
            raise ValueError("Alphabet must be provided to LMDBOCRDataset.")
        
        self.alphabet_str = "".join(alphabet) if isinstance(alphabet, list) else alphabet
        # Create char -> index map (0 is reserved for CTC blank)
        self.char2idx = {char: i + 1 for i, char in enumerate(self.alphabet_str)}
        self.idx2char = {i + 1: char for i, char in enumerate(self.alphabet_str)}
        self.idx2char[0] = '[CTC_BLANK]' # CTC blank character representation

        self.imgH, self.imgW = imgH, imgW
        self.transform = transforms.Compose([
            transforms.Resize((self.imgH, self.imgW)),
            transforms.Grayscale(), # Convert image to grayscale
            transforms.ToTensor(),  # Convert image to PyTorch tensor (scales to [0,1])
            transforms.Normalize(mean=[0.5], std=[0.5]) # Normalize to [-1,1]
        ])

    def __len__(self):
        return self.nSamples

    def __getitem__(self, idx):
        if not 0 <= idx < self.nSamples:
            raise IndexError(f"Index {idx} out of range for dataset with {self.nSamples} samples.")

        # LMDB keys in this dataset are 1-indexed
        item_idx = idx + 1
        with self.env.begin(write=False) as txn:
            img_key = f'image-{item_idx:09d}'.encode()
            lbl_key = f'label-{item_idx:09d}'.encode()
            
            img_bin = txn.get(img_key)
            if img_bin is None:
                raise KeyError(f"Image with key '{img_key.decode()}' not found for index {idx}.")
            
            label_str_bytes = txn.get(lbl_key)
            if label_str_bytes is None:
                raise KeyError(f"Label with key '{lbl_key.decode()}' not found for index {idx}.")
            label_str = label_str_bytes.decode('utf-8')

        # Process image
        # Decode image from binary buffer
        arr = np.frombuffer(img_bin, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR) # Reads as BGR by default
        if img is None:
            raise IOError(f"cv2.imdecode failed for image key {img_key.decode()} at index {idx}. Image data might be corrupt.")
        
        # Convert from BGR (OpenCV) to RGB (Pillow) before Pillow operations
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        
        transformed_img = self.transform(img_pil)
        
        # Encode label
        try:
            label_encoded = torch.tensor([self.char2idx[char] for char in label_str], dtype=torch.long)
        except KeyError as e:
            missing_char = str(e).strip("'")
            raise ValueError(f"Character '{missing_char}' in label '{label_str}' (index {idx}) "
                             f"not found in the provided alphabet. Please ensure the alphabet is correct.")
            
        return transformed_img, label_encoded, torch.tensor(len(label_encoded), dtype=torch.int32)

    @staticmethod
    def collate_fn(batch):
        """
        Collates data samples into batches.
        Suitable for CTC-based models where labels are concatenated and lengths are provided.
        """
        imgs, labels, lengths = zip(*batch)
        
        imgs_stacked = torch.stack(imgs, 0)
        labels_concatenated = torch.cat(labels, 0)
        lengths_tensor = torch.stack(lengths, 0) # Or simply torch.tensor(lengths, dtype=torch.int32)
        
        return imgs_stacked, labels_concatenated, lengths_tensor

    def decode_prediction(self, pred_indices):
        """
        Decodes a list/tensor of predicted indices (output from a model after argmax) 
        into a string, handling CTC blank and collapsing repeated characters.
        """
        # Ensure pred_indices is a list of integers or a 1D tensor
        if isinstance(pred_indices, torch.Tensor):
            pred_indices = pred_indices.squeeze().tolist()

        chars = []
        last_char_idx = 0 # Represents CTC blank
        for idx in pred_indices:
            if idx == last_char_idx: # Skip repeated character
                continue
            if idx == 0: # CTC Blank
                last_char_idx = 0
                continue
            # Check if index is in idx2char (it should be if model output classes match dataset)
            if idx in self.idx2char:
                chars.append(self.idx2char[idx])
            else:
                # Handle unknown index, e.g., by skipping or adding a placeholder
                # For now, we assume valid indices.
                print(f"Warning: Unknown index {idx} in prediction decoding.")

            last_char_idx = idx
        return "".join(chars)
