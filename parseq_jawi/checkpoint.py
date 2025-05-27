import os
import sys
from pytorch_lightning.utilities.cloud_io import load as pl_load
from strhub.models.utils import load_from_checkpoint
from strhub.data.dataset import LmdbDataset
from torch.utils.data import DataLoader
import torch
import lmdb
import io
from PIL import Image

def check_checkpoint_quality(ckpt_path, lmdb_path, max_label_len=25, sample_count=5):
    # 1️⃣ Load raw checkpoint to extract hyperparameters
    raw = pl_load(ckpt_path, map_location="cpu")
    hp = raw.get('hyper_parameters', raw.get('hyperparameters', {}))
    model_cfg = hp.get('model', {})
    charset = model_cfg.get('charset_train') or model_cfg.get('charset_test')
    if charset is None:
        print("❌ Could not find charset_train or charset_test in checkpoint hyperparameters.")
        return
    print(f"✅ Checkpoint charset length: {len(charset)}")

    # 2️⃣ Instantiate model and move to GPU if available
    model = load_from_checkpoint(ckpt_path).eval()
    if torch.cuda.is_available():
        model = model.cuda()
    total_params = sum(p.numel() for p in model.parameters())
    trainable   = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters   : {total_params:,}")
    print(f"Trainable params   : {trainable:,}")

    # 3️⃣ Load dataset with extracted charset
    ds = LmdbDataset(lmdb_path, charset, max_label_len)
    print(f"Dataset length seen by LmdbDataset: {len(ds)}")

    # 4️⃣ Spot-check a few predictions
    loader = DataLoader(ds, batch_size=1, num_workers=4, shuffle=False)
    print("\nSample predictions:")
    for i, (img_tensor, label) in enumerate(loader):
        if i >= sample_count:
            break
        # convert tensor to PIL
        arr = (img_tensor[0].permute(1,2,0).cpu().numpy() * 255).astype('uint8')
        img = Image.fromarray(arr)
        pred = model.predict([img])[0]
        print(f"{i+1}. GT='{' '.join(label)}'  ->  PRED='{pred}'")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', help='Path to model checkpoint')
    parser.add_argument('--lmdb', default='data/test/jawi', help='Path to test LMDB')
    parser.add_argument('--max_len', type=int, default=25, help='Max label length')
    parser.add_argument('--samples', type=int, default=5, help='Number of samples to inspect')
    args = parser.parse_args()
    check_checkpoint_quality(args.checkpoint, args.lmdb, args.max_len, args.samples)
