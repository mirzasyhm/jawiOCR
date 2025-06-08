#!/usr/bin/env python3
import os
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from strhub.data.dataset import LmdbDataset
from strhub.models.utils import load_from_checkpoint
from strhub.data.module import SceneTextDataModule

# Your exact charset from training:
CHARSET = (
    " 0123456789۰۱۲٢۳۴۵۶۷۸۹"
    "اآأؤإءئۓۂئےۍېىيےیبپڀتٹثٿجچحخدڈذڎرڑزژسشصضطظعغفقڤڠݢکكڭگڬلمنںوۏههةۃۀہھڽضئکڤݢۏ-‌!\"#$%&'()*+,./:۔;<=>?@[\\]^_`{|}~"
)


def edit_distance(a, b):
    """Levenshtein distance between two sequences (chars or tokens)."""
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            cur = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j - 1], dp[j])
            prev = cur
    return dp[n]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data_root', default='data')
    parser.add_argument('--split', choices=['train','val','test'], default='val')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Using charset ({len(CHARSET)} chars)")

    # Load model
    model = load_from_checkpoint(
        args.checkpoint,
        charset_test=CHARSET
    ).to(device).eval()
    hp = model.hparams

    # Use same transform as training
    transform = SceneTextDataModule.get_transform(hp.img_size)

    # Prepare dataset and loader
    lmdb_path = os.path.join(args.data_root, args.split, 'jawi')
    ds = LmdbDataset(
        root=lmdb_path,
        charset=CHARSET,
        max_label_len=hp.max_label_length,
        remove_whitespace=False,
        normalize_unicode=False,
        transform=transform
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda b: (torch.stack([x[0] for x in b]), [x[1] for x in b])
    )

    # Debug: first 30 samples
    print("\n--- First 30 GT vs PR ---")
    seen = 0
    for imgs, labels in loader:
        imgs = imgs.to(device)
        probs = model(imgs).softmax(-1)
        preds, _ = model.tokenizer.decode(probs)
        for gt, pr in zip(labels, preds):
            if seen < 30:
                print(f"{seen+1:02d}: GT='{gt}' | PR='{pr}'")
                seen += 1
            else:
                break
        if seen >= 30:
            break

    # Evaluation metrics
    total = correct = 0
    total_ned = total_conf = total_len = 0.0
    total_char_edits = total_chars = 0
    total_word_edits = total_words = 0

    for batch_idx, (imgs, labels) in enumerate(tqdm(loader, desc="Evaluating")):
        out = model.test_step((imgs.to(device), labels), batch_idx)['output']
        # existing metrics
        total      += out.num_samples
        correct    += out.correct
        total_ned  += out.ned
        total_conf += out.confidence
        total_len  += out.label_length
        # WER/CER per sample
        probs = model(imgs.to(device)).softmax(-1)
        preds, _ = model.tokenizer.decode(probs)
        for gt, pr in zip(labels, preds):
            # CER
            char_edits = edit_distance(gt, pr)
            total_char_edits += char_edits
            total_chars += len(gt)
            # WER (split on whitespace)
            gt_words = gt.split()
            pr_words = pr.split()
            word_edits = edit_distance(gt_words, pr_words)
            total_word_edits += word_edits
            total_words += len(gt_words)

    # Print final results
    acc = 100 * correct / total
    one_minus_ned = 100 * (1 - total_ned / total)
    mean_conf = 100 * (total_conf / total)
    avg_len = total_len / total
    cer = 100 * total_char_edits / total_chars if total_chars > 0 else 0.0
    wer = 100 * total_word_edits / total_words if total_words > 0 else 0.0

    print("\n=== Final Results ===")
    print(f"Accuracy       : {acc:.2f}%")
    print(f"1 - NED        : {one_minus_ned:.2f}%")
    print(f"Avg confidence : {mean_conf:.2f}%")
    print(f"Avg label len  : {avg_len:.2f}")
    print(f"CER            : {cer:.2f}%")
    print(f"WER            : {wer:.2f}%")

if __name__ == '__main__':
    main()
