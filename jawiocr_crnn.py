#!/usr/bin/env python3
import os
import sys
import argparse
import cv2
import numpy as np
import torch
from collections import OrderedDict
from PIL import Image as PILImage
from torchvision import transforms as TorchTransforms
import json

# --- CRAFT Utilities (from jawiocr.py) ---
def copyStateDict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v
    return new_state_dict

def normalizeMeanVariance(in_img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
    img = in_img.copy().astype(np.float32)
    img -= np.array([mean[0]*255.0, mean[1]*255.0, mean[2]*255.0], dtype=np.float32)
    img /= np.array([variance[0]*255.0, variance[1]*255.0, variance[2]*255.0], dtype=np.float32)
    return img

def resize_aspect_ratio(img, square_size, interpolation, mag_ratio=1.0):
    height, width, _ = img.shape
    target_size = mag_ratio * max(height, width)
    if target_size > square_size:
        target_size = square_size
    ratio = target_size / max(height, width)
    target_h, target_w = int(height * ratio), int(width * ratio)
    proc = cv2.resize(img, (target_w, target_h), interpolation=interpolation)
    target_h32 = target_h + (32 - target_h % 32) if target_h % 32 != 0 else target_h
    target_w32 = target_w + (32 - target_w % 32) if target_w % 32 != 0 else target_w
    resized = np.zeros((target_h32, target_w32, 3), dtype=np.uint8)
    resized[0:target_h, 0:target_w, :] = proc
    return resized, ratio

def getDetBoxes_core(textmap, linkmap, text_threshold, link_threshold, low_text):
    linkmap, textmap = linkmap.copy(), textmap.copy()
    _, text_score = cv2.threshold(textmap, low_text, 1, 0)
    _, link_score = cv2.threshold(linkmap, link_threshold, 1, 0)
    text_score_comb = np.clip(text_score + link_score, 0, 1)
    nLabels, labels, stats, _ = cv2.connectedComponentsWithStats(
        text_score_comb.astype(np.uint8), connectivity=4)
    boxes = []
    for k in range(1, nLabels):
        size = stats[k, cv2.CC_STAT_AREA]
        if size < 10 or np.max(textmap[labels==k]) < text_threshold:
            continue
        segmap = np.zeros(textmap.shape, dtype=np.uint8)
        segmap[labels==k] = 255
        segmap[np.logical_and(link_score==1, text_score==0)] = 0
        x, y, w, h = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP], stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
        niter = int(np.sqrt(size * min(w,h) / (w*h)) * 2) if w*h>0 else 0
        sx, ex = max(0, x-niter), min(labels.shape[1], x+w+niter+1)
        sy, ey = max(0, y-niter), min(labels.shape[0], y+h+niter+1)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1+niter,1+niter))
        segmap[sy:ey, sx:ex] = cv2.dilate(segmap[sy:ey, sx:ex], kernel)
        contours = np.roll(np.array(np.where(segmap!=0)),1,axis=0).transpose().reshape(-1,2)
        if contours.size == 0:
            continue
        rect = cv2.minAreaRect(contours)
        box = cv2.boxPoints(rect)
        boxes.append(box)
    return boxes

def getDetBoxes(textmap, linkmap, text_threshold, link_threshold, low_text, canvas_size=1280, mag_ratio=1.5):
    boxes = getDetBoxes_core(textmap, linkmap, text_threshold, link_threshold, low_text)
    return boxes

def perform_craft_inference(net, image_bgr, text_threshold, link_threshold, low_text, cuda, canvas_size=1280, mag_ratio=1.5):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img_resized, ratio = resize_aspect_ratio(image_rgb, canvas_size, cv2.INTER_LINEAR, mag_ratio)
    x = normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2,0,1).unsqueeze(0)
    if cuda:
        x = x.cuda()
    with torch.no_grad():
        y,_ = net(x)
    score_text = y[0,:,:,0].cpu().numpy()
    score_link = y[0,:,:,1].cpu().numpy()
    boxes = getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text)
    # scale boxes back to original image
    polys = []
    for box in boxes:
        box = np.array(box) * (1/ratio)
        polys.append(box.astype(np.int32))
    return polys

# --- CRNN Utilities (from test_crnn.py) ---
CRNN_IMG_HEIGHT = 32
CRNN_IMG_WIDTH = 128
CRNN_NUM_CHANNELS = 1

def load_crnn_model_local(model_path, alphabet_path, device):
    # Load alphabet
    with open(alphabet_path, 'r', encoding='utf-8') as f:
        alphabet_chars = json.load(f)
    n_class = len(alphabet_chars) + 1
    from crnn.model import CRNN
    crnn_model = CRNN(imgH=CRNN_IMG_HEIGHT, nc=CRNN_NUM_CHANNELS, nclass=n_class, nh=256)
    state_dict = torch.load(model_path, map_location=device)
    crnn_model.load_state_dict(state_dict)
    crnn_model = crnn_model.to(device).eval()
    crnn_transform = TorchTransforms.Compose([
        TorchTransforms.ToPILImage(),
        TorchTransforms.Resize((CRNN_IMG_HEIGHT, CRNN_IMG_WIDTH)),
        TorchTransforms.Grayscale(num_output_channels=1),
        TorchTransforms.ToTensor(),
        TorchTransforms.Normalize(mean=[0.5], std=[0.5])
    ])
    return crnn_model, crnn_transform, alphabet_chars

def preprocess_for_crnn_local(img_crop_bgr, crnn_transform, device):
    img_tensor = crnn_transform(img_crop_bgr).unsqueeze(0)
    return img_tensor.to(device)

def run_demo(args):
    cuda = (not args.no_cuda and torch.cuda.is_available())
    device = torch.device('cuda' if cuda else 'cpu')
    # Load models
    craft_net = CRAFT()
    ckpt = torch.load(args.craft_model_path, map_location=device)
    if 'state_dict' in ckpt:
        craft_sd = ckpt['state_dict']
    else:
        craft_sd = ckpt
    craft_net.load_state_dict(copyStateDict(craft_sd))
    craft_net = craft_net.to(device).eval()

    crnn_model, crnn_transform, alphabet = load_crnn_model_local(
        args.crnn_model_path, args.alphabet_path, device)

    # Read image
    img = cv2.imread(args.image_path)
    if img is None:
        print(f"Error: Cannot read image {args.image_path}")
        sys.exit(1)

    # Detect text regions
    polys = perform_craft_inference(
        craft_net, img, args.text_threshold, args.link_threshold,
        args.low_text, cuda, args.canvas_size, args.mag_ratio)
    # Sort right-to-left
    regions = []
    for poly in polys:
        cx = int(np.mean(poly[:,0]))
        regions.append((cx, poly))
    regions = sorted(regions, key=lambda x: x[0], reverse=True)

    results = []
    for i, (cx, poly) in enumerate(regions):
        # crop
        rect = cv2.boundingRect(poly)
        x,y,w,h = rect
        crop = img[y:y+h, x:x+w]
        if crop.size==0:
            continue
        inp = preprocess_for_crnn_local(crop, crnn_transform, device)
        with torch.no_grad():
            preds = crnn_model(inp)
            preds = preds.log_softmax(2)
            # greedy decode
            _, max_inds = preds.cpu().max(2)
            seq = max_inds.squeeze(1).numpy().tolist()
            # collapse repeats & blanks
            text = ''
            prev = None
            for idx in seq:
                if idx!=0 and idx!=prev:
                    text += alphabet[idx-1]
                prev = idx
        print(f"Region {i+1}: {text}")
        results.append(text)

    final_text = ' '.join(results)
    print(f"\nFinal OCR Text: {final_text}")

    if args.save_viz:
        os.makedirs(args.output_dir, exist_ok=True)
        vis = img.copy()
        for poly in polys:
            cv2.polylines(vis, [poly], True, (0,0,255), 2)
        out_img = os.path.join(args.output_dir, 'ocr_result.jpg')
        cv2.imwrite(out_img, vis)
        print(f"Visualization saved to {out_img}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Jawi OCR Demo: CRAFT + CRNN')
    parser.add_argument('--image_path', required=True, type=str)
    parser.add_argument('--craft_model_path', required=True, type=str)
    parser.add_argument('--crnn_model_path', required=True, type=str)
    parser.add_argument('--alphabet_path', required=True, type=str)
    parser.add_argument('--output_dir', default='./crnn_ocr_demo_results/', type=str)
    parser.add_argument('--text_threshold', default=0.7, type=float)
    parser.add_argument('--link_threshold', default=0.4, type=float)
    parser.add_argument('--low_text', default=0.4, type=float)
    parser.add_argument('--canvas_size', default=1280, type=int)
    parser.add_argument('--mag_ratio', default=1.5, type=float)
    parser.add_argument('--poly', action='store_true')
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--save_viz', action='store_true')
    args = parser.parse_args()
    run_demo(args)
