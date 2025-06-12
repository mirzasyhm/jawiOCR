# jawiocr_crnn.py

import os
import sys
import argparse
import time
import cv2
import json
import math
import numpy as np
from collections import OrderedDict

import torch
import torch.backends.cudnn as cudnn
from PIL import Image as PILImage
from torchvision import transforms as TorchTransforms

# TensorFlow imports for optional orientation model
try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing import image as tf_keras_image
    from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_preprocess_input
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("Warning: TensorFlow not found. Custom orientation model functionality will be disabled.")


# --- Path Setup ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))
craft_module_dir = os.path.join(current_script_dir, 'craft') 


if craft_module_dir not in sys.path: sys.path.insert(0, craft_module_dir)

# --- CRAFT Model Import and Utilities ---
# (Copied from paste.txt [1] - This part remains largely the same)
try:
    from model.craft import CRAFT # From the 'craft' submodule
except ImportError as e:
    print(f"Error importing 'CRAFT' from 'model.craft': {e}\nEnsure 'craft' submodule is in the Python path (e.g., in same dir or added to sys.path).")
    sys.exit(1)

try:
    from crnn.model import CRNN # Your CRNN model class
except ImportError:
    print("Error: Could not import CRNN from model.py. Ensure model.py is in the Python path.")
    sys.exit(1)


# --- Utility and Model Functions (largely unchanged) ---

def copyStateDict(state_dict):
    """Handles model state dictionaries saved with 'module.' prefix."""
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v
    return new_state_dict

def normalizeMeanVariance(in_img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
    """Normalizes an image for CRAFT model input."""
    img = in_img.copy().astype(np.float32)
    img -= np.array([mean[0] * 255.0, mean[1] * 255.0, mean[2] * 255.0], dtype=np.float32)
    img /= np.array([variance[0] * 255.0, variance[1] * 255.0, variance[2] * 255.0], dtype=np.float32)
    return img

def resize_aspect_ratio(img, square_size, interpolation, mag_ratio=1.):
    """Resizes an image while maintaining aspect ratio for CRAFT."""
    height, width, channel = img.shape
    target_size = mag_ratio * max(height, width)
    if target_size > square_size: target_size = square_size
    ratio = target_size / max(height, width)
    target_h, target_w = int(height * ratio), int(width * ratio)
    proc = cv2.resize(img, (target_w, target_h), interpolation=interpolation)
    target_h32, target_w32 = target_h, target_w
    if target_h % 32 != 0: target_h32 = target_h + (32 - target_h % 32)
    if target_w % 32 != 0: target_w32 = target_w + (32 - target_w % 32)
    resized = np.zeros((target_h32, target_w32, channel), dtype=np.uint8)
    resized[0:target_h, 0:target_w, :] = proc
    return resized, ratio, (int(target_w32/2), int(target_h32/2))

# Using the simpler getDetBoxes from previous versions for clarity
def getDetBoxes(textmap, linkmap, text_threshold, link_threshold, low_text):
    linkmap_vis, textmap_vis = linkmap.copy(), textmap.copy()
    img_h, img_w = textmap.shape
    ret, text_score = cv2.threshold(textmap_vis, low_text, 1, 0)
    ret, link_score = cv2.threshold(linkmap_vis, link_threshold, 1, 0)
    text_score_comb = np.clip(text_score + link_score, 0, 1)
    nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(text_score_comb.astype(np.uint8), connectivity=4)
    det = []
    for k in range(1, nLabels):
        size = stats[k, cv2.CC_STAT_AREA]
        if size < 10: continue
        if np.max(textmap_vis[labels == k]) < text_threshold: continue
        segmap = np.zeros(textmap.shape, dtype=np.uint8)
        segmap[labels == k] = 255
        segmap[np.logical_and(link_score == 1, text_score == 0)] = 0
        x, y = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP]
        w, h = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
        niter = int(math.sqrt(size * min(w, h) / (w * h))) * 2 if w > 0 and h > 0 else 0
        sx, ex, sy, ey = x - niter, x + w + niter + 1, y - niter, y + h + niter + 1
        sx, ex, sy, ey = max(0, sx), min(img_w, ex), max(0, sy), min(img_h, ey)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1 + niter, 1 + niter))
        segmap[sy:ey, sx:ex] = cv2.dilate(segmap[sy:ey, sx:ex], kernel)
        np_contours = np.roll(np.array(np.where(segmap != 0)), 1, axis=0).transpose().reshape(-1, 2)
        if np_contours.size == 0: continue
        rectangle = cv2.minAreaRect(np_contours)
        box = cv2.boxPoints(rectangle)
        startidx = box.sum(axis=1).argmin()
        box = np.roll(box, 4 - startidx, 0)
        det.append(box)
    return det

def perform_craft_inference(net, image_bgr, text_threshold, link_threshold, low_text, cuda, canvas_size=1280, mag_ratio=1.5):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img_resized, target_ratio, _ = resize_aspect_ratio(image_rgb, canvas_size, cv2.INTER_LINEAR, mag_ratio)
    x = normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)
    if cuda: x = x.cuda()
    with torch.no_grad():
        y, _ = net(x)
    score_text = y[0, :, :, 0].cpu().data.numpy()
    score_link = y[0, :, :, 1].cpu().data.numpy()
    boxes = getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text)
    final_polys = []
    for box in boxes:
        poly = np.array(box).astype(np.int32)
        scaled_poly = (poly * 2 / target_ratio)
        final_polys.append(scaled_poly)
    return final_polys


# --- CRNN Model Setup ---
CRNN_IMG_HEIGHT = 32
CRNN_IMG_WIDTH = 128
CRNN_NUM_CHANNELS = 1

def load_crnn_model(model_path, alphabet_path, device):
    print(f"Loading CRNN model from: {model_path}")
    if not os.path.exists(alphabet_path):
        print(f"CRITICAL Error: Alphabet file '{alphabet_path}' not found.")
        sys.exit(1)
    with open(alphabet_path, 'r', encoding='utf-8') as f:
        alphabet_chars = json.load(f)
    
    n_class = len(alphabet_chars) + 1
    model = CRNN(imgH=CRNN_IMG_HEIGHT, nc=CRNN_NUM_CHANNELS, nclass=n_class, nh=256)
    
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"CRITICAL Error loading CRNN state_dict: {e}")
        sys.exit(1)

    model = model.to(device).eval()
    transform = TorchTransforms.Compose([
        TorchTransforms.ToPILImage(),
        TorchTransforms.Resize((CRNN_IMG_HEIGHT, CRNN_IMG_WIDTH)),
        TorchTransforms.Grayscale(num_output_channels=CRNN_NUM_CHANNELS),
        TorchTransforms.ToTensor(),
        TorchTransforms.Normalize(mean=[0.5], std=[0.5])
    ])
    print("CRNN model, alphabet, and transform loaded successfully.")
    return model, transform, alphabet_chars

def preprocess_for_crnn(img_crop_bgr, crnn_transform, device):
    if img_crop_bgr is None or img_crop_bgr.size == 0: return None
    img_tensor = crnn_transform(img_crop_bgr).unsqueeze(0)
    return img_tensor.to(device)

def decode_crnn_output(log_probs, alphabet, beam_size=20):
    try:
        from torchaudio.models.decoder import ctc_decoder
    except ImportError:
        print("Error: torchaudio is required for decoding. Please install it (`pip install torchaudio`).")
        return "DECODER_ERROR", 0.0

    log_probs_for_decoder = log_probs.permute(1, 0, 2)
    blank_token = "-"
    decoder_alphabet = [char for char in alphabet if char != blank_token]
    
    decoder = ctc_decoder(
        lexicon=None, tokens=[blank_token] + decoder_alphabet,
        beam_size=beam_size, blank_token=blank_token, sil_token=blank_token,
        nbest=1, log_add=True
    )
    hypotheses = decoder(log_probs_for_decoder.cpu())
    if not hypotheses or not hypotheses[0]: return "", 0.0
    
    best_hypothesis = hypotheses[0][0]
    confidence = math.exp(best_hypothesis.score)
    text = "".join(decoder.idxs_to_tokens(best_hypothesis.tokens))
    return text, confidence


# --- Image Utilities ---
def rotate_image_cv(image_cv, angle_degrees):
    if angle_degrees == 90: return cv2.rotate(image_cv, cv2.ROTATE_90_CLOCKWISE)
    if angle_degrees == 180: return cv2.rotate(image_cv, cv2.ROTATE_180)
    if angle_degrees == 270: return cv2.rotate(image_cv, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image_cv

def get_cropped_image_from_poly(image_bgr, poly_pts):
    poly = np.asarray(poly_pts, dtype=np.float32)
    target_w = int(round(max(np.linalg.norm(poly[1] - poly[0]), np.linalg.norm(poly[2] - poly[3]))))
    target_h = int(round(max(np.linalg.norm(poly[3] - poly[0]), np.linalg.norm(poly[2] - poly[1]))))
    if target_w <= 0 or target_h <= 0: return None
    dst_pts = np.array([[0, 0], [target_w-1, 0], [target_w-1, target_h-1], [0, target_h-1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(poly, dst_pts)
    warped_crop = cv2.warpPerspective(image_bgr, M, (target_w, target_h))
    return warped_crop

# --- Main Logic ---
def main(args):
    device = torch.device('cuda' if not args.no_cuda and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load models
    craft_net = CRAFT()
    craft_net.load_state_dict(copyStateDict(torch.load(args.craft_model_path, map_location=device)))
    craft_net.to(device).eval()

    crnn_model, crnn_transform, crnn_alphabet = load_crnn_model(
        args.crnn_model_path, args.alphabet_path, device
    )

    # Process image
    print(f"Processing image: {args.image_path}")
    image_bgr = cv2.imread(args.image_path)
    if image_bgr is None:
        print(f"Error: Could not read image {args.image_path}"); return

    # Detection
    detected_polys = perform_craft_inference(
        craft_net, image_bgr, args.text_threshold, args.link_threshold,
        args.low_text, (device.type == 'cuda')
    )
    print(f"CRAFT detected {len(detected_polys)} regions.")

    # Sort regions right-to-left
    regions = sorted(detected_polys, key=lambda p: int(np.mean(p[:, 0])), reverse=True)

    # Recognition
    final_text_snippets = []
    for poly in regions:
        cropped_bgr = get_cropped_image_from_poly(image_bgr, poly)
        if cropped_bgr is None: continue
        
        # Simple height > width check for vertical text
        h, w = cropped_bgr.shape[:2]
        if h > w:
            cropped_bgr = cv2.rotate(cropped_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)

        crnn_input = preprocess_for_crnn(cropped_bgr, crnn_transform, device)
        if crnn_input is None: continue

        with torch.no_grad():
            raw_preds = crnn_model(crnn_input)
            log_probs = raw_preds.log_softmax(2)
            text, conf = decode_crnn_output(log_probs, crnn_alphabet, args.beam_size)
            
            if text:
                print(f"  - Detected: '{text}' (Confidence: {conf:.2f})")
                final_text_snippets.append(text)

    # Final Output
    final_text = " ".join(final_text_snippets)
    print("\n" + "="*50)
    print(f"Final Combined Text (Right-to-Left):")
    print(final_text)
    print("="*50 + "\n")

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        base_fname = os.path.splitext(os.path.basename(args.image_path))[0]
        txt_filepath = os.path.join(args.output_dir, f"res_{base_fname}.txt")
        with open(txt_filepath, 'w', encoding='utf-8') as f:
            f.write(final_text)
        print(f"Result saved to: {txt_filepath}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Jawi OCR with CRAFT and CRNN')
    
    parser.add_argument('--image_path', required=True, type=str, help="Path to the input image file.")
    parser.add_argument('--craft_model_path', required=True, type=str, help="Path to CRAFT model (.pth).")
    parser.add_argument('--crnn_model_path', required=True, type=str, help="Path to pre-trained CRNN model (.pth).")
    parser.add_argument('--alphabet_path', required=True, type=str, help="Path to the alphabet.json for CRNN.")
    parser.add_argument('--output_dir', default=None, type=str, help="Directory to save output text files.")
    
    # Model Parameters
    parser.add_argument('--text_threshold', default=0.7, type=float, help="Text confidence threshold.")
    parser.add_argument('--low_text', default=0.4, type=float, help="Text low-bound score.")
    parser.add_argument('--link_threshold', default=0.4, type=float, help="Link confidence threshold.")
    parser.add_argument('--beam_size', type=int, default=20, help="Beam size for CTC decoder.")
    parser.add_argument('--no_cuda', action='store_true', help="Disable CUDA.")
    
    args = parser.parse_args()
    main(args)
