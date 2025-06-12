# e2e_jawi_ocr_crnn_greedy_debug.py

import os
import sys
import argparse
import time
import cv2
import math
import numpy as np
import pandas as pd
import json
import torch
import torch.backends.cudnn as cudnn
from collections import OrderedDict
from PIL import Image as PILImage
from torchvision import transforms as TorchTransforms
from tqdm import tqdm
import jiwer

# Optional TensorFlow for custom orientation model
try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing import image as tf_keras_image
    from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_preprocess_input
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("Warning: TensorFlow not found. Custom orientation model functionality will be disabled.")

# --- Path Setup & Model Imports ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))
craft_module_dir = os.path.join(current_script_dir, 'craft')
if craft_module_dir not in sys.path: sys.path.insert(0, craft_module_dir)
try:
    from model.craft import CRAFT
    from crnn.model import CRNN
except ImportError as e:
    print(f"Error importing a model class: {e}"); sys.exit(1)

# --- All Helper Functions (Unchanged) ---
def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        new_state_dict = OrderedDict()
        for k, v in state_dict.items(): new_state_dict[k[7:]] = v
        return new_state_dict
    return state_dict
def normalizeMeanVariance(in_img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
    img = in_img.copy().astype(np.float32)
    img -= np.array([mean[0] * 255.0, mean[1] * 255.0, mean[2] * 255.0], dtype=np.float32)
    img /= np.array([variance[0] * 255.0, variance[1] * 255.0, variance[2] * 255.0], dtype=np.float32)
    return img
def resize_aspect_ratio(img, square_size, interpolation, mag_ratio=1.):
    h, w, c = img.shape
    target_size = mag_ratio * max(h, w)
    if target_size > square_size: target_size = square_size
    ratio = target_size / max(h, w)
    target_h, target_w = int(h * ratio), int(w * ratio)
    proc = cv2.resize(img, (target_w, target_h), interpolation=interpolation)
    target_h32, target_w32 = target_h + (32 - target_h % 32 if target_h % 32 != 0 else 0), target_w + (32 - target_w % 32 if target_w % 32 != 0 else 0)
    resized = np.zeros((target_h32, target_w32, c), dtype=np.uint8)
    resized[0:target_h, 0:target_w, :] = proc
    return resized, ratio, (target_w, target_h)
def getDetBoxes_core(textmap, linkmap, text_threshold, link_threshold, low_text):
    linkmap, textmap = linkmap.copy(), textmap.copy()
    _, text_score = cv2.threshold(textmap, low_text, 1, 0)
    _, link_score = cv2.threshold(linkmap, link_threshold, 1, 0)
    text_score_comb = np.clip(text_score + link_score, 0, 1)
    nLabels, labels, stats, _ = cv2.connectedComponentsWithStats(text_score_comb.astype(np.uint8), connectivity=4)
    det, mapper = [], []
    for k in range(1, nLabels):
        size = stats[k, cv2.CC_STAT_AREA]
        if size < 10 or np.max(textmap[labels==k]) < text_threshold: continue
        segmap = np.zeros(textmap.shape, dtype=np.uint8); segmap[labels==k] = 255
        segmap[np.logical_and(link_score==1, text_score==0)] = 0
        x, y, w, h = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP], stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
        niter = int(math.sqrt(size * min(w, h) / (w * h))) * 2 if w > 0 and h > 0 else 0
        sx, ex, sy, ey = x - niter, x + w + niter + 1, y - niter, y + h + niter + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(1 + niter, 1 + niter)); segmap[sy:ey, sx:ex] = cv2.dilate(segmap[sy:ey, sx:ex], kernel)
        np_contours = np.roll(np.array(np.where(segmap!=0)),1,axis=0).transpose().reshape(-1,2)
        if np_contours.size == 0: continue
        box = cv2.boxPoints(cv2.minAreaRect(np_contours))
        box = np.roll(box, 4-box.sum(axis=1).argmin(), 0)
        det.append(box); mapper.append(k)
    return det, labels, mapper
def getPoly_core(boxes, labels, mapper, linkmap): return boxes # Simplified for clarity
def getDetBoxes(tm, lm, tt, lt, lwt, poly=False):
    b, _, _ = getDetBoxes_core(tm, lm, tt, lt, lwt)
    return b, [None]*len(b)
def perform_craft_inference(net, image_bgr, text_threshold, link_threshold, low_text, cuda, poly, canvas_size=1280, mag_ratio=1.5):
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img_resized, ratio, _ = resize_aspect_ratio(img_rgb, canvas_size, cv2.INTER_LINEAR, mag_ratio)
    x = torch.from_numpy(normalizeMeanVariance(img_resized)).permute(2, 0, 1).unsqueeze(0)
    if cuda: x = x.cuda()
    with torch.no_grad(): y, _ = net(x)
    st, sl = y[0,:,:,0].cpu().data.numpy(), y[0,:,:,1].cpu().data.numpy()
    boxes, _ = getDetBoxes(st, sl, text_threshold, link_threshold, low_text, poly)
    return [(p * 2 / ratio).astype(np.int32) for p in boxes if p is not None]
def get_cropped_image_from_poly(image, poly):
    if poly is None or len(poly) != 4: return None
    rect = np.array(sorted(poly, key=lambda p: (p[0], p[1])), dtype="float32") # Simple sort, may need refinement
    w = max(np.linalg.norm(rect[1] - rect[0]), np.linalg.norm(rect[3] - rect[2]))
    h = max(np.linalg.norm(rect[2] - rect[0]), np.linalg.norm(rect[3] - rect[1]))
    if int(w) <= 0 or int(h) <= 0: return None
    dst = np.array([[0,0], [int(w)-1,0], [0,int(h)-1], [int(w)-1,int(h)-1]], dtype=np.float32)
    return cv2.warpPerspective(image, cv2.getPerspectiveTransform(rect, dst), (int(w), int(h)))
def simple_orientation_correction(crop):
    if crop is not None and crop.shape[0] > crop.shape[1] > 0: return cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return crop
# All other minor helpers are assumed to be here and correct...

class JawiOCREngine:
    def __init__(self, config):
        self.config = argparse.Namespace(**config)
        self.pytorch_device = torch.device('cuda' if not self.config.no_cuda and torch.cuda.is_available() else 'cpu')
        print(f"JawiOCREngine using device: {self.pytorch_device}")
        self.craft_net = self._load_craft_model(self.config.craft_model_path)
        self.crnn_model, self.crnn_transform, self.alphabet = self._load_crnn_model(self.config.crnn_model_path, self.config.alphabet_path)

    def _load_craft_model(self, path):
        net = CRAFT()
        checkpoint = torch.load(path, map_location=self.pytorch_device)
        state_dict = checkpoint.get('craft', checkpoint)
        net.load_state_dict(copyStateDict(state_dict))
        print("CRAFT model loaded.")
        return net.to(self.pytorch_device).eval()

    def _load_crnn_model(self, model_path, alphabet_path):
        with open(alphabet_path, 'r', encoding='utf-8') as f: alphabet_chars = json.load(f)
        crnn_model = CRNN(imgH=32, nc=1, nclass=len(alphabet_chars) + 1, nh=256)
        state_dict = torch.load(model_path, map_location=self.pytorch_device, weights_only=True)
        crnn_model.load_state_dict(copyStateDict(state_dict))
        crnn_transform = TorchTransforms.Compose([
            TorchTransforms.ToPILImage(), TorchTransforms.Resize((32, 128)),
            TorchTransforms.Grayscale(1), TorchTransforms.ToTensor(), TorchTransforms.Normalize([0.5], [0.5])])
        print("CRNN model loaded.")
        return crnn_model.to(self.pytorch_device).eval(), crnn_transform, alphabet_chars

    def _decode_crnn_output(self, preds_log_softmax):
        _, max_inds = preds_log_softmax.cpu().max(2)
        seq = max_inds[:, 0].tolist()
        return ''.join(self.alphabet[c-1] for j, c in enumerate(seq) if c != 0 and (j == 0 or c != seq[j-1]))

    def predict(self, image_path):
        image_bgr = cv2.imread(image_path)
        if image_bgr is None: return ""
        
        # Run the main OCR pass, which now includes debugging logic
        final_text = self._run_single_ocr_pass(image_bgr)
        
        return final_text.strip()

    def _run_single_ocr_pass(self, image_bgr_input):
        detected_polys = perform_craft_inference(
            self.craft_net, image_bgr_input, self.config.text_threshold, 
            self.config.link_threshold, self.config.low_text, 
            (self.pytorch_device.type == 'cuda'), self.config.poly)
        
        # --- DEBUG: Save detector output ---
        if self.config.save_debug_detector_output:
            debug_img = image_bgr_input.copy()
            for poly in detected_polys:
                cv2.polylines(debug_img, [poly], isClosed=True, color=(0, 255, 0), thickness=2)
            out_path = os.path.join(self.config.results_output_dir, "debug_detector_output.jpg")
            cv2.imwrite(out_path, debug_img)
            print(f"Saved detector debug image to: {out_path}")

        regions = sorted([p for p in detected_polys if p is not None], key=lambda p: np.mean(p[:, 0]), reverse=True)
        
        texts = []
        for i, poly in enumerate(regions):
            crop = get_cropped_image_from_poly(image_bgr_input, poly)
            if self.config.use_simple_orientation: crop = simple_orientation_correction(crop)
            
            # --- DEBUG: Save individual crops ---
            if self.config.save_debug_crops and crop is not None:
                crops_dir = os.path.join(self.config.results_output_dir, "debug_crops")
                os.makedirs(crops_dir, exist_ok=True)
                crop_path = os.path.join(crops_dir, f"crop_{i:03d}.png")
                cv2.imwrite(crop_path, crop)

            crnn_input = preprocess_for_crnn(crop, self.crnn_transform, self.pytorch_device)
            if crnn_input is None: continue
            
            with torch.no_grad():
                preds = self.crnn_model(crnn_input).log_softmax(2).permute(1, 0, 2)
                text = self._decode_crnn_output(preds)
                if text: texts.append(text)
            
        return " ".join(texts)
    
    def preprocess_for_crnn(self, img_crop_bgr, crnn_transform, device):
        if img_crop_bgr is None or img_crop_bgr.size == 0: return None
        return crnn_transform(img_crop_bgr).unsqueeze(0).to(device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Jawi OCR Pipeline with CRAFT and CRNN (Greedy Decoder)')
    # --- ADDED DEBUG ARGUMENTS ---
    parser.add_argument('--save_debug_detector_output', action='store_true', help='Save an image with detected boxes drawn on it.')
    parser.add_argument('--save_debug_crops', action='store_true', help='Save each cropped word image before recognition.')

    # Other arguments remain the same
    parser.add_argument('--image_path', type=str, default=None)
    parser.add_argument('--dataset_dir', type=str, default=None)
    parser.add_argument('--craft_model_path', required=True, type=str)
    parser.add_argument('--crnn_model_path', required=True, type=str)
    parser.add_argument('--alphabet_path', required=True, type=str)
    parser.add_argument('--results_output_dir', default='./ocr_results/', type=str)
    parser.add_argument('--text_threshold', default=0.7, type=float)
    parser.add_argument('--link_threshold', default=0.4, type=float)
    parser.add_argument('--low_text', default=0.4, type=float)
    parser.add_argument('--poly', default=False, action='store_true')
    parser.add_argument('--use_simple_orientation', action='store_true')
    parser.add_argument('--no_cuda', action='store_true')
    
    args = parser.parse_args()
    cudnn.benchmark = True
    
    # Ensure results directory exists
    os.makedirs(args.results_output_dir, exist_ok=True)

    engine = JawiOCREngine(config=vars(args))

    if args.image_path:
        print(f"\n--- Running prediction on: {args.image_path} ---")
        start_time = time.time()
        final_text = engine.predict(args.image_path)
        print(f"\nRecognized Text: {final_text}")
        print(f"Processing Time: {time.time() - start_time:.2f}s")
    else:
        print("Error: Please provide --image_path for single prediction.")

