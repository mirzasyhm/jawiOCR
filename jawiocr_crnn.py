# e2e_jawi_ocr_crnn_greedy_fixed.py

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

# --- Path Setup ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))
craft_module_dir = os.path.join(current_script_dir, 'craft')
if craft_module_dir not in sys.path:
    sys.path.insert(0, craft_module_dir)

# --- CRAFT Model Import and Utilities ---
try:
    from model.craft import CRAFT
except ImportError as e:
    print(f"Error: Could not import 'CRAFT' from 'model.craft': {e}\n"
          f"Please ensure the 'craft' submodule is in the same directory or Python path.")
    sys.exit(1)

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        return new_state_dict
    return state_dict

# ... [All other utility functions like normalizeMeanVariance, getDetBoxes, etc. remain unchanged] ...
def normalizeMeanVariance(in_img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
    img = in_img.copy().astype(np.float32)
    img -= np.array([mean[0] * 255.0, mean[1] * 255.0, mean[2] * 255.0], dtype=np.float32)
    img /= np.array([variance[0] * 255.0, variance[1] * 255.0, variance[2] * 255.0], dtype=np.float32)
    return img

def resize_aspect_ratio(img, square_size, interpolation, mag_ratio=1.):
    height, width, channel = img.shape
    target_size = mag_ratio * max(height, width)
    if target_size > square_size:
        target_size = square_size
    ratio = target_size / max(height, width)    
    target_h, target_w = int(height * ratio), int(width * ratio)
    proc = cv2.resize(img, (target_w, target_h), interpolation=interpolation)
    target_h32, target_w32 = target_h, target_w
    if target_h % 32 != 0: target_h32 = target_h + (32 - target_h % 32)
    if target_w % 32 != 0: target_w32 = target_w + (32 - target_w % 32)
    resized = np.zeros((target_h32, target_w32, channel), dtype=np.uint8)
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
        segmap = np.zeros(textmap.shape, dtype=np.uint8)
        segmap[labels==k] = 255
        segmap[np.logical_and(link_score==1, text_score==0)] = 0
        x, y, w, h = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP], stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
        niter = int(math.sqrt(size * min(w, h) / (w * h))) * 2 if w > 0 and h > 0 else 0
        sx, ex, sy, ey = x - niter, x + w + niter + 1, y - niter, y + h + niter + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(1 + niter, 1 + niter))
        segmap[sy:ey, sx:ex] = cv2.dilate(segmap[sy:ey, sx:ex], kernel)
        np_contours = np.roll(np.array(np.where(segmap!=0)),1,axis=0).transpose().reshape(-1,2)
        if np_contours.size == 0: continue
        rectangle = cv2.minAreaRect(np_contours)
        box = cv2.boxPoints(rectangle)
        box = np.roll(box, 4-box.sum(axis=1).argmin(), 0)
        det.append(box); mapper.append(k)
    return det, labels, mapper

def getPoly_core(boxes, labels, mapper, linkmap):
    polys = []
    for k, box in enumerate(boxes):
        w, h = int(np.linalg.norm(box[0] - box[1])) + 1, int(np.linalg.norm(box[1] - box[2])) + 1
        if w < 10 or h < 10: polys.append(None); continue
        target = np.float32([[0,0],[w,0],[w,h],[0,h]])
        M = cv2.getPerspectiveTransform(box, target)
        word_label = cv2.warpPerspective(labels, M, (w, h), flags=cv2.INTER_NEAREST)
        try: np.linalg.inv(M)
        except np.linalg.LinAlgError: polys.append(None); continue
        word_label[word_label != mapper[k]] = 0
        polys.append(box)
    return polys

def getDetBoxes(textmap, linkmap, text_threshold, link_threshold, low_text, poly=False):
    boxes, labels, mapper = getDetBoxes_core(textmap, linkmap, text_threshold, link_threshold, low_text)
    return (boxes, getPoly_core(boxes, labels, mapper, linkmap)) if poly else (boxes, [None] * len(boxes))

def perform_craft_inference(net, image_bgr, text_threshold, link_threshold, low_text, cuda, poly, canvas_size=1280, mag_ratio=1.5):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img_resized, target_ratio, _ = resize_aspect_ratio(image_rgb, canvas_size, cv2.INTER_LINEAR, mag_ratio)
    x = normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)
    if cuda: x = x.cuda()
    with torch.no_grad(): y, _ = net(x)
    score_text, score_link = y[0,:,:,0].cpu().data.numpy(), y[0,:,:,1].cpu().data.numpy()
    boxes, polys = getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)
    results_to_scale = polys if poly and any(p is not None for p in polys) else boxes
    final_polys = []
    for p_map in results_to_scale:
        if p_map is not None:
            final_polys.append((p_map * 2 * (1 / target_ratio)).astype(np.int32))
    return final_polys

try:
    from crnn.model import CRNN
except ImportError:
    print("Error: Could not import CRNN from crnn/model.py."); sys.exit(1)
CRNN_IMG_HEIGHT, CRNN_IMG_WIDTH, CRNN_NUM_CHANNELS = 32, 128, 1

def load_crnn_model_local(model_path, alphabet_path, device):
    if not os.path.exists(alphabet_path): print(f"CRITICAL Error: Alphabet file '{alphabet_path}' not found."); sys.exit(1)
    with open(alphabet_path, 'r', encoding='utf-8') as f: alphabet_chars = json.load(f)
    n_class = len(alphabet_chars) + 1
    crnn_model = CRNN(imgH=CRNN_IMG_HEIGHT, nc=CRNN_NUM_CHANNELS, nclass=n_class, nh=256)
    try:
        # Using weights_only=True for security, assuming CRNN model is a pure state_dict
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        crnn_model.load_state_dict(copyStateDict(state_dict))
    except Exception as e: print(f"CRITICAL Error loading CRNN state_dict: {e}"); sys.exit(1)
    crnn_model.to(device).eval()
    crnn_transform = TorchTransforms.Compose([
        TorchTransforms.ToPILImage(), TorchTransforms.Resize((CRNN_IMG_HEIGHT, CRNN_IMG_WIDTH)),
        TorchTransforms.Grayscale(num_output_channels=1), TorchTransforms.ToTensor(),
        TorchTransforms.Normalize(mean=[0.5], std=[0.5])])
    print("CRNN model, alphabet, and transform loaded successfully.")
    return crnn_model, crnn_transform, alphabet_chars

# ... [Other utility functions like preprocess_for_crnn, rotate_image_cv, etc. remain unchanged] ...
def preprocess_for_crnn(img_crop_bgr, crnn_transform, device):
    if img_crop_bgr is None or img_crop_bgr.size == 0: return None
    return crnn_transform(img_crop_bgr).unsqueeze(0).to(device)

def rotate_image_cv(img, angle): return cv2.rotate(img, {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}.get(angle)) if angle in [90, 180, 270] else img

def load_custom_orientation_model_keras(model_path):
    if not TF_AVAILABLE: return None
    try: return tf.keras.models.load_model(model_path)
    except Exception as e: print(f"Warning: Could not load Keras orientation model: {e}"); return None

def get_custom_page_orientation(model, img, names):
    if not TF_AVAILABLE or model is None: return "0_degrees", 1.0
    pil_img = PILImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).resize((224, 224), PILImage.Resampling.LANCZOS)
    img_array = tf_keras_image.img_to_array(pil_img)
    preds = model.predict(resnet50_preprocess_input(np.expand_dims(img_array, axis=0)), verbose=0)
    return names[np.argmax(preds[0])], np.max(preds[0])

def simple_orientation_correction(crop):
    if crop is not None and crop.shape[0] > crop.shape[1] > 0: return cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return crop

def get_cropped_image_from_poly(image, poly):
    if poly is None or len(poly) != 4: return None
    rect = np.zeros((4, 2), dtype="float32")
    s, d = poly.sum(axis=1), np.diff(poly, axis=1)
    rect[0], rect[2] = poly[np.argmin(s)], poly[np.argmax(s)]
    rect[1], rect[3] = poly[np.argmin(d)], poly[np.argmax(d)]
    w = max(np.linalg.norm(rect[1] - rect[0]), np.linalg.norm(rect[2] - rect[3]))
    h = max(np.linalg.norm(rect[3] - rect[0]), np.linalg.norm(rect[2] - rect[1]))
    if int(w) <= 0 or int(h) <= 0: return None
    dst = np.array([[0,0], [int(w)-1,0], [int(w)-1,int(h)-1], [0,int(h)-1]], dtype=np.float32)
    return cv2.warpPerspective(image, cv2.getPerspectiveTransform(rect, dst), (int(w), int(h)))

class JawiOCREngine:
    def __init__(self, config):
        self.config = argparse.Namespace(**config)
        self.pytorch_device = torch.device('cuda' if not self.config.no_cuda and torch.cuda.is_available() else 'cpu')
        print(f"JawiOCREngine using device: {self.pytorch_device}")

        self.craft_net = self._load_craft_model(self.config.craft_model_path)
        self.crnn_model, self.crnn_transform, self.alphabet = self._load_crnn_model(self.config.crnn_model_path, self.config.alphabet_path)
        self.orientation_model = self._load_orientation_model()

    def _load_craft_model(self, path):
        """
        FIXED: Loads a CRAFT model from a file, handling both raw state_dicts
        and full training checkpoints.
        """
        net = CRAFT()
        print(f"Loading CRAFT model from: {path}")

        # Load the entire file. It could be a checkpoint or a state_dict.
        # We don't use weights_only=True here because it IS a checkpoint file with non-tensor data.
        checkpoint = torch.load(path, map_location=self.pytorch_device)

        # Check if the file is a training checkpoint by looking for the 'craft' key.
        if 'craft' in checkpoint.keys():
            print("Checkpoint file detected. Extracting model weights from 'craft' key.")
            model_state_dict = checkpoint['craft']
        else:
            # Assume the file is a regular state_dict.
            print("Assuming file is a raw state_dict.")
            model_state_dict = checkpoint
        
        # Apply the 'module.' prefix fix and load the weights.
        net.load_state_dict(copyStateDict(model_state_dict))
        print("CRAFT model loaded successfully.")
        return net.to(self.pytorch_device).eval()

    def _load_crnn_model(self, model_path, alphabet_path):
        return load_crnn_model_local(model_path, alphabet_path, self.pytorch_device)
        
    def _load_orientation_model(self):
        return load_custom_orientation_model_keras(self.config.custom_orientation_model_path) if self.config.custom_orientation_model_path else None

    def _decode_crnn_output(self, preds_log_softmax):
        _, max_inds = preds_log_softmax.cpu().max(2)
        seq = max_inds[:, 0].tolist()
        pred_text = ''.join(
            self.alphabet[c-1] for j, c in enumerate(seq)
            if c != 0 and (j == 0 or c != seq[j-1])
        )
        return pred_text, 1.0

    def predict(self, image_path):
        image_bgr = cv2.imread(image_path)
        if image_bgr is None: return ""
        image_to_process, _ = self._orient_page(image_bgr)
        final_text, _ = self._run_single_ocr_pass(image_to_process)
        return final_text.strip()

    def _orient_page(self, image_bgr):
        if not self.orientation_model: return image_bgr, 0
        class_names = self.config.orientation_class_names.split(',')
        pred_class, conf = get_custom_page_orientation(self.orientation_model, image_bgr, class_names)
        if conf * 100 < self.config.orientation_confidence_threshold: return image_bgr, 0
        angle = {"90_degrees": 270, "180_degrees": 180, "270_degrees": 90}.get(pred_class, 0)
        return (rotate_image_cv(image_bgr, angle), angle) if angle > 0 else (image_bgr, 0)

    def _run_single_ocr_pass(self, image_bgr_input):
        detected_polys = perform_craft_inference(
            self.craft_net, image_bgr_input, self.config.text_threshold, 
            self.config.link_threshold, self.config.low_text, 
            (self.pytorch_device.type == 'cuda'), self.config.poly)
        regions = sorted([p for p in detected_polys if p is not None], key=lambda p: np.mean(p[:, 0]), reverse=True)
        texts = []
        for poly in regions:
            crop = get_cropped_image_from_poly(image_bgr_input, poly)
            if self.config.use_simple_orientation: crop = simple_orientation_correction(crop)
            crnn_input = preprocess_for_crnn(crop, self.crnn_transform, self.pytorch_device)
            if crnn_input is None: continue
            with torch.no_grad():
                preds_log_softmax = self.crnn_model(crnn_input).log_softmax(2).permute(1, 0, 2)
                text, _ = self._decode_crnn_output(preds_log_softmax)
                if text: texts.append(text)
        return " ".join(texts), 1.0

# --- E2E Test Session & Main Execution ---
# ... [This part of the code remains unchanged] ...
def run_e2e_test_session(args, engine):
    labels_csv_path = os.path.join(args.dataset_dir, 'labels.csv')
    try: labels_df = pd.read_csv(labels_csv_path)
    except FileNotFoundError: print(f"Error: labels.csv not found at {labels_csv_path}"); return
    if args.limit_test_to > 0: labels_df = labels_df.head(args.limit_test_to)
    gts, preds = [], []
    for _, row in tqdm(labels_df.iterrows(), total=len(labels_df), desc="Evaluating Dataset"):
        img_path = os.path.join(args.dataset_dir, 'images', row['file'])
        if os.path.exists(img_path):
            gts.append(str(row['text']).strip())
            preds.append(engine.predict(img_path))
    wer = jiwer.wer(gts, preds) * 100
    cer = jiwer.cer(gts, preds) * 100
    print(f"\n--- Evaluation Results ---\nWord Error Rate (WER): {wer:.2f}%\nCharacter Error Rate (CER): {cer:.2f}%")
    if args.results_output_dir:
        os.makedirs(args.results_output_dir, exist_ok=True)
        pd.DataFrame({'file': labels_df['file'], 'ground_truth': gts, 'prediction': preds}).to_csv(os.path.join(args.results_output_dir, "e2e_crnn_greedy_test_results.csv"), index=False, encoding='utf-8-sig')
        print(f"Detailed results saved to {args.results_output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Jawi OCR Pipeline with CRAFT and CRNN (Greedy Decoder)')
    parser.add_argument('--image_path', type=str, default=None)
    parser.add_argument('--dataset_dir', type=str, default=None)
    parser.add_argument('--craft_model_path', required=True, type=str)
    parser.add_argument('--crnn_model_path', required=True, type=str)
    parser.add_argument('--alphabet_path', required=True, type=str)
    parser.add_argument('--custom_orientation_model_path', type=str, default=None)
    parser.add_argument('--results_output_dir', default='./ocr_results/', type=str)
    parser.add_argument('--limit_test_to', type=int, default=0)
    parser.add_argument('--text_threshold', default=0.7, type=float)
    parser.add_argument('--link_threshold', default=0.4, type=float)
    parser.add_argument('--low_text', default=0.4, type=float)
    parser.add_argument('--poly', default=False, action='store_true')
    parser.add_argument('--use_simple_orientation', action='store_true')
    parser.add_argument('--orientation_class_names', type=str, default='0_degrees,180_degrees,270_degrees,90_degrees')
    parser.add_argument('--orientation_confidence_threshold', type=float, default=75.0)
    parser.add_argument('--no_cuda', action='store_true')
    args = parser.parse_args()
    cudnn.benchmark = True
    engine = JawiOCREngine(config=vars(args))
    if args.image_path:
        print(f"\n--- Running prediction on: {args.image_path} ---")
        start_time = time.time()
        final_text = engine.predict(args.image_path)
        print(f"\nRecognized Text: {final_text}\nProcessing Time: {time.time() - start_time:.2f}s")
    elif args.dataset_dir:
        print(f"\n--- Starting E2E Evaluation on: {args.dataset_dir} ---")
        run_e2e_test_session(args, engine)
    else:
        print("Error: Please provide --image_path for single prediction or --dataset_dir for batch evaluation.")
