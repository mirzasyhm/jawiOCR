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
# Ensure local modules can be found
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
    """Handles loading state_dicts which may have a 'module.' prefix."""
    if list(state_dict.keys())[0].startswith("module"):
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        return new_state_dict
    return state_dict

def normalizeMeanVariance(in_img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
    """Normalizes an image for CRAFT model input."""
    img = in_img.copy().astype(np.float32)
    img -= np.array([mean[0] * 255.0, mean[1] * 255.0, mean[2] * 255.0], dtype=np.float32)
    img /= np.array([variance[0] * 255.0, variance[1] * 255.0, variance[2] * 255.0], dtype=np.float32)
    return img

def resize_aspect_ratio(img, square_size, interpolation, mag_ratio=1.):
    """Resizes an image to a square size while maintaining aspect ratio."""
    height, width, channel = img.shape
    target_size = mag_ratio * max(height, width)
    if target_size > square_size:
        target_size = square_size
    ratio = target_size / max(height, width)    
    target_h, target_w = int(height * ratio), int(width * ratio)
    proc = cv2.resize(img, (target_w, target_h), interpolation=interpolation)

    target_h32, target_w32 = target_h, target_w
    if target_h % 32 != 0:
        target_h32 = target_h + (32 - target_h % 32)
    if target_w % 32 != 0:
        target_w32 = target_w + (32 - target_w % 32)
    resized = np.zeros((target_h32, target_w32, channel), dtype=np.uint8)
    resized[0:target_h, 0:target_w, :] = proc
    return resized, ratio, (target_w, target_h)

def getDetBoxes_core(textmap, linkmap, text_threshold, link_threshold, low_text):
    linkmap = linkmap.copy()
    textmap = textmap.copy()
    img_h, img_w = textmap.shape

    _, text_score = cv2.threshold(textmap, low_text, 1, 0)
    _, link_score = cv2.threshold(linkmap, link_threshold, 1, 0)
    text_score_comb = np.clip(text_score + link_score, 0, 1)
    
    nLabels, labels, stats, _ = cv2.connectedComponentsWithStats(text_score_comb.astype(np.uint8), connectivity=4)

    det, mapper = [], []
    for k in range(1, nLabels):
        size = stats[k, cv2.CC_STAT_AREA]
        if size < 10: continue

        if np.max(textmap[labels==k]) < text_threshold: continue
        
        segmap = np.zeros(textmap.shape, dtype=np.uint8)
        segmap[labels==k] = 255
        segmap[np.logical_and(link_score==1, text_score==0)] = 0
        
        x, y = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP]
        w, h = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
        
        niter = int(math.sqrt(size * min(w, h) / (w * h))) * 2 if w > 0 and h > 0 else 0
        sx, ex, sy, ey = x - niter, x + w + niter + 1, y - niter, y + h + niter + 1
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(1 + niter, 1 + niter))
        segmap[sy:ey, sx:ex] = cv2.dilate(segmap[sy:ey, sx:ex], kernel)
        
        np_contours = np.roll(np.array(np.where(segmap!=0)),1,axis=0).transpose().reshape(-1,2)
        if np_contours.size == 0: continue

        rectangle = cv2.minAreaRect(np_contours)
        box = cv2.boxPoints(rectangle)
        
        w_box, h_box = np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2])
        if abs(1 - max(w_box, h_box) / (min(w_box, h_box) + 1e-5)) <= 0.1:
            l, r = min(np_contours[:,0]), max(np_contours[:,0])
            t, b = min(np_contours[:,1]), max(np_contours[:,1])
            box = np.array([[l, t], [r, t], [r, b], [l, b]], dtype=np.float32)
        
        startidx = box.sum(axis=1).argmin()
        box = np.roll(box, 4-startidx, 0)
        det.append(box)
        mapper.append(k)
        
    return det, labels, mapper

def getPoly_core(boxes, labels, mapper, linkmap):
    num_cp, max_len_ratio, step_r = 5, 0.7, 0.2
    polys = []
    for k, box in enumerate(boxes):
        w = int(np.linalg.norm(box[0] - box[1])) + 1
        h = int(np.linalg.norm(box[1] - box[2])) + 1
        if w < 10 or h < 10:
            polys.append(None); continue

        target = np.float32([[0,0],[w,0],[w,h],[0,h]])
        M = cv2.getPerspectiveTransform(box, target)
        word_label = cv2.warpPerspective(labels, M, (w, h), flags=cv2.INTER_NEAREST)
        try:
            Minv = np.linalg.inv(M)
        except np.linalg.LinAlgError:
            polys.append(None); continue
            
        word_label[word_label != mapper[k]] = 0
        polys.append(box)
    return polys

def getDetBoxes(textmap, linkmap, text_threshold, link_threshold, low_text, poly=False):
    boxes, labels, mapper = getDetBoxes_core(textmap, linkmap, text_threshold, link_threshold, low_text)
    if poly:
        polys = getPoly_core(boxes, labels, mapper, linkmap)
    else:
        polys = [None] * len(boxes)
    return boxes, polys

def perform_craft_inference(net, image_bgr, text_threshold, link_threshold, low_text, cuda, poly, canvas_size=1280, mag_ratio=1.5):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img_resized, target_ratio, _ = resize_aspect_ratio(image_rgb, canvas_size, cv2.INTER_LINEAR, mag_ratio)
    
    x = normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)
    if cuda:
        x = x.cuda()

    with torch.no_grad():
        y, _ = net(x)

    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()
    
    boxes, polys = getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)
    
    results_to_scale = polys if poly and any(p is not None for p in polys) else boxes
    
    final_polys = []
    for p_map in results_to_scale:
        if p_map is None: continue
        scaled_poly = p_map * 2 * (1 / target_ratio)
        final_polys.append(scaled_poly.astype(np.int32))
        
    return final_polys

# --- CRNN Model Import and Utilities ---
try:
    from crnn.model import CRNN
except ImportError:
    print("Error: Could not import CRNN from crnn/model.py. "
          "Please ensure the model definition file is accessible.")
    sys.exit(1)

CRNN_IMG_HEIGHT = 32
CRNN_IMG_WIDTH = 128
CRNN_NUM_CHANNELS = 1 # Grayscale

def load_crnn_model_local(model_path, alphabet_path, device):
    """Loads a CRNN model and its corresponding alphabet."""
    if not os.path.exists(alphabet_path):
        print(f"CRITICAL Error: Alphabet file '{alphabet_path}' not found.")
        sys.exit(1)
    try:
        with open(alphabet_path, 'r', encoding='utf-8') as f:
            alphabet_chars = json.load(f)
    except Exception as e:
        print(f"CRITICAL Error loading alphabet file: {e}"); sys.exit(1)
        
    n_class = len(alphabet_chars) + 1  # +1 for CTC blank
    crnn_model = CRNN(imgH=CRNN_IMG_HEIGHT, nc=CRNN_NUM_CHANNELS, nclass=n_class, nh=256)
    
    try:
        state_dict = torch.load(model_path, map_location=device)
        crnn_model.load_state_dict(copyStateDict(state_dict))
    except Exception as e:
        print(f"CRITICAL Error loading CRNN state_dict: {e}"); sys.exit(1)
        
    crnn_model.to(device).eval()
    
    crnn_transform = TorchTransforms.Compose([
        TorchTransforms.ToPILImage(),
        TorchTransforms.Resize((CRNN_IMG_HEIGHT, CRNN_IMG_WIDTH)),
        TorchTransforms.Grayscale(num_output_channels=1),
        TorchTransforms.ToTensor(),
        TorchTransforms.Normalize(mean=[0.5], std=[0.5])
    ])
    print("CRNN model, alphabet, and transform loaded successfully.")
    return crnn_model, crnn_transform, alphabet_chars

def preprocess_for_crnn(img_crop_bgr, crnn_transform, device):
    if img_crop_bgr is None or img_crop_bgr.size == 0: return None
    img_tensor = crnn_transform(img_crop_bgr).unsqueeze(0)
    return img_tensor.to(device)

# --- General Image Utilities (Orientation, Cropping) ---
def rotate_image_cv(image_cv, angle_degrees):
    if angle_degrees == 90: return cv2.rotate(image_cv, cv2.ROTATE_90_CLOCKWISE)
    if angle_degrees == 180: return cv2.rotate(image_cv, cv2.ROTATE_180)
    if angle_degrees == 270: return cv2.rotate(image_cv, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image_cv

def load_custom_orientation_model_keras(model_path):
    if not TF_AVAILABLE: return None
    try:
        model = tf.keras.models.load_model(model_path)
        print("Keras orientation model loaded successfully.")
        return model
    except Exception as e:
        print(f"Warning: Could not load Keras orientation model from {model_path}: {e}")
        return None

def get_custom_page_orientation(orientation_model, image_bgr, class_names):
    if not TF_AVAILABLE or orientation_model is None: return "0_degrees", 1.0
    pil_img = PILImage.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)).resize((224, 224), PILImage.Resampling.LANCZOS)
    img_array = tf_keras_image.img_to_array(pil_img)
    img_batch = np.expand_dims(img_array, axis=0)
    preprocessed_img = resnet50_preprocess_input(img_batch)
    
    predictions = orientation_model.predict(preprocessed_img, verbose=0)
    pred_index = np.argmax(predictions[0])
    confidence = np.max(predictions[0])
    return class_names[pred_index], confidence

def simple_orientation_correction(crop_bgr):
    if crop_bgr is None or crop_bgr.size == 0: return crop_bgr
    h, w = crop_bgr.shape[:2]
    if h > w and w > 0:
        return cv2.rotate(crop_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return crop_bgr

def get_cropped_image_from_poly(image_bgr, poly_pts):
    if poly_pts is None or len(poly_pts) != 4: return None
    
    poly = np.asarray(poly_pts, dtype=np.float32)
    rect = np.zeros((4, 2), dtype="float32")
    s = poly.sum(axis=1)
    rect[0] = poly[np.argmin(s)]
    rect[2] = poly[np.argmax(s)]
    diff = np.diff(poly, axis=1)
    rect[1] = poly[np.argmin(diff)]
    rect[3] = poly[np.argmax(diff)]
    
    w = max(np.linalg.norm(rect[1] - rect[0]), np.linalg.norm(rect[2] - rect[3]))
    h = max(np.linalg.norm(rect[3] - rect[0]), np.linalg.norm(rect[2] - rect[1]))
    target_w, target_h = int(round(w)), int(round(h))
    if target_w <= 0 or target_h <= 0: return None
    
    dst_pts = np.array([[0, 0], [target_w-1, 0], [target_w-1, target_h-1], [0, target_h-1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(rect, dst_pts)
    warped_crop = cv2.warpPerspective(image_bgr, M, (target_w, target_h))
    return warped_crop

# --- Main OCR Engine Class ---
class JawiOCREngine:
    def __init__(self, config):
        self.config = argparse.Namespace(**config)
        self.pytorch_device = torch.device('cuda' if not self.config.no_cuda and torch.cuda.is_available() else 'cpu')
        print(f"JawiOCREngine using device: {self.pytorch_device}")

        self.craft_net = self._load_craft_model(self.config.craft_model_path)
        self.crnn_model, self.crnn_transform, self.alphabet = self._load_crnn_model(self.config.crnn_model_path, self.config.alphabet_path)
        self.orientation_model = self._load_orientation_model()

    def _load_craft_model(self, path):
        net = CRAFT()
        state_dict = torch.load(path, map_location=self.pytorch_device)
        net.load_state_dict(copyStateDict(state_dict))
        return net.to(self.pytorch_device).eval()

    def _load_crnn_model(self, model_path, alphabet_path):
        return load_crnn_model_local(model_path, alphabet_path, self.pytorch_device)
        
    def _load_orientation_model(self):
        return load_custom_orientation_model_keras(self.config.custom_orientation_model_path) if self.config.custom_orientation_model_path else None

    def _decode_crnn_output(self, preds_log_softmax):
        """
        Decodes CRNN output using the user-specified greedy decoding method.
        """
        # Get the most likely character index at each time step
        _, max_inds = preds_log_softmax.cpu().max(2)
        
        # The pipeline processes one crop at a time, so batch size (N) is 1.
        seq = max_inds[:, 0].tolist()
        
        # Reconstruct text by removing blank tokens (0) and consecutive duplicates
        pred_text = ''.join(
            self.alphabet[c-1] for j, c in enumerate(seq)
            if c != 0 and (j == 0 or c != seq[j-1])
        )
        
        # Greedy decoding does not provide a confidence score. Returning 1.0 as a placeholder.
        confidence = 1.0 
        return pred_text, confidence

    def predict(self, image_path):
        image_bgr = cv2.imread(image_path)
        if image_bgr is None: return ""

        # Use orientation model if available to get the best page orientation first
        image_to_process, _ = self._orient_page(image_bgr)
        
        # Run a single OCR pass on the oriented image
        final_text, _ = self._run_single_ocr_pass(image_to_process)
        
        return final_text.strip()

    def _orient_page(self, image_bgr):
        if not self.orientation_model: return image_bgr, 0
        class_names = self.config.orientation_class_names.split(',')
        pred_class, conf = get_custom_page_orientation(self.orientation_model, image_bgr, class_names)
        if conf * 100 < self.config.orientation_confidence_threshold: return image_bgr, 0
        
        rotations = {"90_degrees": 270, "180_degrees": 180, "270_degrees": 90}
        angle = rotations.get(pred_class, 0)
        if angle > 0:
            return rotate_image_cv(image_bgr, angle), angle
        return image_bgr, 0

    def _run_single_ocr_pass(self, image_bgr_input):
        detected_polys = perform_craft_inference(
            self.craft_net, image_bgr_input, self.config.text_threshold, 
            self.config.link_threshold, self.config.low_text, 
            (self.pytorch_device.type == 'cuda'), self.config.poly
        )

        regions = sorted([p for p in detected_polys if p is not None], key=lambda p: np.mean(p[:, 0]), reverse=True)
        
        texts, confs = [], []
        for poly in regions:
            crop = get_cropped_image_from_poly(image_bgr_input, poly)
            if self.config.use_simple_orientation: crop = simple_orientation_correction(crop)
            
            crnn_input = preprocess_for_crnn(crop, self.crnn_transform, self.pytorch_device)
            if crnn_input is None: continue
            
            with torch.no_grad():
                raw_preds = self.crnn_model(crnn_input)
                preds_log_softmax = raw_preds.log_softmax(2).permute(1, 0, 2)
                text, conf = self._decode_crnn_output(preds_log_softmax)
                if text:
                    texts.append(text)
                    confs.append(conf)
            
        return " ".join(texts), np.mean(confs) if confs else 0.0

# --- E2E Test Session ---
def run_e2e_test_session(args, engine):
    labels_csv_path = os.path.join(args.dataset_dir, 'labels.csv')
    try:
        labels_df = pd.read_csv(labels_csv_path)
    except FileNotFoundError:
        print(f"Error: labels.csv not found at {labels_csv_path}"); return

    if args.limit_test_to > 0:
        labels_df = labels_df.head(args.limit_test_to)

    gts, preds = [], []
    for _, row in tqdm(labels_df.iterrows(), total=len(labels_df), desc="Evaluating Dataset"):
        img_path = os.path.join(args.dataset_dir, 'images', row['file'])
        if os.path.exists(img_path):
            gts.append(str(row['text']).strip())
            preds.append(engine.predict(img_path))
        else:
            print(f"Warning: Image not found {img_path}")

    wer = jiwer.wer(gts, preds) * 100
    cer = jiwer.cer(gts, preds) * 100
    
    print("\n--- Evaluation Results ---")
    print(f"Word Error Rate (WER): {wer:.2f}%")
    print(f"Character Error Rate (CER): {cer:.2f}%")

    if args.results_output_dir:
        os.makedirs(args.results_output_dir, exist_ok=True)
        results_df = pd.DataFrame({'file': labels_df['file'], 'ground_truth': gts, 'prediction': preds})
        results_df.to_csv(os.path.join(args.results_output_dir, "e2e_crnn_greedy_test_results.csv"), index=False, encoding='utf-8-sig')
        print(f"Detailed results saved to {args.results_output_dir}")

# --- Main Execution ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Jawi OCR Pipeline with CRAFT and CRNN (Greedy Decoder)')
    # Modes
    parser.add_argument('--image_path', type=str, default=None, help="Path to a single image for prediction.")
    parser.add_argument('--dataset_dir', type=str, default=None, help="Path to dataset for batch evaluation.")
    
    # Model Paths
    parser.add_argument('--craft_model_path', required=True, type=str)
    parser.add_argument('--crnn_model_path', required=True, type=str)
    parser.add_argument('--alphabet_path', required=True, type=str)
    parser.add_argument('--custom_orientation_model_path', type=str, default=None)
    
    # Output and Debug
    parser.add_argument('--results_output_dir', default='./ocr_results/', type=str)
    parser.add_argument('--save_debug_crops', action='store_true')
    parser.add_argument('--limit_test_to', type=int, default=0, help="Limit evaluation to first N images.")
    
    # OCR Parameters
    parser.add_argument('--text_threshold', default=0.7, type=float)
    parser.add_argument('--link_threshold', default=0.4, type=float)
    parser.add_argument('--low_text', default=0.4, type=float)
    parser.add_argument('--poly', default=False, action='store_true')
    parser.add_argument('--canvas_size', default=1280, type=int)
    parser.add_argument('--mag_ratio', default=1.5, type=float)
    parser.add_argument('--use_simple_orientation', action='store_true')
    parser.add_argument('--orientation_class_names', type=str, default='0_degrees,180_degrees,270_degrees,90_degrees')
    parser.add_argument('--orientation_confidence_threshold', type=float, default=75.0)
    parser.add_argument('--no_cuda', action='store_true')
    
    args = parser.parse_args()
    cudnn.benchmark = True

    ocr_config = vars(args)
    engine = JawiOCREngine(config=ocr_config)

    if args.image_path:
        print(f"\n--- Running prediction on single image: {args.image_path} ---")
        start_time = time.time()
        final_text = engine.predict(args.image_path)
        end_time = time.time()
        print(f"\nRecognized Text: {final_text}")
        print(f"Processing Time: {end_time - start_time:.2f} seconds")
    elif args.dataset_dir:
        print(f"\n--- Starting E2E Evaluation on dataset: {args.dataset_dir} ---")
        run_e2e_test_session(args, engine)
    else:
        print("Error: Please provide either --image_path for a single prediction "
              "or --dataset_dir for batch evaluation.")
