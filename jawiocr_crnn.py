import os
import sys
import argparse
import time
import cv2
import math
import numpy as np
import pandas as pd
import jiwer 
from tqdm import tqdm
import json # For loading alphabet

import torch
import torch.backends.cudnn as cudnn
from collections import OrderedDict
from PIL import Image as PILImage
from torchvision import transforms as TorchTransforms # For CRNN preprocessing

# TensorFlow imports (optional, for orientation model)
try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing import image as tf_keras_image
    from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_preprocess_input
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow not found. Custom orientation model functionality will be disabled.")

# --- Path Setup ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))
craft_module_dir = os.path.join(current_script_dir, 'craft')
if craft_module_dir not in sys.path: sys.path.insert(0, craft_module_dir)

# --- CRAFT Model Import and Utilities ---
try:
    from model.craft import CRAFT # From the 'craft' submodule
except ImportError as e:
    print(f"Error importing 'CRAFT' from 'model.craft': {e}\nEnsure 'craft' submodule is in the Python path (e.g., in same dir or added to sys.path).")
    sys.exit(1)

# --- Start of CRAFT utility functions (from original file) ---
def copyStateDict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v
    return new_state_dict

def normalizeMeanVariance(in_img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)): # For CRAFT
    img = in_img.copy().astype(np.float32)
    img -= np.array([mean[0] * 255.0, mean[1] * 255.0, mean[2] * 255.0], dtype=np.float32)
    img /= np.array([variance[0] * 255.0, variance[1] * 255.0, variance[2] * 255.0], dtype=np.float32)
    return img

def resize_aspect_ratio(img, square_size, interpolation, mag_ratio=1.): # For CRAFT
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

def getDetBoxes_core(textmap, linkmap, text_threshold, link_threshold, low_text):
    linkmap, textmap = linkmap.copy(), textmap.copy()
    img_h, img_w = textmap.shape
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
        x,y,w,h = stats[k,cv2.CC_STAT_LEFT],stats[k,cv2.CC_STAT_TOP],stats[k,cv2.CC_STAT_WIDTH],stats[k,cv2.CC_STAT_HEIGHT]
        niter = int(np.sqrt(size * min(w,h) / (w*h)) * 2) if w*h > 0 else 0
        sx,ex,sy,ey = max(0,x-niter),min(img_w,x+w+niter+1),max(0,y-niter),min(img_h,y+h+niter+1)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(1+niter,1+niter))
        segmap[sy:ey,sx:ex] = cv2.dilate(segmap[sy:ey,sx:ex], kernel)
        np_contours = np.roll(np.array(np.where(segmap!=0)),1,axis=0).transpose().reshape(-1,2)
        if np_contours.size == 0: continue
        rect = cv2.minAreaRect(np_contours)
        box_pts = cv2.boxPoints(rect)
        box_w,box_h = np.linalg.norm(box_pts[0]-box_pts[1]),np.linalg.norm(box_pts[1]-box_pts[2])
        if abs(1-max(box_w,box_h)/(min(box_w,box_h)+1e-5)) <= 0.1:
            l,r,t,b = min(np_contours[:,0]),max(np_contours[:,0]),min(np_contours[:,1]),max(np_contours[:,1])
            box_pts = np.array([[l,t],[r,t],[r,b],[l,b]], dtype=np.float32)
        startidx = box_pts.sum(axis=1).argmin()
        box_pts = np.roll(box_pts, 4-startidx, 0)
        det.append(box_pts); mapper.append(k)
    return det, labels, mapper

def getPoly_core(boxes, labels, mapper, linkmap):
    num_cp,max_len_ratio,max_r,step_r = 5,0.7,2.0,0.2
    polys = []
    for k, box_pts_param in enumerate(boxes):
        w_box,h_box=int(np.linalg.norm(box_pts_param[0]-box_pts_param[1])+0.5),int(np.linalg.norm(box_pts_param[1]-box_pts_param[2])+0.5)
        if w_box<10 or h_box<10: polys.append(None); continue
        tar = np.float32([[0,0],[w_box,0],[w_box,h_box],[0,h_box]])
        M = cv2.getPerspectiveTransform(box_pts_param, tar)
        word_label = cv2.warpPerspective(labels, M, (w_box,h_box), flags=cv2.INTER_NEAREST)
        try: Minv = np.linalg.inv(M)
        except np.linalg.LinAlgError: polys.append(None); continue
        word_label[word_label!=mapper[k]]=0; word_label[word_label>0]=1
        cp_top_list,cp_bot_list,cp_checked = [],[],False
        for i in range(num_cp):
            curr = int(w_box/(num_cp-1)*i) if num_cp > 1 else 0
            if curr==w_box and w_box>0: curr-=1
            if not (0<=curr<word_label.shape[1]): continue
            pts = np.where(word_label[:,curr]!=0)[0]
            if not pts.size > 0: continue
            cp_checked=True
            cp_top_list.append(np.array([curr,pts[0]],dtype=np.int32))
            cp_bot_list.append(np.array([curr,pts[-1]],dtype=np.int32))
        if not cp_checked or len(cp_top_list)<num_cp or len(cp_bot_list)<num_cp: polys.append(None); continue
        cp_top,cp_bot = np.array(cp_top_list).reshape(-1,2),np.array(cp_bot_list).reshape(-1,2)
        final_poly = None
        for r_val in np.arange(0.5,max_r,step_r):
            top_link_pts,bot_link_pts=0,0
            if cp_top.shape[0]>=2: top_link_pts=sum(1 for i_pt in range(cp_top.shape[0]-1) if 0<=int((cp_top[i_pt][1]+cp_top[i_pt+1][1])/2)<linkmap.shape[0] and 0<=int((cp_top[i_pt][0]+cp_top[i_pt+1][0])/2)<linkmap.shape[1] and linkmap[int((cp_top[i_pt][1]+cp_top[i_pt+1][1])/2),int((cp_top[i_pt][0]+cp_top[i_pt+1][0])/2)]==1)
            if cp_bot.shape[0]>=2: bot_link_pts=sum(1 for i_pt in range(cp_bot.shape[0]-1) if 0<=int((cp_bot[i_pt][1]+cp_bot[i_pt+1][1])/2)<linkmap.shape[0] and 0<=int((cp_bot[i_pt][0]+cp_bot[i_pt+1][0])/2)<linkmap.shape[1] and linkmap[int((cp_bot[i_pt][1]+cp_bot[i_pt+1][1])/2),int((cp_bot[i_pt][0]+cp_bot[i_pt+1][0])/2)]==1)
            if top_link_pts>max_len_ratio*cp_top.shape[0] or bot_link_pts>max_len_ratio*cp_bot.shape[0]: final_poly=None; break
            if top_link_pts==0 and bot_link_pts==0:
                poly_coords=np.concatenate((cp_top,np.flip(cp_bot,axis=0)),axis=0)
                final_poly=cv2.perspectiveTransform(np.array([poly_coords],dtype=np.float32),Minv)[0]; break
        polys.append(final_poly)
    return polys

def getDetBoxes(textmap, linkmap, text_threshold, link_threshold, low_text, poly=False): # For CRAFT
    boxes,labels,mapper = getDetBoxes_core(textmap,linkmap,text_threshold,link_threshold,low_text)
    if poly and boxes: polys_from_craft = getPoly_core(boxes,labels,mapper,linkmap)
    else: polys_from_craft = [None]*len(boxes)
    return boxes, polys_from_craft
    
def perform_craft_inference(net,image_bgr,text_threshold,link_threshold,low_text,cuda,poly,canvas_size=1280,mag_ratio=1.5): # For CRAFT
    image_rgb = cv2.cvtColor(image_bgr,cv2.COLOR_BGR2RGB)
    img_resized, target_ratio, _ = resize_aspect_ratio(image_rgb, canvas_size, cv2.INTER_LINEAR, mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    x = normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)
    if cuda: x = x.cuda()
    
    with torch.no_grad(): y, _ = net(x)
    
    score_text = y[0, :, :, 0].cpu().data.numpy()
    score_link = y[0, :, :, 1].cpu().data.numpy()
    
    boxes_map, polys_map = getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)
    
    final_polys = []
    for p_map in (polys_map if poly else boxes_map):
        if p_map is None: continue
        scaled_poly = (p_map * 2 * ratio_w) if isinstance(ratio_w, (int, float)) else (p_map * 2 * np.array([ratio_w, ratio_h]))
        final_polys.append(scaled_poly.astype(np.int32))
        
    return final_polys
# --- End of CRAFT utility functions ---

# --- CRNN Model Import and Utilities ---
try:
    from crnn.model import CRNN # Your CRNN model class
except ImportError:
    print("Error: Could not import CRNN from model.py. Ensure model.py is in the Python path.")
    sys.exit(1)

CRNN_IMG_HEIGHT = 32
CRNN_IMG_WIDTH = 128
CRNN_NUM_CHANNELS = 1

def load_crnn_model_local(model_path, alphabet_path, device):
    print(f"Loading CRNN model from: {model_path}")
    print(f"Loading alphabet from: {alphabet_path}")
    
    if not os.path.exists(alphabet_path):
        print(f"CRITICAL Error: Alphabet file '{alphabet_path}' not found.")
        sys.exit(1)
    try:
        with open(alphabet_path, 'r', encoding='utf-8') as f:
            alphabet_chars = json.load(f)
    except Exception as e:
        print(f"CRITICAL Error: Could not load or parse alphabet file '{alphabet_path}': {e}")
        sys.exit(1)
        
    n_class = len(alphabet_chars) + 1 # +1 for CTC blank

    crnn_model = CRNN(imgH=CRNN_IMG_HEIGHT, nc=CRNN_NUM_CHANNELS, nclass=n_class, nh=256) # Assuming nh=256
    
    try:
        state_dict = torch.load(model_path, map_location=device)
        crnn_model.load_state_dict(state_dict)
    except Exception as e:
        print(f"CRITICAL Error loading CRNN state_dict: {e}")
        sys.exit(1)
        
    crnn_model = crnn_model.to(device).eval()
    
    crnn_transform = TorchTransforms.Compose([
        TorchTransforms.ToPILImage(),
        TorchTransforms.Resize((CRNN_IMG_HEIGHT, CRNN_IMG_WIDTH)),
        TorchTransforms.Grayscale(num_output_channels=1),
        TorchTransforms.ToTensor(),
        TorchTransforms.Normalize(mean=[0.5], std=[0.5])
    ])
    print("CRNN model and alphabet loaded successfully.")
    return crnn_model, crnn_transform, alphabet_chars


def preprocess_for_crnn_local(img_crop_bgr, crnn_transform, device):
    if img_crop_bgr is None or img_crop_bgr.shape[0] == 0 or img_crop_bgr.shape[1] == 0:
        return None
    img_tensor = crnn_transform(img_crop_bgr).unsqueeze(0)
    return img_tensor.to(device)

# --- Image Rotation, Custom Orientation, Cropping Utilities ---
def rotate_image_cv(image_cv, angle_degrees):
    if angle_degrees == 0: return image_cv
    elif angle_degrees == 90: return cv2.rotate(image_cv, cv2.ROTATE_90_CLOCKWISE)
    elif angle_degrees == 180: return cv2.rotate(image_cv, cv2.ROTATE_180)
    elif angle_degrees == 270: return cv2.rotate(image_cv, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else: return image_cv

def load_custom_orientation_model_keras(model_path):
    if not TF_AVAILABLE:
        print("TensorFlow is not available. Cannot load Keras orientation model.")
        return None
    print(f"Loading Keras orientation model from {model_path}...")
    try: model = tf.keras.models.load_model(model_path); print("Keras model loaded."); return model
    except Exception as e: print(f"Error loading Keras model: {e}"); return None

def preprocess_for_custom_orientation(image_bgr, target_img_size=(224, 224)):
    if not TF_AVAILABLE or image_bgr is None: return None
    pil_image = PILImage.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    pil_image_resized = pil_image.resize(target_img_size, PILImage.Resampling.LANCZOS)
    img_array = tf_keras_image.img_to_array(pil_image_resized)
    if img_array.shape[-1] == 1: img_array = tf.image.grayscale_to_rgb(tf.convert_to_tensor(img_array)).numpy()
    if img_array.shape[-1] != 3: print(f"Error: Orient model needs 3 channels. Got {img_array.shape}"); return None
    img_array_batched = np.expand_dims(img_array, axis=0)
    preprocessed_img_array = resnet50_preprocess_input(img_array_batched.copy())
    return preprocessed_img_array

def get_custom_page_orientation(orientation_model, image_bgr, class_names, target_img_size=(224, 224)):
    if not TF_AVAILABLE or orientation_model is None: return "0_degrees", 1.0 
    preprocessed_image = preprocess_for_custom_orientation(image_bgr, target_img_size)
    if preprocessed_image is None: return "0_degrees", 1.0
    predictions = orientation_model.predict(preprocessed_image, verbose=0)
    predicted_index = np.argmax(predictions[0])
    confidence = np.max(predictions[0]) 
    try: return class_names[predicted_index], confidence 
    except IndexError: return "0_degrees", 0.0

def simple_orientation_correction(image_crop_bgr: np.ndarray) -> tuple[np.ndarray, str]:
    if image_crop_bgr is None or image_crop_bgr.shape[0]==0 or image_crop_bgr.shape[1]==0: return image_crop_bgr, "no_action_invalid_crop"
    h,w = image_crop_bgr.shape[:2]
    if w < h and w > 0:
        return cv2.rotate(image_crop_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE), "rotated_90_ccw_simple"
    else: return image_crop_bgr, "no_rotation_simple"

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
    poly = rect

    w1 = np.linalg.norm(poly[1] - poly[0])
    w2 = np.linalg.norm(poly[2] - poly[3])
    h1 = np.linalg.norm(poly[3] - poly[0])
    h2 = np.linalg.norm(poly[2] - poly[1])
    target_w = int(round(max(w1, w2)))
    target_h = int(round(max(h1, h2)))
    if target_w <= 0 or target_h <= 0: return None
    dst_pts = np.array([[0, 0], [target_w - 1, 0], [target_w - 1, target_h - 1], [0, target_h - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(poly, dst_pts)
    warped_crop = cv2.warpPerspective(image_bgr, M, (target_w, target_h))
    return warped_crop if warped_crop.size else None

# --- OCR Pass Function ---
def run_ocr_pass(image_bgr_input_for_pass, base_image_filename_for_pass, pass_name, ocr_engine,
                 debug_output_dir, global_orientation_applied_deg=0):
    
    detected_polys = perform_craft_inference(
        ocr_engine.craft_net, image_bgr_input_for_pass, ocr_engine.config.text_threshold, 
        ocr_engine.config.link_threshold, ocr_engine.config.low_text, 
        (ocr_engine.pytorch_device.type == 'cuda'), ocr_engine.config.poly, 
        ocr_engine.config.canvas_size, ocr_engine.config.mag_ratio
    )
    
    regions_with_x_coords = []
    if detected_polys:
        for poly_pts in detected_polys:
            if poly_pts is not None and len(poly_pts) > 0:
                try:
                    moments = cv2.moments(poly_pts.astype(np.int32))
                    center_x = int(moments["m10"] / moments["m00"]) if moments["m00"] != 0 else int(np.mean(poly_pts[:, 0]))
                    regions_with_x_coords.append((center_x, poly_pts))
                except (ZeroDivisionError, ValueError):
                    center_x = int(np.mean(poly_pts[:, 0]))
                    regions_with_x_coords.append((center_x, poly_pts))
        regions_with_x_coords.sort(key=lambda item: item[0], reverse=True)
        sorted_polys = [item[1] for item in regions_with_x_coords]
    else:
        sorted_polys = []

    pass_text_snippets = []
    pass_confidences = []

    for i, poly_pts in enumerate(sorted_polys):
        cropped_bgr = get_cropped_image_from_poly(image_bgr_input_for_pass, poly_pts)
        if cropped_bgr is None or cropped_bgr.shape[0] == 0 or cropped_bgr.shape[1] == 0:
            continue

        if ocr_engine.config.save_debug_crops and debug_output_dir:
            crop_fn = f"{base_image_filename_for_pass}_pass_{pass_name}_sregion_{i+1}_crop.png"
            try: cv2.imwrite(os.path.join(debug_output_dir, crop_fn), cropped_bgr)
            except Exception as e: print(f"Err save crop {crop_fn}: {e}")
        
        crop_for_crnn = cropped_bgr
        if ocr_engine.config.use_simple_orientation:
            crop_for_crnn, _ = simple_orientation_correction(cropped_bgr)
        
        crnn_input_tensor = preprocess_for_crnn_local(
            crop_for_crnn, ocr_engine.crnn_img_transform, ocr_engine.pytorch_device
        )
        if crnn_input_tensor is None:
            continue
        
        with torch.no_grad():
            raw_preds = ocr_engine.crnn_model(crnn_input_tensor)
            preds_log_softmax = raw_preds.log_softmax(2)

            text_segment, confidence = ocr_engine._decode_crnn_output_with_greedy_search(preds_log_softmax)
            
            if text_segment:
                pass_text_snippets.append(text_segment)
                pass_confidences.append(confidence)
            
    avg_pass_conf = np.mean(pass_confidences) if pass_confidences else 0.0
    final_text = " ".join(pass_text_snippets)
    
    return final_text, avg_pass_conf, None

# --- JawiOCREngine Class ---
class JawiOCREngine:
    def __init__(self, craft_model_path, crnn_model_path, alphabet_path, ocr_config_dict):
        self.config = argparse.Namespace(**ocr_config_dict)
        self.config.craft_model_path = craft_model_path
        self.config.crnn_model_path = crnn_model_path
        self.config.alphabet_path = alphabet_path

        self.pytorch_device = torch.device('cuda' if not self.config.no_cuda and torch.cuda.is_available() else 'cpu')
        print(f"JawiOCREngine: Using PyTorch device: {self.pytorch_device}")

        self.craft_net = self._load_craft_model()
        self.crnn_model, self.crnn_img_transform, self.crnn_alphabet_chars = self._load_crnn_model()
        self.orientation_model_keras = self._load_orientation_model()

    def _load_craft_model(self):
        craft_net = CRAFT()
        print(f'Loading CRAFT weights from: {self.config.craft_model_path}')
        try: ckpt_craft = torch.load(self.config.craft_model_path, map_location=self.pytorch_device, weights_only=False) 
        except Exception: ckpt_craft = torch.load(self.config.craft_model_path, map_location=self.pytorch_device, weights_only=True)
        
        if 'craft' in ckpt_craft: craft_sd = ckpt_craft['craft']
        elif 'model' in ckpt_craft: craft_sd = ckpt_craft['model']
        elif 'state_dict' in ckpt_craft: craft_sd = ckpt_craft['state_dict']
        else: craft_sd = ckpt_craft
        craft_net.load_state_dict(copyStateDict(craft_sd))
        craft_net.to(self.pytorch_device); craft_net.eval()
        print("CRAFT model loaded successfully.")
        return craft_net

    def _load_crnn_model(self):
        print(f"Loading CRNN model from: {self.config.crnn_model_path}")
        model, transform, alphabet = load_crnn_model_local(
            self.config.crnn_model_path, 
            self.config.alphabet_path, 
            self.pytorch_device
        )
        return model, transform, alphabet
    
    def _decode_crnn_output_with_greedy_search(self, log_probs_tensor):
        """
        Decodes CRNN log-probabilities using a simple greedy (best path) decoding.
        """
        log_probs_cpu = log_probs_tensor.cpu()
        max_probs, pred_indices_batch = log_probs_cpu.max(2)
        pred_indices_sample = pred_indices_batch.squeeze(1).tolist()
        max_probs_sample = max_probs.squeeze(1)

        decoded_chars = []
        confidence_scores = []
        
        for i, char_index in enumerate(pred_indices_sample):
            if char_index == 0:
                continue
            if i > 0 and char_index == pred_indices_sample[i - 1]:
                continue
            
            decoded_chars.append(self.crnn_alphabet_chars[char_index])
            confidence_scores.append(max_probs_sample[i].item())

        final_text = "".join(decoded_chars)

        if confidence_scores:
            confidence = np.mean([math.exp(score) for score in confidence_scores])
        else:
            confidence = 0.0

        return final_text, confidence
        
    def _load_orientation_model(self):
        if hasattr(self.config, 'custom_orientation_model_path') and self.config.custom_orientation_model_path:
            if TF_AVAILABLE: return load_custom_orientation_model_keras(self.config.custom_orientation_model_path)
            else: print("Skipping Keras orientation model: TensorFlow not available.")
        return None

    def predict(self, image_path):
        image_bgr_original = cv2.imread(image_path)
        if image_bgr_original is None: 
            print(f"Warning: Could not read image at {image_path}. Skipping.")
            return ""
        
        base_image_filename = os.path.splitext(os.path.basename(image_path))[0]
        debug_output_dir_for_image = ""
        if self.config.save_debug_crops and hasattr(self.config, 'output_dir') and self.config.output_dir:
            debug_output_base = os.path.join(self.config.output_dir, "debug_crops")
            debug_output_dir_for_image = os.path.join(debug_output_base, base_image_filename)
            os.makedirs(debug_output_dir_for_image, exist_ok=True)

        image_for_pass1 = image_bgr_original.copy()
        initial_rotation_applied_deg = 0
        
        if self.orientation_model_keras and hasattr(self.config, 'orientation_class_names'):
            keras_class_names = self.config.orientation_class_names.split(',')
            pred_orient_class, pred_orient_conf = get_custom_page_orientation(
                self.orientation_model_keras, image_bgr_original, keras_class_names,
                target_img_size=(self.config.orientation_img_size, self.config.orientation_img_size)
            )
            new_rotation_deg = 0
            if pred_orient_class == "90_degrees": new_rotation_deg = 270
            elif pred_orient_class == "270_degrees": new_rotation_deg = 90
            elif pred_orient_class == "180_degrees": new_rotation_deg = 180

            if new_rotation_deg != 0 and pred_orient_conf * 100 >= self.config.orientation_confidence_threshold:
                image_for_pass1 = rotate_image_cv(image_bgr_original, new_rotation_deg) 
                initial_rotation_applied_deg = new_rotation_deg
                if self.config.save_debug_crops and debug_output_dir_for_image:
                    fn = f"{base_image_filename}_page_custom_oriented_{initial_rotation_applied_deg}deg.png"
                    try: cv2.imwrite(os.path.join(debug_output_dir_for_image, fn), image_for_pass1)
                    except Exception as e: print(f"Err saving oriented page: {e}")

        final_text_pass1, metric_pass1, _ = run_ocr_pass(
            image_for_pass1, 
            base_image_filename, 
            "Pass1", 
            self,
            debug_output_dir_for_image,
            global_orientation_applied_deg=initial_rotation_applied_deg
        )

        chosen_final_text = final_text_pass1
        
        rerun_condition_met = False
        if metric_pass1 * 100 < self.config.rerun_180_confidence_threshold:
            print(f"INFO: Pass 1 confidence ({metric_pass1*100:.2f}%) is below threshold ({self.config.rerun_180_confidence_threshold}%). Rerunning with 180-degree rotation.")
            rerun_condition_met = True
        
        if rerun_condition_met:
            image_for_pass2 = rotate_image_cv(image_for_pass1, 180)
            
            final_text_pass2, metric_pass2, _ = run_ocr_pass(
                image_for_pass2, 
                base_image_filename, 
                "Pass2_180_Rot", 
                self,
                debug_output_dir_for_image,
                global_orientation_applied_deg=(initial_rotation_applied_deg + 180) % 360
            )
            
            if metric_pass2 > metric_pass1:
                chosen_final_text = final_text_pass2
                print(f"INFO: Pass 2 result chosen (Confidence: {metric_pass2*100:.2f}%)")
            else:
                print(f"INFO: Pass 1 result kept (Confidence: {metric_pass1*100:.2f}%)")
                
        return chosen_final_text.strip()

# --- NEW: Function to handle single image prediction ---
def run_single_image_prediction(args):
    """Initializes the engine and runs OCR on a single image."""
    if not os.path.exists(args.image_path):
        print(f"Error: Image file not found at '{args.image_path}'")
        sys.exit(1)
    
    # Check for required model paths
    if not args.craft_model_path or not args.crnn_model_path or not args.alphabet_path:
        print("Error: For single image prediction, you must provide --craft_model_path, --crnn_model_path, and --alphabet_path.")
        sys.exit(1)

    ocr_config = {
        "text_threshold": args.text_threshold, "link_threshold": args.link_threshold,
        "low_text": args.low_text, "poly": args.poly, "canvas_size": args.canvas_size,
        "mag_ratio": args.mag_ratio, 
        "custom_orientation_model_path": args.custom_orientation_model_path if TF_AVAILABLE else None,
        "orientation_class_names": args.orientation_class_names,
        "orientation_img_size": args.orientation_img_size,
        "orientation_confidence_threshold": args.orientation_confidence_threshold,
        "use_simple_orientation": args.use_simple_orientation,
        "rerun_180_confidence_threshold": args.rerun_180_confidence_threshold, 
        "save_debug_crops": args.save_debug_crops, "no_cuda": args.no_cuda,
        "output_dir": args.results_output_dir 
    }

    print("Initializing JawiOCREngine for single prediction...")
    ocr_engine = JawiOCREngine(
        craft_model_path=args.craft_model_path,
        crnn_model_path=args.crnn_model_path,
        alphabet_path=args.alphabet_path,
        ocr_config_dict=ocr_config
    )
    print("Engine initialized.")

    print(f"\nRunning OCR on: {args.image_path}")
    start_time = time.time()
    predicted_text = ocr_engine.predict(args.image_path)
    end_time = time.time()

    print("\n" + "="*20)
    print("     OCR RESULT")
    print("="*20)
    print(predicted_text)
    print("="*20)
    print(f"\nProcessed in {end_time - start_time:.2f} seconds.")

# --- E2E Test Session (Batch Evaluation) ---
def run_e2e_test_session(test_args):
    labels_csv_path = os.path.join(test_args.dataset_dir, 'labels.csv')
    try: labels_df = pd.read_csv(labels_csv_path)
    except FileNotFoundError: print(f"Error: labels.csv not found at {labels_csv_path}"); sys.exit(1)
    if not {'file', 'text'}.issubset(labels_df.columns): print("Error: labels.csv must have 'file','text'"); sys.exit(1)

    if test_args.limit_test_to_n_images > 0:
        if test_args.limit_test_to_n_images < len(labels_df):
            print(f"INFO: Limiting test to the first {test_args.limit_test_to_n_images} images as requested.")
            labels_df = labels_df.head(test_args.limit_test_to_n_images)
        else:
            print(f"INFO: Requested limit ({test_args.limit_test_to_n_images}) is >= total images ({len(labels_df)}). Processing all available images in labels.csv.")

    ocr_config = {
        "text_threshold": test_args.text_threshold, "link_threshold": test_args.link_threshold,
        "low_text": test_args.low_text, "poly": test_args.poly, "canvas_size": test_args.canvas_size,
        "mag_ratio": test_args.mag_ratio, 
        "custom_orientation_model_path": test_args.custom_orientation_model_path if TF_AVAILABLE else None,
        "orientation_class_names": test_args.orientation_class_names,
        "orientation_img_size": test_args.orientation_img_size,
        "orientation_confidence_threshold": test_args.orientation_confidence_threshold,
        "use_simple_orientation": test_args.use_simple_orientation,
        "rerun_180_confidence_threshold": test_args.rerun_180_confidence_threshold, 
        "save_debug_crops": test_args.save_debug_crops, "no_cuda": test_args.no_cuda,
        "output_dir": test_args.results_output_dir
    }

    print("Initializing JawiOCREngine for batch evaluation...")
    ocr_engine = JawiOCREngine(
        craft_model_path=test_args.craft_model_path,
        crnn_model_path=test_args.crnn_model_path,
        alphabet_path=test_args.alphabet_path,
        ocr_config_dict=ocr_config
    )
    print("JawiOCREngine initialized.")

    all_ground_truths = []
    all_predictions = []
    skipped_images_count = 0
    
    print(f"\nProcessing {len(labels_df)} images from {test_args.dataset_dir}...")
    for index, row in tqdm(labels_df.iterrows(), total=len(labels_df), desc="Evaluating Images"):
        image_filename = row['file']
        gt_text = str(row['text']).strip() if pd.notna(row['text']) else ""
        image_full_path = os.path.join(test_args.dataset_dir, 'images', image_filename)
        if not os.path.exists(image_full_path):
            print(f"Warning: Image file {image_full_path} not found. Skipping.")
            all_ground_truths.append(gt_text); all_predictions.append("") 
            skipped_images_count +=1; continue

        pred_text = ocr_engine.predict(image_full_path)
        all_ground_truths.append(gt_text); all_predictions.append(pred_text)

    if skipped_images_count > 0: print(f"Warning: Skipped {skipped_images_count} images.")
    if not all_ground_truths: print("No data to evaluate."); return

    print("\n--- Evaluation Results ---")
    correct_sentences = sum(1 for gt, pred in zip(all_ground_truths, all_predictions) if gt == pred)
    sentence_accuracy = (correct_sentences / len(all_ground_truths)) * 100 if all_ground_truths else 0
    print(f"Sentence Recognition Accuracy: {sentence_accuracy:.2f}% ({correct_sentences}/{len(all_ground_truths)})")

    try:
        wer_value = jiwer.wer(all_ground_truths, all_predictions)
        wer = wer_value * 100
        cer_value = jiwer.cer(all_ground_truths, all_predictions)
        cer = cer_value * 100
        word_recognition_accuracy = (1.0 - wer_value) * 100
        
        print(f"Word Error Rate (WER): {wer:.2f}%")
        print(f"Character Error Rate (CER): {cer:.2f}%")
        print(f"Word Recognition Accuracy (100 - WER): {word_recognition_accuracy:.2f}%")
    except Exception as e:
        print(f"Error calculating WER/CER with jiwer: {e}")
        wer, cer, word_recognition_accuracy = float('nan'), float('nan'), float('nan')

    print("\n--- First 10 Predictions vs Ground Truths ---")
    for i in range(min(10, len(all_ground_truths))):
        img_file_name = labels_df['file'].iloc[i] if i < len(labels_df) else "N/A"
        print(f"Image {i+1} ({img_file_name}):")
        print(f"  GT  : '{all_ground_truths[i]}'")
        print(f"  Pred: '{all_predictions[i]}'")
        print("-" * 20)

    if test_args.results_output_dir:
        os.makedirs(test_args.results_output_dir, exist_ok=True)
        results_df_data = {'file': labels_df['file'].iloc[:len(all_predictions)].tolist(), 'ground_truth': all_ground_truths, 'prediction': all_predictions}
        results_df = pd.DataFrame(results_df_data)
        results_df['exact_match'] = (results_df['ground_truth'] == results_df['prediction'])
        output_results_csv = os.path.join(test_args.results_output_dir, "e2e_crnn_test_results.csv")
        try: results_df.to_csv(output_results_csv, index=False, encoding='utf-8'); print(f"\nDetailed results saved to: {output_results_csv}")
        except Exception as e: print(f"Error saving results CSV: {e}")
        summary_path = os.path.join(test_args.results_output_dir, "e2e_crnn_test_summary.txt")
        try:
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write("--- JawiOCR E2E CRNN Test Summary ---\n")
                f.write(f"Dataset Directory: {test_args.dataset_dir}\nTotal Images: {len(all_ground_truths)}\n")
                if skipped_images_count > 0: f.write(f"Skipped: {skipped_images_count}\n")
                f.write(f"Sentence Acc: {sentence_accuracy:.2f}%\nWER: {wer:.2f}%\nCER: {cer:.2f}%\nWord Acc: {word_recognition_accuracy:.2f}%\n")
            print(f"Summary saved to: {summary_path}")
        except Exception as e: print(f"Error saving summary TXT: {e}")

# --- Argument Parser and Main Execution ---
if __name__ == '__main__':
    if TF_AVAILABLE:
        print(f"TensorFlow version: {tf.__version__}")
        gpus_tf = tf.config.list_physical_devices('GPU')
        if gpus_tf: print(f"TensorFlow Found GPUs: {gpus_tf}")
        else: print("TensorFlow: No GPU found.")
    
    cudnn.benchmark = True

    # MODIFIED: Updated parser description
    parser = argparse.ArgumentParser(description='JawiOCR: Run on a single image or perform batch evaluation.')
    
    # --- Mode Selection Arguments ---
    parser.add_argument('--image_path', type=str, default=None, help="Path to a single image file for direct OCR.")
    # MODIFIED: Made dataset_dir optional
    parser.add_argument('--dataset_dir', type=str, default=None, help="Path to evaluation dataset (must contain 'images/' and 'labels.csv').")
    
    # --- Model and Path Arguments (Required for both modes) ---
    parser.add_argument('--craft_model_path', required=True, type=str, help="Path to CRAFT model (.pth).")
    parser.add_argument('--crnn_model_path', required=True, type=str, help="Path to pre-trained CRNN model (.pth state_dict).")
    parser.add_argument('--alphabet_path', required=True, type=str, help="Path to alphabet.json for CRNN.")
    parser.add_argument('--custom_orientation_model_path', type=str, default=None, help="Path to Keras orientation model (.h5).")

    # --- Configuration and Hyperparameters ---
    parser.add_argument('--results_output_dir', default='./jawiocr_results/', type=str, help="Output directory for results and debug files.")
    parser.add_argument('--limit_test_to_n_images', type=int, default=0, help="For batch evaluation, limit to the first N images (0 for all).")
    parser.add_argument('--orientation_class_names', type=str, default='0_degrees,180_degrees,270_degrees,90_degrees')
    parser.add_argument('--orientation_img_size', type=int, default=224)
    parser.add_argument('--orientation_confidence_threshold', type=float, default=75.0)
    parser.add_argument('--text_threshold', default=0.7, type=float)
    parser.add_argument('--low_text', default=0.4, type=float)
    parser.add_argument('--link_threshold', default=0.4, type=float)
    parser.add_argument('--canvas_size', default=1280, type=int)
    parser.add_argument('--mag_ratio', default=1.5, type=float)
    parser.add_argument('--poly', default=False, action='store_true')
    parser.add_argument('--use_simple_orientation', action='store_true')
    parser.add_argument('--rerun_180_confidence_threshold', type=float, default=75.0, help="If Pass1 confidence is below this, trigger 180-deg re-run.")
    parser.add_argument('--save_debug_crops', action='store_true')
    parser.add_argument('--no_cuda', action='store_true')
    
    args = parser.parse_args()
        
    # --- Main Logic: Decide which mode to run ---
    if args.image_path:
        run_single_image_prediction(args)
    elif args.dataset_dir:
        run_e2e_test_session(args)
    else:
        print("Error: No mode selected. You must provide either --image_path for single image OCR or --dataset_dir for batch evaluation.")
        parser.print_help()
        sys.exit(1)
