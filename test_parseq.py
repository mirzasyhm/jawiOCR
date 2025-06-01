import os
import sys
import argparse
import time
import cv2
import numpy as np
import pandas as pd # For reading labels.csv
import jiwer # For WER and CER calculation
from tqdm import tqdm # For progress bar

import torch # PyTorch
import torch.backends.cudnn as cudnn
from collections import OrderedDict
from PIL import Image as PILImage
from torchvision import transforms as TorchTransforms

# TensorFlow imports for custom orientation model (optional)
try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing import image as tf_keras_image
    from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_preprocess_input
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow not found. Custom orientation model functionality will be disabled.")

# --- Path Setup (adjust if your modules are elsewhere) ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))
craft_module_dir = os.path.join(current_script_dir, 'craft') # Assuming 'craft' is a sibling folder or adjust path
parseq_module_dir = os.path.join(current_script_dir, 'parseq_jawi') # Assuming 'parseq_jawi' is a sibling folder

if craft_module_dir not in sys.path: sys.path.insert(0, craft_module_dir)
if parseq_module_dir not in sys.path: sys.path.insert(0, parseq_module_dir)

# --- CRAFT Model Import and Utilities ---
# (Copied from paste.txt [1])
try:
    from model.craft import CRAFT
except ImportError as e:
    print(f"Error importing 'CRAFT' from 'model.craft': {e}\nSearched in: {craft_module_dir}")
    sys.exit(1)

def copyStateDict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v
    return new_state_dict

def normalizeMeanVariance(in_img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
    img = in_img.copy().astype(np.float32)
    img -= np.array([mean[0] * 255.0, mean[1] * 255.0, mean[2] * 255.0], dtype=np.float32)
    img /= np.array([variance[0] * 255.0, variance[1] * 255.0, variance[2] * 255.0], dtype=np.float32)
    return img

def resize_aspect_ratio(img, square_size, interpolation, mag_ratio=1.):
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

def getDetBoxes(textmap, linkmap, text_threshold, link_threshold, low_text, poly=False):
    boxes,labels,mapper = getDetBoxes_core(textmap,linkmap,text_threshold,link_threshold,low_text)
    if poly and boxes: polys = getPoly_core(boxes,labels,mapper,linkmap)
    else: polys = [None]*len(boxes)
    return boxes, polys

def adjustResultCoordinates(coords, ratio_w, ratio_h, ratio_net=2):
    if coords is None or len(coords)==0: return []
    adjusted_coords = []
    for item in coords:
        if item is not None: adjusted_coords.append(item*(ratio_w*ratio_net,ratio_h*ratio_net))
        else: adjusted_coords.append(None)
    return adjusted_coords

def perform_craft_inference(net,image_bgr,text_threshold,link_threshold,low_text,cuda,poly,canvas_size=1280,mag_ratio=1.5):
    image_rgb = cv2.cvtColor(image_bgr,cv2.COLOR_BGR2RGB)
    img_resized,target_ratio,_ = resize_aspect_ratio(image_rgb,canvas_size,cv2.INTER_LINEAR,mag_ratio)
    ratio_h=ratio_w=1/target_ratio
    x = normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2,0,1).unsqueeze(0)
    if cuda: x=x.cuda()
    with torch.no_grad(): y,_ = net(x)
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()
    boxes,polys_from_craft = getDetBoxes(score_text,score_link,text_threshold,link_threshold,low_text,poly) # Renamed polys to polys_from_craft
    boxes = adjustResultCoordinates(boxes,ratio_w,ratio_h)
    polys_from_craft = adjustResultCoordinates(polys_from_craft,ratio_w,ratio_h) # Renamed polys to polys_from_craft
    final_polys = []
    for k in range(max(len(polys_from_craft) if polys_from_craft else 0, len(boxes) if boxes else 0)):
        poly_item = polys_from_craft[k] if polys_from_craft and k < len(polys_from_craft) else None
        box_item = boxes[k] if boxes and k < len(boxes) else None
        if poly_item is not None: final_polys.append(poly_item)
        elif box_item is not None: final_polys.append(box_item)
    return [p for p in final_polys if p is not None]

# --- Parseq Model Import and Utilities ---
# (Copied from paste.txt [1])
try:
    from strhub.models.utils import load_from_checkpoint as parseq_load_from_checkpoint
    from strhub.data.module import SceneTextDataModule
except ImportError as e:
    print(f"Error importing Parseq components: {e}\nSearched in: {parseq_module_dir}")
    sys.exit(1)

def load_parseq_model_strhub(checkpoint_path, device):
    try:
        model = parseq_load_from_checkpoint(checkpoint_path).eval().to(device)
        img_transform = SceneTextDataModule.get_transform(model.hparams.img_size)
        if not hasattr(model,'tokenizer') or model.tokenizer is None:
             print("CRITICAL Error: Parseq model missing 'tokenizer'."); sys.exit(1)
        return model, img_transform
    except Exception as e: print(f"Error loading Parseq model: {e}"); sys.exit(1)

def preprocess_for_parseq_strhub(img_crop_bgr, parseq_transform, device):
    if img_crop_bgr is None or img_crop_bgr.shape[0]==0 or img_crop_bgr.shape[1]==0: return None
    img_rgb_pil = PILImage.fromarray(cv2.cvtColor(img_crop_bgr, cv2.COLOR_BGR2RGB))
    img_tensor = parseq_transform(img_rgb_pil).unsqueeze(0)
    return img_tensor.to(device)

# --- Image Rotation Utility ---
# (Copied from paste.txt [1])
def rotate_image_cv(image_cv, angle_degrees):
    if angle_degrees == 0: return image_cv
    elif angle_degrees == 90: return cv2.rotate(image_cv, cv2.ROTATE_90_CLOCKWISE)
    elif angle_degrees == 180: return cv2.rotate(image_cv, cv2.ROTATE_180)
    elif angle_degrees == 270: return cv2.rotate(image_cv, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else: return image_cv # Should not happen with current logic

# --- Custom TensorFlow Orientation Model Utilities (Optional) ---
# (Copied from paste.txt [1])
def load_custom_orientation_model_keras(model_path):
    if not TF_AVAILABLE:
        print("TensorFlow is not available. Cannot load Keras orientation model.")
        return None
    print(f"Loading Keras orientation model from {model_path}...")
    try:
        model = tf.keras.models.load_model(model_path)
        print("Keras orientation model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading Keras orientation model: {e}")
        return None

def preprocess_for_custom_orientation(image_bgr, target_img_size=(224, 224)):
    if not TF_AVAILABLE or image_bgr is None: return None
    pil_image = PILImage.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    pil_image_resized = pil_image.resize(target_img_size, PILImage.Resampling.LANCZOS)
    img_array = tf_keras_image.img_to_array(pil_image_resized)
    if img_array.shape[-1] == 1: 
        img_array = tf.image.grayscale_to_rgb(tf.convert_to_tensor(img_array)).numpy()
    if img_array.shape[-1] != 3:
        print(f"Error: Orient model needs 3 channels. Got {img_array.shape}"); return None
    img_array_batched = np.expand_dims(img_array, axis=0)
    preprocessed_img_array = resnet50_preprocess_input(img_array_batched.copy())
    return preprocessed_img_array

def get_custom_page_orientation(orientation_model, image_bgr, class_names, target_img_size=(224, 224)):
    if not TF_AVAILABLE or orientation_model is None:
        # print("Keras Orientation model not loaded or TF not available. Assuming 0_degrees.")
        return "0_degrees", 1.0 
    preprocessed_image = preprocess_for_custom_orientation(image_bgr, target_img_size)
    if preprocessed_image is None:
        # print("Preprocessing for Keras orientation model failed. Assuming 0_degrees.")
        return "0_degrees", 1.0
    predictions = orientation_model.predict(preprocessed_image, verbose=0) # prevent predict logs
    predicted_index = np.argmax(predictions[0])
    confidence = np.max(predictions[0]) 
    try:
        predicted_class_name = class_names[predicted_index]
        return predicted_class_name, confidence 
    except IndexError:
        # print(f"Error: Predicted index {predicted_index} out of range. Assuming 0_degrees.")
        return "0_degrees", 0.0

# --- Simple Per-Crop Orientation Correction ---
# (Copied from paste.txt [1])
def simple_orientation_correction(image_crop_bgr: np.ndarray) -> tuple[np.ndarray, str]:
    if image_crop_bgr is None or image_crop_bgr.shape[0]==0 or image_crop_bgr.shape[1]==0: 
        return image_crop_bgr, "no_action_invalid_crop"
    h,w = image_crop_bgr.shape[:2]
    if w < h: # Simple heuristic: if width is less than height, rotate
        return cv2.rotate(image_crop_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE), "rotated_90_cw_simple" # Assuming text is horizontal after rotation
    else: 
        return image_crop_bgr, "no_rotation_simple"

# --- Cropping Utility ---
# (Copied from paste.txt [1])
def get_cropped_image_from_poly(image_bgr, poly_pts):
    if poly_pts is None or len(poly_pts) != 4: return None
    poly = np.asarray(poly_pts, dtype=np.float32)
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

# --- Function to run one full OCR pass (CRAFT + Sort + Parseq) ---
# (Adapted from paste.txt [1], `args` is now `ocr_config`)
def run_ocr_pass(image_bgr_input_for_pass, base_image_filename_for_pass, pass_name, ocr_config, craft_net, parseq_model, parseq_img_transform, device, debug_output_dir, global_orientation_applied_deg=0):
    # print(f"\n--- Starting OCR Pass: {pass_name} (Input page globally rotated by {global_orientation_applied_deg} deg CW if applicable) ---")
    
    detected_polys = perform_craft_inference(
        craft_net, image_bgr_input_for_pass, ocr_config.text_threshold, ocr_config.link_threshold,
        ocr_config.low_text, (device.type=='cuda'), ocr_config.poly, ocr_config.canvas_size, ocr_config.mag_ratio
    )
    # print(f"{pass_name} - CRAFT detected {len(detected_polys)} regions.")

    regions_with_x_coords = []
    if detected_polys:
        for poly_pts in detected_polys:
            if poly_pts is not None and len(poly_pts) > 0:
                try: # Add try-except for moments calculation
                    moments = cv2.moments(poly_pts.astype(np.int32))
                    center_x = int(moments["m10"] / moments["m00"]) if moments["m00"] != 0 else int(np.mean(poly_pts[:, 0]))
                    regions_with_x_coords.append((center_x, poly_pts))
                except ZeroDivisionError: # Handle cases where m00 is zero
                    # Fallback to mean if moments calculation fails
                    center_x = int(np.mean(poly_pts[:, 0]))
                    regions_with_x_coords.append((center_x, poly_pts))
                
        regions_with_x_coords.sort(key=lambda item: item[0], reverse=True) # Sort R-L
        sorted_polys = [item[1] for item in regions_with_x_coords]
    else: sorted_polys = []
    # print(f"{pass_name} - Regions sorted R-L: {len(sorted_polys)} regions.")

    pass_results_data = [] # Though not fully used by JawiOCREngine.predict, kept for avg_conf
    pass_text_snippets = []
    
    for i, poly_pts in enumerate(sorted_polys):
        cropped_bgr = get_cropped_image_from_poly(image_bgr_input_for_pass, poly_pts)
        if cropped_bgr is None or cropped_bgr.shape[0] == 0 or cropped_bgr.shape[1] == 0:
            # print(f"{pass_name} - Skipping invalid crop for sorted region {i+1}"); 
            continue

        if ocr_config.save_debug_crops and debug_output_dir: # Check debug_output_dir exists
            crop_fn = f"{base_image_filename_for_pass}_pass_{pass_name}_sregion_{i+1}_crop.png"
            try: cv2.imwrite(os.path.join(debug_output_dir, crop_fn), cropped_bgr)
            except Exception as e: print(f"Err save crop {crop_fn}: {e}")
        
        crop_for_parseq = cropped_bgr
        simple_orientation_action = "not_applied"
        if ocr_config.use_simple_orientation:
            crop_for_parseq, simple_orientation_action = simple_orientation_correction(cropped_bgr)
            if ocr_config.save_debug_crops and debug_output_dir and (crop_for_parseq is not cropped_bgr):
                corrected_fn = f"{base_image_filename_for_pass}_pass_{pass_name}_sregion_{i+1}_corrected_{simple_orientation_action}.png"
                try: cv2.imwrite(os.path.join(debug_output_dir, corrected_fn), crop_for_parseq)
                except Exception as e: print(f"Err save corrected {corrected_fn}: {e}")

        parseq_input = preprocess_for_parseq_strhub(crop_for_parseq, parseq_img_transform, device)
        if parseq_input is None: 
            # print(f"{pass_name} - Skip region {i+1} (Parseq preprocess fail)."); 
            continue
        
        with torch.no_grad():
            logits = parseq_model(parseq_input)
            probs = logits.softmax(-1)
            # Filter out padding tokens if necessary, based on Parseq model's tokenizer
            texts, confs = parseq_model.tokenizer.decode(probs) # Assuming decode handles this
            
            if texts:
                text_seg = texts[0]
                conf_tensor = confs[0] if confs and len(confs)>0 else None # Check confs not empty
                seq_conf = None
                if conf_tensor is not None and isinstance(conf_tensor,torch.Tensor):
                    seq_conf = conf_tensor.mean().item() if conf_tensor.numel()>0 else (conf_tensor.item() if conf_tensor.numel()==1 else None) # Handle empty tensor
                
                pass_results_data.append({'text':text_seg,'conf':seq_conf}) # Simplified for avg_conf
                pass_text_snippets.append(text_seg)
            # else: print(f"{pass_name} - No text decoded for sorted region {i+1}")
            
    avg_confidence = np.mean([res['conf'] for res in pass_results_data if res['conf'] is not None]) if pass_results_data else 0.0
    final_text = " ".join(pass_text_snippets) # Join R-L sorted snippets
    # print(f"{pass_name} - Avg Confidence: {avg_confidence:.4f}, Combined Text: {final_text}")
    return final_text, avg_confidence, pass_results_data


class JawiOCREngine:
    def __init__(self, craft_model_path, parseq_model_path, ocr_config_dict):
        self.config = argparse.Namespace(**ocr_config_dict)
        self.config.craft_model_path = craft_model_path
        self.config.parseq_model_path = parseq_model_path

        self.pytorch_device = torch.device('cuda' if not self.config.no_cuda and torch.cuda.is_available() else 'cpu')
        print(f"JawiOCREngine: Using PyTorch device: {self.pytorch_device}")

        self.craft_net = self._load_craft_model()
        self.parseq_model, self.parseq_img_transform = self._load_parseq_model()
        self.orientation_model_keras = self._load_orientation_model()

    def _load_craft_model(self):
        craft_net = CRAFT()
        print(f'Loading CRAFT weights from: {self.config.craft_model_path}')
        try:
            # PyTorch 2.0+ default weights_only=True, set to False if pickle is involved
            ckpt_craft = torch.load(self.config.craft_model_path, map_location=self.pytorch_device, weights_only=False) 
        except Exception as e: # Catch pickle errors if weights_only should be True
             print(f"Warning: torch.load with weights_only=False failed ('{e}'). Trying weights_only=True.")
             try:
                 ckpt_craft = torch.load(self.config.craft_model_path, map_location=self.pytorch_device, weights_only=True)
             except Exception as e_true:
                 print(f"Error: Failed to load CRAFT model with both weights_only False and True: {e_true}")
                 sys.exit(1)
        
        if 'craft' in ckpt_craft: craft_sd = ckpt_craft['craft']
        elif 'model' in ckpt_craft: craft_sd = ckpt_craft['model']
        elif 'state_dict' in ckpt_craft: craft_sd = ckpt_craft['state_dict']
        else: craft_sd = ckpt_craft
        craft_net.load_state_dict(copyStateDict(craft_sd))
        craft_net.to(self.pytorch_device)
        craft_net.eval()
        print("CRAFT model loaded successfully.")
        return craft_net

    def _load_parseq_model(self):
        print(f"Loading Parseq model from: {self.config.parseq_model_path}")
        model, transform = load_parseq_model_strhub(self.config.parseq_model_path, self.pytorch_device)
        print("Parseq model loaded successfully.")
        return model, transform
        
    def _load_orientation_model(self):
        if hasattr(self.config, 'custom_orientation_model_path') and self.config.custom_orientation_model_path:
            if TF_AVAILABLE:
                return load_custom_orientation_model_keras(self.config.custom_orientation_model_path)
            else:
                print("Skipping Keras orientation model load as TensorFlow is not available.")
        return None

    def predict(self, image_path):
        image_bgr_original = cv2.imread(image_path)
        if image_bgr_original is None:
            # print(f"Error: Could not read image {image_path}")
            return "" 
        
        base_image_filename = os.path.splitext(os.path.basename(image_path))[0]
        
        debug_output_dir_for_image = ""
        if self.config.save_debug_crops:
            if not hasattr(self.config, 'output_dir') or not self.config.output_dir:
                 print("Warning: save_debug_crops is True, but output_dir (for e2e results) is not set. Debug crops might not be saved properly.")
            else:
                # Save debug crops in a subfolder named after the image inside the main debug_crops folder
                debug_output_base = os.path.join(self.config.output_dir, "debug_crops")
                debug_output_dir_for_image = os.path.join(debug_output_base, base_image_filename)
                if not os.path.exists(debug_output_dir_for_image): 
                    os.makedirs(debug_output_dir_for_image, exist_ok=True)

        image_for_pass1 = image_bgr_original.copy()
        initial_rotation_applied_deg = 0
        
        if self.orientation_model_keras and hasattr(self.config, 'orientation_class_names'):
            keras_class_names = self.config.orientation_class_names.split(',')
            pred_orient_class, pred_orient_conf = get_custom_page_orientation(
                self.orientation_model_keras, image_bgr_original, keras_class_names,
                target_img_size=(self.config.orientation_img_size, self.config.orientation_img_size)
            )
            
            if pred_orient_class == "90_degrees" and pred_orient_conf * 100 >= self.config.orientation_confidence_threshold:
                image_for_pass1 = rotate_image_cv(image_bgr_original, 270) 
                initial_rotation_applied_deg = 270
            elif pred_orient_class == "270_degrees" and pred_orient_conf * 100 >= self.config.orientation_confidence_threshold:
                image_for_pass1 = rotate_image_cv(image_bgr_original, 90)
                initial_rotation_applied_deg = 90
            
            if self.config.save_debug_crops and initial_rotation_applied_deg != 0 and debug_output_dir_for_image:
                fn = f"{base_image_filename}_page_custom_oriented_{initial_rotation_applied_deg}deg.png"
                try: cv2.imwrite(os.path.join(debug_output_dir_for_image, fn), image_for_pass1)
                except Exception as e: print(f"Err saving oriented page: {e}")
        
        final_text_pass1, avg_conf_pass1, _ = run_ocr_pass(
            image_for_pass1, base_image_filename, "Pass1", self.config, 
            self.craft_net, self.parseq_model, self.parseq_img_transform, self.pytorch_device, 
            debug_output_dir_for_image, # Pass specific dir for this image
            global_orientation_applied_deg=initial_rotation_applied_deg
        )
    
        chosen_final_text = final_text_pass1
        
        if avg_conf_pass1 * 100 < self.config.rerun_180_threshold:
            # print(f"\nPass 1 conf ({avg_conf_pass1*100:.2f}%) < threshold ({self.config.rerun_180_threshold}%). Re-running with 180-deg rotation.")
            image_for_pass2 = rotate_image_cv(image_for_pass1, 180)
            if self.config.save_debug_crops and debug_output_dir_for_image:
                fn = f"{base_image_filename}_page_for_pass2_180rot.png"
                try: cv2.imwrite(os.path.join(debug_output_dir_for_image, fn), image_for_pass2)
                except Exception as e: print(f"Err saving 180-rot page: {e}")

            final_text_pass2, avg_conf_pass2, _ = run_ocr_pass(
                image_for_pass2, base_image_filename, "Pass2_180_Rot", self.config,
                self.craft_net, self.parseq_model, self.parseq_img_transform, self.pytorch_device, 
                debug_output_dir_for_image, # Pass specific dir
                global_orientation_applied_deg=(initial_rotation_applied_deg + 180) % 360
            )
            if avg_conf_pass2 > avg_conf_pass1:
                # print("Pass 2 (180-deg rotated) > Pass 1. Using Pass 2 results.")
                chosen_final_text = final_text_pass2
            # else:
                # print("Pass 1 >= Pass 2. Sticking with Pass 1 results.")
        # else:
            # print(f"\nPass 1 conf ({avg_conf_pass1*100:.2f}%) sufficient. Skipping 180-deg re-run.")
        
        return chosen_final_text.strip()


def run_e2e_test_session(test_args):
    labels_csv_path = os.path.join(test_args.dataset_dir, 'labels.csv')
    try:
        labels_df = pd.read_csv(labels_csv_path)
    except FileNotFoundError:
        print(f"Error: labels.csv not found at {labels_csv_path}")
        sys.exit(1)
    
    if not {'file', 'text'}.issubset(labels_df.columns):
        print(f"Error: labels.csv must contain 'file' and 'text' columns.")
        sys.exit(1)
    
     # --- Modification to limit to the first N images ---
    if test_args.limit_test_to_n_images > 0: # Check if a limit is set
        if test_args.limit_test_to_n_images < len(labels_df):
            print(f"INFO: Limiting test to the first {test_args.limit_test_to_n_images} images as requested.")
            labels_df = labels_df.head(test_args.limit_test_to_n_images)
        else:
            print(f"INFO: Requested limit ({test_args.limit_test_to_n_images}) is >= total images ({len(labels_df)}). Processing all available images in labels.csv.")
    # --- End of modification ---

    # Prepare OCR configuration
    ocr_config = {
        "text_threshold": test_args.text_threshold, "link_threshold": test_args.link_threshold,
        "low_text": test_args.low_text, "poly": test_args.poly, "canvas_size": test_args.canvas_size,
        "mag_ratio": test_args.mag_ratio, 
        "custom_orientation_model_path": test_args.custom_orientation_model_path if TF_AVAILABLE else None,
        "orientation_class_names": test_args.orientation_class_names,
        "orientation_img_size": test_args.orientation_img_size,
        "orientation_confidence_threshold": test_args.orientation_confidence_threshold,
        "use_simple_orientation": test_args.use_simple_orientation,
        "rerun_180_threshold": test_args.rerun_180_threshold,
        "save_debug_crops": test_args.save_debug_crops, "no_cuda": test_args.no_cuda,
        "output_dir": test_args.results_output_dir # This is the main output dir for test results
    }

    print("Initializing JawiOCR Engine...")
    ocr_engine = JawiOCREngine(
        craft_model_path=test_args.craft_model_path,
        parseq_model_path=test_args.parseq_model_path,
        ocr_config_dict=ocr_config
    )
    print("JawiOCR Engine initialized.")

    all_ground_truths = []
    all_predictions = []
    skipped_images_count = 0
    
    print(f"\nProcessing {len(labels_df)} images from {test_args.dataset_dir}...")
    for index, row in tqdm(labels_df.iterrows(), total=len(labels_df), desc="Evaluating Images"):
        image_filename = row['file']
        gt_text = str(row['text']).strip() if pd.notna(row['text']) else "" # Handle NaN in ground truth
        
        image_full_path = os.path.join(test_args.dataset_dir, 'images', image_filename)
        if not os.path.exists(image_full_path):
            print(f"Warning: Image file {image_full_path} (from labels.csv row {index+2}) not found. Skipping.")
            # To keep gt and pred lists aligned for jiwer, add GT and an empty prediction
            all_ground_truths.append(gt_text)
            all_predictions.append("") # Indicate failed processing for this file
            skipped_images_count +=1
            continue

        pred_text = ocr_engine.predict(image_full_path) # Already stripped by engine

        all_ground_truths.append(gt_text)
        all_predictions.append(pred_text)

    if skipped_images_count > 0:
        print(f"Warning: Skipped {skipped_images_count} images due to missing files.")

    if not all_ground_truths:
        print("No data to evaluate (all images might have been skipped or labels.csv was empty).")
        return

    print("\n--- Evaluation Results ---")
    
    correct_sentences = 0
    for gt, pred in zip(all_ground_truths, all_predictions):
        if gt == pred:
            correct_sentences += 1
    sentence_accuracy = (correct_sentences / len(all_ground_truths)) * 100 if all_ground_truths else 0
    print(f"Sentence Recognition Accuracy: {sentence_accuracy:.2f}% ({correct_sentences}/{len(all_ground_truths)})")

    try:
        # Calculate WER using compute_measures or jiwer.wer()
        # Using jiwer.wer() directly is often clearer for just WER
        wer_value = jiwer.wer(all_ground_truths, all_predictions) # Returns a fraction
        wer = wer_value * 100

        # Calculate CER using the dedicated jiwer.cer() function
        cer_value = jiwer.cer(all_ground_truths, all_predictions) # Returns a fraction
        cer = cer_value * 100
        
        # Word Recognition Accuracy is often taken as 100 - WER
        word_recognition_accuracy = (1.0 - wer_value) * 100 # Use the fractional wer_value here
        
        print(f"Word Error Rate (WER): {wer:.2f}%")
        print(f"Character Error Rate (CER): {cer:.2f}%")
        print(f"Word Recognition Accuracy (100 - WER): {word_recognition_accuracy:.2f}%")

    except Exception as e:
        print(f"Error calculating WER/CER with jiwer: {e}")
        print("Skipping WER/CER calculation. This can happen if all ground truths are empty or due to other library issues.")
        wer, cer, word_recognition_accuracy = float('nan'), float('nan'), float('nan')


    print("\n--- First 10 Predictions vs Ground Truths ---")
    for i in range(min(10, len(all_ground_truths))):
        img_file_name = labels_df['file'].iloc[i] if i < len(labels_df) else "N/A"
        print(f"Image {i+1} ({img_file_name}):")
        print(f"  GT  : '{all_ground_truths[i]}'")
        print(f"  Pred: '{all_predictions[i]}'")
        print("-" * 20)

    if test_args.results_output_dir:
        if not os.path.exists(test_args.results_output_dir):
             os.makedirs(test_args.results_output_dir, exist_ok=True)
        
        results_df_data = {
            'file': labels_df['file'].iloc[:len(all_predictions)].tolist(), # Ensure correct length
            'ground_truth': all_ground_truths,
            'prediction': all_predictions
        }
        results_df = pd.DataFrame(results_df_data)
        results_df['exact_match'] = (results_df['ground_truth'] == results_df['prediction'])
        
        output_results_csv = os.path.join(test_args.results_output_dir, "e2e_test_results.csv")
        try:
            results_df.to_csv(output_results_csv, index=False, encoding='utf-8')
            print(f"\nDetailed results saved to: {output_results_csv}")
        except Exception as e:
            print(f"Error saving detailed results to CSV '{output_results_csv}': {e}")

        summary_path = os.path.join(test_args.results_output_dir, "e2e_test_summary.txt")
        try:
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write("--- JawiOCR E2E Test Summary ---\n")
                f.write(f"Dataset Directory: {test_args.dataset_dir}\n")
                f.write(f"Total Images Processed: {len(all_ground_truths)}\n")
                if skipped_images_count > 0:
                    f.write(f"Images Skipped (not found): {skipped_images_count}\n")
                f.write(f"Sentence Recognition Accuracy: {sentence_accuracy:.2f}% ({correct_sentences}/{len(all_ground_truths)})\n")
                f.write(f"Word Error Rate (WER): {wer:.2f}%\n")
                f.write(f"Character Error Rate (CER): {cer:.2f}%\n")
                f.write(f"Word Recognition Accuracy (100 - WER): {word_recognition_accuracy:.2f}%\n")
            print(f"Summary saved to: {summary_path}")
        except Exception as e:
            print(f"Error saving summary to TXT '{summary_path}': {e}")


if __name__ == '__main__':
    if TF_AVAILABLE:
        print(f"TensorFlow version: {tf.__version__}")
        gpus_tf = tf.config.list_physical_devices('GPU')
        if gpus_tf: print(f"TensorFlow Found GPUs: {gpus_tf}")
        else: print("TensorFlow: No GPU found.")
    
    cudnn.benchmark = True # For PyTorch

    test_parser = argparse.ArgumentParser(description='JawiOCR End-to-End Test Session Script')
    test_parser.add_argument('--dataset_dir', required=True, type=str, help="Path to the dataset directory (must contain 'images/' subfolder and 'labels.csv')")
    test_parser.add_argument('--results_output_dir', default='./e2e_jawiocr_results/', type=str, help="Directory to save evaluation CSV, summary TXT, and optional debug crops")
    
    test_parser.add_argument('--craft_model_path', required=True, type=str, help="Path to pre-trained CRAFT model (.pth file)")
    test_parser.add_argument('--parseq_model_path', required=True, type=str, help="Path to pre-trained Parseq model (.ckpt or .pth file)")
    test_parser.add_argument('--custom_orientation_model_path', type=str, default=None, help="Path to Keras custom orientation model (.h5 file, optional)")
    test_parser.add_argument('--limit_test_to_n_images', type=int, default=0, help="Limit the test to the first N images. Set to 0 to process all images (default).")
    # OCR Parameters (defaults from paste.txt [1])
    test_parser.add_argument('--orientation_class_names', type=str, default='0_degrees,180_degrees,270_degrees,90_degrees', help="Comma-separated class names for orientation model (e.g., '0_degrees,90_degrees,180_degrees,270_degrees')")
    test_parser.add_argument('--orientation_img_size', type=int, default=224, help="Input image size for Keras orientation model")
    test_parser.add_argument('--orientation_confidence_threshold', type=float, default=75.0, help="Confidence threshold (0-100) for applying global page orientation from Keras model")
    test_parser.add_argument('--text_threshold', default=0.7, type=float, help="Text confidence threshold for CRAFT text detection")
    test_parser.add_argument('--low_text', default=0.4, type=float, help="Text low_text score threshold for CRAFT")
    test_parser.add_argument('--link_threshold', default=0.4, type=float, help="Link confidence threshold for CRAFT text detection")
    test_parser.add_argument('--canvas_size', default=1280, type=int, help="Maximum dimension for resizing image before CRAFT processing")
    test_parser.add_argument('--mag_ratio', default=1.5, type=float, help="Image magnification ratio before CRAFT processing")
    test_parser.add_argument('--poly', default=False, action='store_true', help="Use polygon detection for CRAFT instead of bounding boxes")
    test_parser.add_argument('--use_simple_orientation', action='store_true', help="Enable simple per-crop orientation correction (rotates if width < height)")
    test_parser.add_argument('--rerun_180_threshold', type=float, default=90.0, help="Average confidence threshold (0-100) for Parseq. If Pass1 confidence is below this, a 180-degree rotated re-run is triggered.")
    test_parser.add_argument('--save_debug_crops', action='store_true', help="Save intermediate debug images and cropped regions during OCR processing")
    test_parser.add_argument('--no_cuda', action='store_true', help="Force CPU usage, disable CUDA even if available")
    
    parsed_test_args = test_parser.parse_args()
    
    run_e2e_test_session(parsed_test_args)
