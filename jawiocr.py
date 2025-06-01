import os
import sys
import argparse
import time
import cv2
import numpy as np
import torch # PyTorch
import torch.backends.cudnn as cudnn
from collections import OrderedDict
from PIL import Image as PILImage
from torchvision import transforms as TorchTransforms

# TensorFlow imports for custom orientation model
import tensorflow as tf
from tensorflow.keras.preprocessing import image as tf_keras_image
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_preprocess_input


# --- Path Setup ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))
craft_module_dir = os.path.join(current_script_dir, 'craft')
parseq_module_dir = os.path.join(current_script_dir, 'parseq_jawi')

if craft_module_dir not in sys.path: sys.path.insert(0, craft_module_dir)
if parseq_module_dir not in sys.path: sys.path.insert(0, parseq_module_dir)
print(f"Adjusted sys.path. Current sys.path[0:3]: {sys.path[0:3]}")
print(f"TensorFlow version: {tf.__version__}")
gpus_tf = tf.config.list_physical_devices('GPU')
if gpus_tf: print(f"TensorFlow Found GPUs: {gpus_tf}")
else: print("TensorFlow: No GPU found.")


# --- CRAFT Model Import and Utilities ---
try:
    from model.craft import CRAFT
except ImportError as e:
    print(f"Error importing 'CRAFT' from 'model.craft': {e}\nLooked in: {craft_module_dir}"); sys.exit(1)

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
    boxes,polys = getDetBoxes(score_text,score_link,text_threshold,link_threshold,low_text,poly)
    boxes = adjustResultCoordinates(boxes,ratio_w,ratio_h)
    polys = adjustResultCoordinates(polys,ratio_w,ratio_h)
    final_polys = []
    for k in range(max(len(polys) if polys else 0, len(boxes) if boxes else 0)):
        poly_item = polys[k] if polys and k < len(polys) else None
        box_item = boxes[k] if boxes and k < len(boxes) else None
        if poly_item is not None: final_polys.append(poly_item)
        elif box_item is not None: final_polys.append(box_item)
    return [p for p in final_polys if p is not None]

# --- Parseq Model Import and Utilities ---
try:
    from strhub.models.utils import load_from_checkpoint as parseq_load_from_checkpoint
    from strhub.data.module import SceneTextDataModule
except ImportError as e:
    print(f"Error importing Parseq components: {e}\nLooked in: {parseq_module_dir}"); sys.exit(1)

def load_parseq_model_strhub(checkpoint_path, device):
    print(f"Loading Parseq model from: {checkpoint_path}")
    try:
        model = parseq_load_from_checkpoint(checkpoint_path).eval().to(device)
        print("Parseq model loaded successfully.")
        img_transform = SceneTextDataModule.get_transform(model.hparams.img_size)
        if not hasattr(model,'tokenizer') or model.tokenizer is None:
            print("CRITICAL Error: Parseq model missing 'tokenizer'."); sys.exit(1)
        print(f"Parseq tokenizer found. Type: {type(model.tokenizer)}")
        return model, img_transform
    except Exception as e: print(f"Error loading Parseq model: {e}"); sys.exit(1)

def preprocess_for_parseq_strhub(img_crop_bgr, parseq_transform, device):
    if img_crop_bgr is None or img_crop_bgr.shape[0]==0 or img_crop_bgr.shape[1]==0: return None
    img_rgb_pil = PILImage.fromarray(cv2.cvtColor(img_crop_bgr, cv2.COLOR_BGR2RGB))
    img_tensor = parseq_transform(img_rgb_pil).unsqueeze(0)
    return img_tensor.to(device)

# --- Image Rotation Utility ---
def rotate_image_cv(image_cv, angle_degrees):
    if angle_degrees == 0: return image_cv
    elif angle_degrees == 90: return cv2.rotate(image_cv, cv2.ROTATE_90_CLOCKWISE)
    elif angle_degrees == 180: return cv2.rotate(image_cv, cv2.ROTATE_180)
    elif angle_degrees == 270: return cv2.rotate(image_cv, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else: return image_cv

# --- Custom TensorFlow Orientation Model Utilities ---
def load_custom_orientation_model_keras(model_path):
    print(f"Loading Keras orientation model from {model_path}...")
    try:
        model = tf.keras.models.load_model(model_path)
        print("Keras orientation model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading Keras orientation model: {e}")
        return None

def preprocess_for_custom_orientation(image_bgr, target_img_size=(224, 224)):
    if image_bgr is None: return None
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
    if orientation_model is None:
        print("Keras Orientation model not loaded. Assuming 0_degrees.")
        return "0_degrees", 1.0 
    preprocessed_image = preprocess_for_custom_orientation(image_bgr, target_img_size)
    if preprocessed_image is None:
        print("Preprocessing for Keras orientation model failed. Assuming 0_degrees.")
        return "0_degrees", 1.0
    predictions = orientation_model.predict(preprocessed_image)
    predicted_index = np.argmax(predictions[0])
    confidence = np.max(predictions[0]) 
    try:
        predicted_class_name = class_names[predicted_index]
        # Output for global orientation is printed in main_ocr_pipeline now
        return predicted_class_name, confidence 
    except IndexError:
        print(f"Error: Predicted index {predicted_index} out of range. Assuming 0_degrees.")
        return "0_degrees", 0.0

# --- Simple Per-Crop Orientation Correction ---
def simple_orientation_correction(image_crop_bgr: np.ndarray) -> tuple[np.ndarray, str]: # Returns image and action string
    if image_crop_bgr is None or image_crop_bgr.shape[0]==0 or image_crop_bgr.shape[1]==0: 
        return image_crop_bgr, "no_action_invalid_crop"
    h,w = image_crop_bgr.shape[:2]
    if w < h: 
        return cv2.rotate(image_crop_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE), "rotated_90_cw_simple"
    else: 
        return image_crop_bgr, "no_rotation_simple"

# --- Cropping Utility ---
def get_cropped_image_from_poly(image_bgr, poly_pts):
    """
    Rectify and crop a quadrilateral region from `image_bgr`.

    Parameters
    ----------
    image_bgr : np.ndarray
        OpenCV BGR image.
    poly_pts  : list/np.ndarray, shape (4, 2)
        Quad vertices in consistent clockwise order:
        TL, TR, BR, BL  (float or int).

    Returns
    -------
    warped_crop : np.ndarray  |  None
        The upright crop, or None if the quad is degenerate.
    """
    # 1. quick validation
    if poly_pts is None or len(poly_pts) != 4:
        return None
    poly = np.asarray(poly_pts, dtype=np.float32)

    # 2. derive target width & height directly from edge lengths
    w1 = np.linalg.norm(poly[1] - poly[0])   # top edge  (TL → TR)
    w2 = np.linalg.norm(poly[2] - poly[3])   # bottom    (BR → BL)
    h1 = np.linalg.norm(poly[3] - poly[0])   # left edge (BL → TL)
    h2 = np.linalg.norm(poly[2] - poly[1])   # right     (BR → TR)

    target_w = int(round(max(w1, w2)))
    target_h = int(round(max(h1, h2)))
    if target_w <= 0 or target_h <= 0:
        return None                          # degenerate quad

    # 3. destination rectangle (same order: TL, TR, BR, BL)
    dst_pts = np.array([
        [0, 0],
        [target_w - 1, 0],
        [target_w - 1, target_h - 1],
        [0, target_h - 1]
    ], dtype=np.float32)

    # 4. compute homography & warp
    M = cv2.getPerspectiveTransform(poly, dst_pts)
    warped_crop = cv2.warpPerspective(image_bgr, M, (target_w, target_h))

    return warped_crop if warped_crop.size else None

# --- Function to run one full OCR pass (CRAFT + Sort + Parseq) ---
def run_ocr_pass(image_bgr_input_for_pass, base_image_filename_for_pass, pass_name, args, craft_net, parseq_model, parseq_img_transform, device, debug_output_dir, global_orientation_applied_deg=0):
    print(f"\n--- Starting OCR Pass: {pass_name} (Input page globally rotated by {global_orientation_applied_deg} deg CW if applicable) ---")
    
    detected_polys = perform_craft_inference(
        craft_net, image_bgr_input_for_pass, args.text_threshold, args.link_threshold,
        args.low_text, (device.type=='cuda'), args.poly, args.canvas_size, args.mag_ratio
    )
    print(f"{pass_name} - CRAFT detected {len(detected_polys)} regions.")

    regions_with_x_coords = []
    if detected_polys:
        for poly_pts in detected_polys:
            if poly_pts is not None and len(poly_pts) > 0:
                moments = cv2.moments(poly_pts.astype(np.int32))
                center_x = int(moments["m10"] / moments["m00"]) if moments["m00"] != 0 else int(np.mean(poly_pts[:, 0]))
                regions_with_x_coords.append((center_x, poly_pts))
        regions_with_x_coords.sort(key=lambda item: item[0], reverse=True)
        sorted_polys = [item[1] for item in regions_with_x_coords]
    else: sorted_polys = []
    print(f"{pass_name} - Regions sorted R-L: {len(sorted_polys)} regions.")

    pass_results_data = []
    pass_text_snippets = []
    
    for i, poly_pts in enumerate(sorted_polys):
        cropped_bgr = get_cropped_image_from_poly(image_bgr_input_for_pass, poly_pts)
        if cropped_bgr is None or cropped_bgr.shape[0] == 0 or cropped_bgr.shape[1] == 0:
            print(f"{pass_name} - Skipping invalid crop for sorted region {i+1}"); continue

        if args.save_debug_crops:
            crop_fn = f"{base_image_filename_for_pass}_pass_{pass_name}_sregion_{i+1}_crop.png"
            try: cv2.imwrite(os.path.join(debug_output_dir, crop_fn), cropped_bgr)
            except Exception as e: print(f"Err save crop {crop_fn}: {e}")
        
        crop_for_parseq = cropped_bgr
        simple_orientation_action = "not_applied"
        if args.use_simple_orientation:
            crop_for_parseq, simple_orientation_action = simple_orientation_correction(cropped_bgr) # Get action
            if args.save_debug_crops and crop_for_parseq is not cropped_bgr: # Check if changed
                corrected_fn = f"{base_image_filename_for_pass}_pass_{pass_name}_sregion_{i+1}_corrected_{simple_orientation_action}.png"
                try: cv2.imwrite(os.path.join(debug_output_dir, corrected_fn), crop_for_parseq)
                except Exception as e: print(f"Err save corrected {corrected_fn}: {e}")

        parseq_input = preprocess_for_parseq_strhub(crop_for_parseq, parseq_img_transform, device)
        if parseq_input is None: print(f"{pass_name} - Skip region {i+1} (Parseq preprocess fail)."); continue
        
        with torch.no_grad():
            logits = parseq_model(parseq_input)
            probs = logits.softmax(-1)
            texts, confs = parseq_model.tokenizer.decode(probs)
            if texts:
                text_seg = texts[0]
                conf_tensor = confs[0] if confs else None
                seq_conf = None
                if conf_tensor is not None and isinstance(conf_tensor,torch.Tensor):
                    seq_conf = conf_tensor.mean().item() if conf_tensor.numel()>1 else conf_tensor.item()
                
                x_key = regions_with_x_coords[i][0] if i < len(regions_with_x_coords) else "N/A"
                conf_s = f"{seq_conf:.4f}" if seq_conf is not None else "N/A"
                print(f"{pass_name} SRegion {i+1}(X:{x_key}, SimpleOrient:{simple_orientation_action}):Txt='{text_seg}',Conf={conf_s}") # Added simple_orientation_action
                pass_results_data.append({'orig_x':x_key,'poly':poly_pts,'text':text_seg,'conf':seq_conf, 'simple_orientation': simple_orientation_action}) # Store action
                pass_text_snippets.append(text_seg)
            else: print(f"{pass_name} - No text decoded for sorted region {i+1}")
            
    avg_confidence = np.mean([res['conf'] for res in pass_results_data if res['conf'] is not None]) if pass_results_data else 0.0
    final_text = " ".join(pass_text_snippets)
    print(f"{pass_name} - Avg Confidence: {avg_confidence:.4f}, Combined Text: {final_text}")
    return final_text, avg_confidence, pass_results_data

# --- Main OCR Pipeline ---
def main_ocr_pipeline(args):
    cuda_enabled = not args.no_cuda and torch.cuda.is_available()
    pytorch_device = torch.device('cuda' if cuda_enabled else 'cpu')
    print(f"Using PyTorch device: {pytorch_device}")

    debug_output_dir = ""
    if args.save_debug_crops:
        debug_output_dir = os.path.join(args.output_dir, "debug_crops")
        if not os.path.exists(debug_output_dir): os.makedirs(debug_output_dir)
        print(f"Created debug crops directory: {debug_output_dir}")

    orientation_model_keras = None
    if args.custom_orientation_model_path:
        orientation_model_keras = load_custom_orientation_model_keras(args.custom_orientation_model_path)
        if orientation_model_keras is None: print("Proceeding without custom page orientation.")

    print(f"Processing image: {args.image_path}")
    image_bgr_original = cv2.imread(args.image_path)
    if image_bgr_original is None: print(f"Error: Could not read image {args.image_path}"); return
    base_image_filename = os.path.splitext(os.path.basename(args.image_path))[0]
    
    image_for_pass1 = image_bgr_original.copy()
    initial_rotation_applied_deg = 0 
    global_orientation_prediction_str = "N/A (Keras model not used or failed)"
    
    if orientation_model_keras:
        keras_class_names = args.orientation_class_names.split(',')
        pred_orient_class, pred_orient_conf = get_custom_page_orientation(
            orientation_model_keras, image_bgr_original, keras_class_names, 
            target_img_size=(args.orientation_img_size, args.orientation_img_size)
        )
        global_orientation_prediction_str = f"{pred_orient_class} (Conf: {pred_orient_conf:.2f})" # Store for logging
        print(f"Global Page Orientation Prediction by Keras Model: {global_orientation_prediction_str}")

        if pred_orient_class == "90_degrees" and pred_orient_conf * 100 >= args.orientation_confidence_threshold:
            image_for_pass1 = rotate_image_cv(image_bgr_original, 270) 
            initial_rotation_applied_deg = 270
        elif pred_orient_class == "270_degrees" and pred_orient_conf * 100 >= args.orientation_confidence_threshold:
            image_for_pass1 = rotate_image_cv(image_bgr_original, 90)
            initial_rotation_applied_deg = 90
        
        if args.save_debug_crops and initial_rotation_applied_deg != 0:
            fn = f"{base_image_filename}_page_custom_oriented_{initial_rotation_applied_deg}deg.png"
            cv2.imwrite(os.path.join(debug_output_dir, fn), image_for_pass1)
    else:
        print("Custom Keras orientation model not used for global page orientation.")
    
    craft_net = CRAFT()
    print(f'Loading CRAFT w: {args.craft_model_path}')
    ckpt_craft = torch.load(args.craft_model_path,map_location=pytorch_device,weights_only=False)
    if 'craft' in ckpt_craft: craft_sd = ckpt_craft['craft']
    elif 'model' in ckpt_craft: craft_sd = ckpt_craft['model']
    elif 'state_dict' in ckpt_craft: craft_sd = ckpt_craft['state_dict']
    else: craft_sd = ckpt_craft
    craft_net.load_state_dict(copyStateDict(craft_sd)); craft_net.to(pytorch_device); craft_net.eval()

    parseq_model,parseq_img_transform = load_parseq_model_strhub(args.parseq_model_path,pytorch_device)

    final_text_pass1, avg_conf_pass1, results_data_pass1 = run_ocr_pass(
        image_for_pass1, base_image_filename, "Pass1", args, 
        craft_net, parseq_model, parseq_img_transform, pytorch_device, debug_output_dir,
        global_orientation_applied_deg=initial_rotation_applied_deg # Pass this info
    )
    
    chosen_final_text = final_text_pass1
    chosen_results_data = results_data_pass1
    chosen_pass_name = "Pass1"
    image_for_visualization = image_for_pass1.copy()

    if avg_conf_pass1 * 100 < args.rerun_180_threshold:
        print(f"\nPass 1 conf ({avg_conf_pass1*100:.2f}%) < threshold ({args.rerun_180_threshold}%). Re-running with 180-deg rotation.")
        image_for_pass2 = rotate_image_cv(image_for_pass1, 180)
        if args.save_debug_crops:
            fn = f"{base_image_filename}_page_for_pass2_180rot.png"
            cv2.imwrite(os.path.join(debug_output_dir, fn), image_for_pass2)

        final_text_pass2, avg_conf_pass2, results_data_pass2 = run_ocr_pass(
            image_for_pass2, base_image_filename, "Pass2_180_Rot", args,
            craft_net, parseq_model, parseq_img_transform, pytorch_device, debug_output_dir,
            global_orientation_applied_deg=(initial_rotation_applied_deg + 180) % 360 # Track total rotation for this pass
        )
        
        if avg_conf_pass2 > avg_conf_pass1:
            print("Pass 2 (180-deg rotated) > Pass 1. Using Pass 2 results.")
            chosen_final_text = final_text_pass2
            chosen_results_data = results_data_pass2
            chosen_pass_name = "Pass2_180_Rot"
            image_for_visualization = image_for_pass2.copy()
        else:
            print("Pass 1 >= Pass 2. Sticking with Pass 1 results.")
    else:
        print(f"\nPass 1 conf ({avg_conf_pass1*100:.2f}%) sufficient. Skipping 180-deg re-run.")

    print(f"\n--- Final Chosen Result (from {chosen_pass_name}) ---")
    print(f"Initial Global Page Orientation by Keras Model: {global_orientation_prediction_str}")
    print(f"Global Page Rotation Applied for Chosen Pass: {initial_rotation_applied_deg if chosen_pass_name == 'Pass1' else (initial_rotation_applied_deg + 180)%360} deg CW")
    print(f"Final Combined Right-to-Left Text: {chosen_final_text}\n")

    if args.output_dir:
        if not os.path.exists(args.output_dir): os.makedirs(args.output_dir)
        final_output_image_viz = image_for_visualization.copy()
        for res_item_viz in chosen_results_data:
            if 'polygon' in res_item_viz and res_item_viz['polygon'] is not None:
                 try:
                     poly_to_draw = np.array(res_item_viz['polygon'], dtype=np.float32)
                     cv2.polylines(final_output_image_viz, [poly_to_draw.astype(np.int32)], True, (0,0,255), 2)
                 except Exception as e_draw: print(f"Warn: Could not draw polygon: {e_draw}")

        viz_filepath = os.path.join(args.output_dir,f"res_ocr_{base_image_filename}_final.jpg")
        cv2.imwrite(viz_filepath, final_output_image_viz)
        print(f"Final viz saved to: {viz_filepath}")
        
        txt_filepath = os.path.join(args.output_dir,f"res_ocr_{base_image_filename}_final.txt")
        with open(txt_filepath,'w',encoding='utf-8') as f:
            f.write(f"Final Chosen Pass: {chosen_pass_name}\n")
            f.write(f"Initial Global Page Orientation by Keras Model: {global_orientation_prediction_str}\n")
            f.write(f"Global Page Rotation Applied for Chosen Pass Input: {initial_rotation_applied_deg if chosen_pass_name == 'Pass1' else (initial_rotation_applied_deg + 180)%360} deg CW\n")
            f.write(f"Final Combined Text (R-L): {chosen_final_text}\n\n")
            f.write(f"Individual Region Detections (from chosen pass, sorted R-L):\n")
            f.write(f"X-Key|SimpleOrient|Polygon|Text|Confidence\n") # Header for clarity
            for res in chosen_results_data:
                poly_s = "N/A_Poly"
                if 'polygon' in res and res['polygon'] is not None:
                    try: 
                        current_poly = np.array(res['polygon'], dtype=np.float32)
                        if current_poly.ndim == 2 and current_poly.shape[1] == 2:
                             poly_s = ";".join([f"{int(p[0])},{int(p[1])}" for p in current_poly])
                        else: poly_s = "Malformed_Poly"
                    except Exception: poly_s = "Err_Parse_Poly"
                
                conf_s = f"{res['conf']:.4f}" if 'conf' in res and res['conf'] is not None else "N/A"
                text_s = res.get('text', "N/A_Text")
                x_key_s = res.get('orig_x', "N/A_X")
                simple_o_s = res.get('simple_orientation', "N/A") # Get simple orientation action
                f.write(f"{x_key_s}|{simple_o_s}|[{poly_s}]|{text_s}|{conf_s}\n")
        print(f"Final text results saved to: {txt_filepath}")
    print("OCR pipeline finished.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Jawi OCR: Custom Global Orient + CRAFT + Simple Per-Crop Orient + Parseq + R-L Sort + Conditional 180 Re-run')
    parser.add_argument('--image_path', required=True, type=str)
    parser.add_argument('--craft_model_path', required=True, type=str)
    parser.add_argument('--parseq_model_path', required=True, type=str)
    parser.add_argument('--custom_orientation_model_path', type=str, default=None)
    parser.add_argument('--output_dir', default='./jawi_ocr_results/', type=str)
    parser.add_argument('--orientation_class_names', type=str, default='0_degrees,180_degrees,270_degrees,90_degrees')
    parser.add_argument('--orientation_img_size', type=int, default=224)
    parser.add_argument('--orientation_confidence_threshold', type=float, default=75.0)
    parser.add_argument('--text_threshold', default=0.7,type=float)
    parser.add_argument('--low_text', default=0.4,type=float)
    parser.add_argument('--link_threshold', default=0.4,type=float)
    parser.add_argument('--canvas_size', default=1280,type=int)
    parser.add_argument('--mag_ratio', default=1.5,type=float)
    parser.add_argument('--poly',default=False,action='store_true')
    parser.add_argument('--use_simple_orientation', action='store_true')
    parser.add_argument('--rerun_180_threshold', type=float, default=90.0)
    parser.add_argument('--save_debug_crops', action='store_true')
    parser.add_argument('--no_cuda', action='store_true')
    
    args = parser.parse_args()
    main_ocr_pipeline(args)
