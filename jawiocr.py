import os
import sys
import argparse
import time
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from collections import OrderedDict
from PIL import Image as PILImage # Use a different alias for PIL
from torchvision import transforms as TorchTransforms # Alias for torchvision transforms
import pytesseract # Import pytesseract

# --- Path Setup ---
# This script (jawiocr.py) is in the jawiOCR/ directory.
# craft/ and parseq_jawi/ are subdirectories.
current_script_dir = os.path.dirname(os.path.abspath(__file__))
craft_module_dir = os.path.join(current_script_dir, 'craft')
parseq_module_dir = os.path.join(current_script_dir, 'parseq_jawi')

if craft_module_dir not in sys.path:
    sys.path.insert(0, craft_module_dir)
if parseq_module_dir not in sys.path:
    sys.path.insert(0, parseq_module_dir)

print(f"Adjusted sys.path. Current sys.path[0:3]: {sys.path[0:3]}")


# --- CRAFT Model Import and Utilities ---
try:
    # Assuming craft_module_dir/model/craft.py exists
    from model.craft import CRAFT
except ImportError as e:
    print(f"Error importing 'CRAFT' from 'model.craft': {e}")
    print(f"Please ensure that '{os.path.join(craft_module_dir, 'model', 'craft.py')}' exists and is accessible.")
    print(f"Attempted to look in: {craft_module_dir}")
    sys.exit(1)

def copyStateDict(state_dict):
    """
    Copies a state dictionary, removing 'module.' prefix from keys
    if it exists (common when models are saved using nn.DataParallel).
    """
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("module."):
            name = k[7:]  # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    return new_state_dict

def normalizeMeanVariance(in_img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
    # should be RGB order
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
    if target_h % 32 != 0:
        target_h32 = target_h + (32 - target_h % 32)
    if target_w % 32 != 0:
        target_w32 = target_w + (32 - target_w % 32)
    resized = np.zeros((target_h32, target_w32, channel), dtype=np.uint8)
    resized[0:target_h, 0:target_w, :] = proc
    return resized, ratio, (int(target_w32/2), int(target_h32/2)) # size_heatmap

def getDetBoxes_core(textmap, linkmap, text_threshold, link_threshold, low_text):
    linkmap = linkmap.copy()
    textmap = textmap.copy()
    img_h, img_w = textmap.shape
    _, text_score = cv2.threshold(textmap, low_text, 1, 0)
    _, link_score = cv2.threshold(linkmap, link_threshold, 1, 0)
    text_score_comb = np.clip(text_score + link_score, 0, 1)
    nLabels, labels, stats, _ = cv2.connectedComponentsWithStats(text_score_comb.astype(np.uint8), connectivity=4)
    det = []
    mapper = []
    for k in range(1, nLabels):
        size = stats[k, cv2.CC_STAT_AREA]
        if size < 10 or np.max(textmap[labels==k]) < text_threshold:
            continue
        segmap = np.zeros(textmap.shape, dtype=np.uint8)
        segmap[labels==k] = 255
        segmap[np.logical_and(link_score==1, text_score==0)] = 0 # remove link area
        x, y = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP]
        w, h = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
        niter = int(np.sqrt(size * min(w,h) / (w*h)) * 2) if w*h > 0 else 0 # Avoid division by zero
        sx, ex, sy, ey = max(0, x-niter), min(img_w, x+w+niter+1), max(0, y-niter), min(img_h, y+h+niter+1)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(1+niter, 1+niter))
        segmap[sy:ey, sx:ex] = cv2.dilate(segmap[sy:ey, sx:ex], kernel)
        np_contours = np.roll(np.array(np.where(segmap!=0)),1,axis=0).transpose().reshape(-1,2)
        if np_contours.size == 0: # No points found after dilation
            continue
        rectangle = cv2.minAreaRect(np_contours)
        box_pts = cv2.boxPoints(rectangle)
        box_w, box_h = np.linalg.norm(box_pts[0] - box_pts[1]), np.linalg.norm(box_pts[1] - box_pts[2])
        if abs(1 - max(box_w,box_h)/(min(box_w,box_h)+1e-5)) <= 0.1: # Check if diamond shape
            l, r = min(np_contours[:,0]), max(np_contours[:,0])
            t, b = min(np_contours[:,1]), max(np_contours[:,1])
            box_pts = np.array([[l, t], [r, t], [r, b], [l, b]], dtype=np.float32)
        startidx = box_pts.sum(axis=1).argmin() # Clockwise order
        box_pts = np.roll(box_pts, 4-startidx, 0)
        det.append(box_pts)
        mapper.append(k)
    return det, labels, mapper

def getPoly_core(boxes, labels, mapper, linkmap):
    num_cp, max_len_ratio, max_r, step_r = 5, 0.7, 2.0, 0.2
    polys = []
    for k, box_pts_param in enumerate(boxes): # Renamed box to box_pts_param
        w_box, h_box = int(np.linalg.norm(box_pts_param[0]-box_pts_param[1])+0.5), int(np.linalg.norm(box_pts_param[1]-box_pts_param[2])+0.5)
        if w_box < 10 or h_box < 10:
            polys.append(None)
            continue
        tar = np.float32([[0,0],[w_box,0],[w_box,h_box],[0,h_box]])
        M = cv2.getPerspectiveTransform(box_pts_param, tar)
        word_label = cv2.warpPerspective(labels, M, (w_box,h_box), flags=cv2.INTER_NEAREST)
        try:
            Minv = np.linalg.inv(M)
        except np.linalg.LinAlgError:
            polys.append(None)
            continue
        word_label[word_label != mapper[k]] = 0
        word_label[word_label > 0] = 1
        cp_top_list, cp_bot_list, cp_checked = [], [], False
        for i in range(num_cp):
            curr = int(w_box/(num_cp-1)*i) if num_cp > 1 else 0
            if curr == w_box and w_box > 0: curr -=1
            if not (0 <= curr < word_label.shape[1]):
                continue
            pts = np.where(word_label[:,curr]!=0)[0]
            if not pts.size > 0:
                continue
            cp_checked = True
            cp_top_list.append(np.array([curr,pts[0]], dtype=np.int32))
            cp_bot_list.append(np.array([curr,pts[-1]], dtype=np.int32))
        if not cp_checked or len(cp_top_list) < num_cp or len(cp_bot_list) < num_cp :
            polys.append(None)
            continue
        cp_top, cp_bot = np.array(cp_top_list).reshape(-1,2), np.array(cp_bot_list).reshape(-1,2)
        final_poly = None
        for r_val in np.arange(0.5, max_r, step_r):
            top_link_pts, bot_link_pts = 0,0
            if cp_top.shape[0]>=2: top_link_pts = sum(1 for i_pt in range(cp_top.shape[0]-1) if 0<=int((cp_top[i_pt][1]+cp_top[i_pt+1][1])/2)<linkmap.shape[0] and 0<=int((cp_top[i_pt][0]+cp_top[i_pt+1][0])/2)<linkmap.shape[1] and linkmap[int((cp_top[i_pt][1]+cp_top[i_pt+1][1])/2), int((cp_top[i_pt][0]+cp_top[i_pt+1][0])/2)]==1)
            if cp_bot.shape[0]>=2: bot_link_pts = sum(1 for i_pt in range(cp_bot.shape[0]-1) if 0<=int((cp_bot[i_pt][1]+cp_bot[i_pt+1][1])/2)<linkmap.shape[0] and 0<=int((cp_bot[i_pt][0]+cp_bot[i_pt+1][0])/2)<linkmap.shape[1] and linkmap[int((cp_bot[i_pt][1]+cp_bot[i_pt+1][1])/2), int((cp_bot[i_pt][0]+cp_bot[i_pt+1][0])/2)]==1)
            if top_link_pts > max_len_ratio*cp_top.shape[0] or bot_link_pts > max_len_ratio*cp_bot.shape[0]:
                final_poly=None
                break
            if top_link_pts==0 and bot_link_pts==0:
                poly_coords = np.concatenate((cp_top, np.flip(cp_bot,axis=0)), axis=0)
                final_poly = cv2.perspectiveTransform(np.array([poly_coords],dtype=np.float32),Minv)[0]
                break
        polys.append(final_poly)
    return polys

def getDetBoxes(textmap, linkmap, text_threshold, link_threshold, low_text, poly=False):
    boxes, labels, mapper = getDetBoxes_core(textmap, linkmap, text_threshold, link_threshold, low_text)
    if poly and boxes:
        polys = getPoly_core(boxes, labels, mapper, linkmap)
    else:
        polys = [None] * len(boxes) # Ensure polys has same length as boxes
    return boxes, polys

def adjustResultCoordinates(coords, ratio_w, ratio_h, ratio_net=2):
    if coords is None or len(coords) == 0: return []
    adjusted_coords = []
    for item in coords:
        if item is not None:
            adjusted_coords.append(item * (ratio_w * ratio_net, ratio_h * ratio_net))
        else:
            adjusted_coords.append(None)
    return adjusted_coords

def perform_craft_inference(net, image_bgr, text_threshold, link_threshold, low_text, cuda, poly,
                            canvas_size=1280, mag_ratio=1.5):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img_resized, target_ratio, _ = resize_aspect_ratio(image_rgb, canvas_size, cv2.INTER_LINEAR, mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio
    x = normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2,0,1).unsqueeze(0)
    if cuda: x = x.cuda()
    with torch.no_grad(): y, _ = net(x) # Use underscore for feature if not used
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()
    boxes, polys = getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)
    boxes = adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = adjustResultCoordinates(polys, ratio_w, ratio_h)
    # Fallback to box if poly is None
    final_polys = []
    for k in range(len(polys)): # Or max(len(polys), len(boxes)) if they can differ
        poly_item = polys[k] if k < len(polys) else None
        box_item = boxes[k] if k < len(boxes) else None
        if poly_item is not None:
            final_polys.append(poly_item)
        elif box_item is not None:
            final_polys.append(box_item)
        else:
            final_polys.append(None) # Should not happen if boxes/polys are aligned
    final_polys = [p for p in final_polys if p is not None] # Clean out any None that might have slipped in
    return final_polys


# --- Parseq Model Import and Utilities ---
try:
    from strhub.models.utils import load_from_checkpoint as parseq_load_from_checkpoint
    from strhub.data.module import SceneTextDataModule
    # from strhub.data.tokenizer import JawiTokenizer # Not strictly needed if model.tokenizer is complete
except ImportError as e:
    print(f"Error importing Parseq components: {e}\nLooked in: {parseq_module_dir}")
    sys.exit(1)

def load_parseq_model_strhub(checkpoint_path, device):
    print(f"Loading Parseq model using STRHub method from: {checkpoint_path}")
    try:
        kwargs = {} # Add if your model loading needs specific kwargs like charset_test
        model = parseq_load_from_checkpoint(checkpoint_path, **kwargs).eval().to(device)
        print("Parseq model loaded successfully via STRHub.")
        img_transform = SceneTextDataModule.get_transform(model.hparams.img_size)
        if not hasattr(model, 'tokenizer') or model.tokenizer is None:
            print("CRITICAL Error: Parseq model loaded via STRHub does not have a 'tokenizer'.")
            sys.exit(1)
        else:
            print(f"Parseq model tokenizer found. Type: {type(model.tokenizer)}")
        return model, img_transform
    except Exception as e:
        print(f"Error loading Parseq model via STRHub: {e}")
        sys.exit(1)

def preprocess_for_parseq_strhub(img_crop_bgr, parseq_transform, device):
    """Prepares a single image crop for Parseq model using STRHub's transform."""
    if img_crop_bgr is None or img_crop_bgr.shape[0] == 0 or img_crop_bgr.shape[1] == 0:
        return None
    img_rgb_pil = PILImage.fromarray(cv2.cvtColor(img_crop_bgr, cv2.COLOR_BGR2RGB))
    img_tensor = parseq_transform(img_rgb_pil).unsqueeze(0) # Add batch dimension
    return img_tensor.to(device)


# --- Orientation Correction Components using Tesseract ---
def rotate_image_cv(image_cv, angle_degrees):
    """Rotates an OpenCV image by a multiple of 90 degrees."""
    if angle_degrees == 0:
        return image_cv
    elif angle_degrees == 90:
        return cv2.rotate(image_cv, cv2.ROTATE_90_CLOCKWISE)
    elif angle_degrees == 180:
        return cv2.rotate(image_cv, cv2.ROTATE_180)
    elif angle_degrees == 270: # Tesseract OSD outputs 0, 90, 180, 270 for orientation
        return cv2.rotate(image_cv, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        # print(f"Warning: Unsupported rotation angle {angle_degrees}. Returning original.")
        return image_cv

def correct_orientation_tesseract(image_crop_bgr, tesseract_lang='ara', min_height=20, min_width=20): # Added min_height/width
    """Corrects the orientation of a text crop using Tesseract OSD."""
    if image_crop_bgr is None or image_crop_bgr.shape[0] == 0 or image_crop_bgr.shape[1] == 0:
        return image_crop_bgr

    h, w = image_crop_bgr.shape[:2]
    if h < min_height or w < min_width:
        # print(f"Skipping Tesseract OSD for small crop (h:{h}, w:{w}). Assuming 0 deg orientation.")
        return image_crop_bgr # Return original, assuming it's correctly oriented or too small to tell
    UPSCALE_THRESHOLD_H = 50 # Example: if height is less than this, consider upscaling
    UPSCALE_THRESHOLD_W = 50 # Example
    UPSCALE_FACTOR = 1.5 # Example: upscale by 50%

    temp_crop_for_osd = image_crop_bgr.copy()
    if h < UPSCALE_THRESHOLD_H or w < UPSCALE_THRESHOLD_W:
    # print(f"Small crop (h:{h}, w:{w}). Upscaling by {UPSCALE_FACTOR}x for OSD attempt.")
        new_w = int(w * UPSCALE_FACTOR)
        new_h = int(h * UPSCALE_FACTOR)
    if new_w > 0 and new_h > 0:
        temp_crop_for_osd = cv2.resize(image_crop_bgr, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    try:
        pil_img = PILImage.fromarray(cv2.cvtColor(temp_crop_for_osd, cv2.COLOR_BGR2RGB))
        osd_data = pytesseract.image_to_osd(pil_img, lang=tesseract_lang, config='--psm 0')
        
        detected_orientation = 0 
        for line in osd_data.split('\n'):
            if 'Orientation in degrees:' in line:
                try:
                    detected_orientation = int(line.split(':')[1].strip())
                    break
                except ValueError:
                    print(f"Warning: Could not parse orientation value from OSD line: {line}")
                    detected_orientation = 0 # Fallback
                    break 
            elif 'Rotate:' in line: # Fallback
                try:
                    detected_orientation = int(line.split(':')[1].strip())
                    break
                except ValueError:
                    detected_orientation = 0
                    break
        
        effective_rotation = 0
        if detected_orientation == 90: effective_rotation = 270
        elif detected_orientation == 180: effective_rotation = 180
        elif detected_orientation == 270: effective_rotation = 90
        
        if effective_rotation != 0:
            # print(f"Tesseract OSD orientation: {detected_orientation} deg. Rotating by {effective_rotation} deg.")
            corrected_image = rotate_image_cv(image_crop_bgr, effective_rotation)
            return corrected_image
        else:
            return image_crop_bgr
    except Exception as e:
        print(f"Error during Tesseract OSD for orientation correction: {e}")
        return image_crop_bgr


# --- Cropping Utility ---
def get_cropped_image_from_poly(image_bgr, poly_pts):
    """Crops an image region defined by a polygon (4 points) using perspective transform."""
    if poly_pts is None or len(poly_pts) < 3: return None
    poly = np.array(poly_pts, dtype=np.float32)
    
    # Get the minimum area rectangle for the polygon
    rect = cv2.minAreaRect(poly) # ((center_x, center_y), (width, height), angle)
    
    # Get the 4 corner points of the rotated rectangle
    box = cv2.boxPoints(rect) # Note: order might not be TL, TR, BR, BL
    
    # Get width and height from the rotated rectangle
    # rect[1][0] is width, rect[1][1] is height as per OpenCV docs for minAreaRect
    # Angle also determines which is width and which is height visually
    width = int(rect[1][0])
    height = int(rect[1][1])
    angle = rect[2] # Angle is in [-90, 0)

    # Handle cases where width and height might be swapped by minAreaRect
    # if angle is close to -90 (e.g. text is vertical), width and height from rect[1] might be swapped
    # compared to visual expectation if we want "text width" and "text height".
    # For warping, it's often better to ensure maxWidth corresponds to the longest side of the text.
    
    # For simplicity in warping, we want the "text line" to be horizontal in the crop.
    # If angle indicates the box is more vertical than horizontal, swap width and height for destination.
    if angle < -45: # Typically means height > width from rect[1] for horizontal text
        actual_width, actual_height = height, width
    else:
        actual_width, actual_height = width, height

    if actual_width == 0 or actual_height == 0: return None

    # Destination points for a straightened rectangle
    dst_pts = np.array([
        [0, actual_height - 1],
        [0, 0],
        [actual_width - 1, 0],
        [actual_width - 1, actual_height - 1]
    ], dtype="float32")

    # Source points (the detected polygon corners)
    # We need to order src_pts (box) to match dst_pts: BL, TL, TR, BR
    # Sum (x+y) is min for TL, max for BR. Diff (y-x) is min for TR, max for BL (for image coords)
    s = box.sum(axis=1)
    diff = np.diff(box, axis=1).reshape(-1) # y-x for each point relative to the next, needs care
    
    # A simpler way to order for perspective transform:
    # Identify TL, TR, BL, BR from the `box` points (which are from minAreaRect)
    # This part can be tricky. minAreaRect does not guarantee order.
    # We can use the provided poly_pts if its order is consistent (e.g. TL, TR, BR, BL)
    # If poly_pts from CRAFT is already in a consistent order:
    src_pts = poly.astype("float32")
    # Otherwise, re-order `box` (corners from minAreaRect)
    # Example reordering (assuming image coordinates where y increases downwards):
    # tl = box[np.argmin(s)]
    # br = box[np.argmax(s)]
    # temp_pts = np.array([p for p in box if not np.all(p == tl) and not np.all(p == br)])
    # if temp_pts[0][0] > temp_pts[1][0]: # if x of first is greater, it's TR
    #     tr = temp_pts[0]; bl = temp_pts[1]
    # else:
    #     tr = temp_pts[1]; bl = temp_pts[0]
    # src_pts = np.array([bl, tl, tr, br], dtype="float32") # Match dst_pts order for warpPerspective

    # The perspective transformation matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    # Directly warp the rotated rectangle to get the straightened rectangle
    warped_crop = cv2.warpPerspective(image_bgr, M, (actual_width, actual_height))
    
    return warped_crop


# --- Main OCR Pipeline ---
def main_ocr_pipeline(args):
    cuda_enabled = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if cuda_enabled else 'cpu')
    print(f"Using device: {device}")

    # 1. Load CRAFT model
    craft_net = CRAFT()
    print(f'Loading CRAFT weights from: {args.craft_model_path}')
    # Consider weights_only=True if loading untrusted checkpoints
    checkpoint_craft = torch.load(args.craft_model_path, map_location=device, weights_only=False) 
    if 'craft' in checkpoint_craft: model_state_dict_craft = checkpoint_craft['craft']
    elif 'model' in checkpoint_craft: model_state_dict_craft = checkpoint_craft['model']
    elif 'state_dict' in checkpoint_craft: model_state_dict_craft = checkpoint_craft['state_dict']
    else: model_state_dict_craft = checkpoint_craft # Assume it's directly the state_dict
    craft_net.load_state_dict(copyStateDict(model_state_dict_craft))
    craft_net.to(device); craft_net.eval()

    # 2. Load Parseq model
    parseq_model, parseq_img_transform = load_parseq_model_strhub(args.parseq_model_path, device)
    
    # 3. Load image
    print(f"Processing image: {args.image_path}")
    image_bgr_original = cv2.imread(args.image_path)
    if image_bgr_original is None:
        print(f"Error: Could not read image at {args.image_path}")
        return

    # 4. Perform Text Detection (CRAFT)
    detected_polys_from_craft = perform_craft_inference(
        craft_net, image_bgr_original, args.text_threshold, args.link_threshold,
        args.low_text, cuda_enabled, args.poly, args.canvas_size, args.mag_ratio
    )
    print(f"CRAFT detected {len(detected_polys_from_craft)} potential text regions.")

    # 5. Process each detected region
    results = []
    output_image_viz = image_bgr_original.copy()

    for i, poly_pts in enumerate(detected_polys_from_craft):
        if poly_pts is None:
            continue

        cropped_bgr = get_cropped_image_from_poly(image_bgr_original, poly_pts)
        
        if cropped_bgr is None or cropped_bgr.shape[0] == 0 or cropped_bgr.shape[1] == 0:
            print(f"Warning: Skipping invalid crop for polygon {i+1}")
            continue

        # >>> STAGE: Orientation Correction (using Tesseract if enabled) <<<
        crop_for_parseq = cropped_bgr 
        if args.use_tesseract_orientation:
            corrected_bgr_crop = correct_orientation_tesseract(
                cropped_bgr, 
                tesseract_lang=args.tesseract_lang,
                min_height=args.tesseract_min_crop_height, # Add new args
                min_width=args.tesseract_min_crop_width   # Add new args
            )
            if corrected_bgr_crop is not None:
                crop_for_parseq = corrected_bgr_crop
            else:
                print(f"Warning: Tesseract orientation correction failed for region {i+1}, using original.")
        
        # >>> STAGE: Text Recognition (Parseq) <<<
        parseq_input_tensor = preprocess_for_parseq_strhub(crop_for_parseq, parseq_img_transform, device)
        if parseq_input_tensor is None:
            print(f"Warning: Skipping region {i+1} (Parseq preprocess failed).")
            continue

        with torch.no_grad():
            logits = parseq_model(parseq_input_tensor)
            probabilities = logits.softmax(-1)
            # model.tokenizer.decode(probabilities) returns (list_of_texts, list_of_confidences_tensors)
            pred_texts, pred_confs_tensors_list = parseq_model.tokenizer.decode(probabilities)
            
            if pred_texts and len(pred_texts) > 0:
                recognized_text = pred_texts[0] # For batch size 1
                # pred_confs_tensors_list[0] is the tensor of token confidences for the first item
                token_confidences_tensor = pred_confs_tensors_list[0] if pred_confs_tensors_list and len(pred_confs_tensors_list) > 0 else None
                
                sequence_confidence_float = None
                if token_confidences_tensor is not None and isinstance(token_confidences_tensor, torch.Tensor):
                    if token_confidences_tensor.numel() == 1: # Scalar tensor
                        sequence_confidence_float = token_confidences_tensor.item()
                    elif token_confidences_tensor.numel() > 1: # 1D tensor of token confidences
                        sequence_confidence_float = token_confidences_tensor.mean().item() # Use mean
                
                print_conf_str = f"{sequence_confidence_float:.4f}" if sequence_confidence_float is not None else "N/A"
                print(f"Region {i+1}: Text = '{recognized_text}', Conf = {print_conf_str}")
                results.append({'polygon': poly_pts, 'text': recognized_text, 'confidence': sequence_confidence_float})
                cv2.polylines(output_image_viz, [poly_pts.astype(np.int32)], True, (0,0,255), 2) # Draw original CRAFT box
            else:
                print(f"Warning: No text decoded from Parseq for region {i+1}")
    
    # 6. Save results
    if args.output_dir:
        if not os.path.exists(args.output_dir): os.makedirs(args.output_dir)
        base_filename = os.path.splitext(os.path.basename(args.image_path))[0]
        viz_filepath = os.path.join(args.output_dir, f"res_ocr_{base_filename}.jpg")
        cv2.imwrite(viz_filepath, output_image_viz)
        print(f"Visualized OCR result saved to: {viz_filepath}")
        text_result_filepath = os.path.join(args.output_dir, f"res_ocr_{base_filename}.txt")
        with open(text_result_filepath, 'w', encoding='utf-8') as f:
            for res in results:
                # Ensure polygon points are integers for saving if they came from CRAFT as float
                poly_str = ";".join([f"{int(p[0])},{int(p[1])}" for p_sublist in res['polygon'] for p in (p_sublist if isinstance(p_sublist, list) or isinstance(p_sublist, np.ndarray) else [p_sublist]) if isinstance(p, (list, np.ndarray)) and len(p) == 2])

                conf_str = f"{res['confidence']:.4f}" if res['confidence'] is not None else "N/A"
                f.write(f"Polygon: [{poly_str}] | Text: {res['text']} | Confidence: {conf_str}\n")
        print(f"Text OCR results saved to: {text_result_filepath}")
    print("OCR pipeline finished.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Jawi OCR Pipeline with Tesseract Orientation Correction')
    # Image and Model Paths
    parser.add_argument('--image_path', required=True, type=str, help='Path to the input image')
    parser.add_argument('--craft_model_path', required=True, type=str, help='Path to fine-tuned CRAFT model')
    parser.add_argument('--parseq_model_path', required=True, type=str, help='Path to fine-tuned Parseq model (STRHub format)')
    # Output
    parser.add_argument('--output_dir', default='./jawi_ocr_results/', type=str, help='Directory to save OCR results')
    # CRAFT Parameters
    parser.add_argument('--text_threshold', default=0.7, type=float, help='CRAFT text confidence threshold')
    parser.add_argument('--low_text', default=0.4, type=float, help='CRAFT text low_text threshold')
    parser.add_argument('--link_threshold', default=0.4, type=float, help='CRAFT link confidence threshold')
    parser.add_argument('--canvas_size', default=1280, type=int, help='CRAFT image size for inference')
    parser.add_argument('--mag_ratio', default=1.5, type=float, help='CRAFT image magnification ratio')
    parser.add_argument('--poly', default=False, action='store_true', help='CRAFT enable polygon type detection (outputs quadrilateral)')
    # Tesseract Orientation
    parser.add_argument('--use_tesseract_orientation', action='store_true', help='Enable orientation correction using Tesseract OSD')
    parser.add_argument('--tesseract_lang', type=str, default='ara', help='Language for Tesseract OSD (e.g., "ara", "ara+fas")')
    parser.add_argument('--tesseract_min_crop_height', type=int, default=25, help='Min crop height for attempting Tesseract OSD')
    parser.add_argument('--tesseract_min_crop_width', type=int, default=25, help='Min crop width for attempting Tesseract OSD')
    # General
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    
    args = parser.parse_args()
    main_ocr_pipeline(args)