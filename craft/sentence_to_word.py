import os
import csv
import cv2
import numpy as np
import argparse
import torch
import torch.backends.cudnn as cudnn
from collections import OrderedDict
from typing import List, Tuple

# --- [Your existing CRAFT code imports and functions here] ---
# (Copy-paste all the utility, detection, and inference functions from your demo.py above)
# (e.g., copyStateDict, normalizeMeanVariance, resize_aspect_ratio, getDetBoxes, perform_inference, etc.)

# Attempt to import CRAFT model from EasyOCR's structure
try:
    from model.craft import CRAFT # Ensure this import works in your environment
except ImportError:
    print("--------------------------------------------------------------------------------")
    print("Error: Could not import 'CRAFT' from 'model.craft'.")
    print("Please ensure that EasyOCR is correctly installed and that this script")
    print("is run from a context where 'model.craft' is accessible.")
    print("This typically means running from within the 'EasyOCR/trainer/craft/' directory")
    print("or having the EasyOCR project path in your PYTHONPATH.")
    print("--------------------------------------------------------------------------------")
    raise

# --- Utility functions (adapted from EasyOCR/trainer/craft/utils) ---

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
    proc = cv2.resize(img, (target_w, target_h), interpolation = interpolation)
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
    ret, text_score = cv2.threshold(textmap, low_text, 1, 0)
    ret, link_score = cv2.threshold(linkmap, link_threshold, 1, 0)
    text_score_comb = np.clip(text_score + link_score, 0, 1)
    nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(text_score_comb.astype(np.uint8), connectivity=4)
    det = []
    mapper = []
    for k in range(1,nLabels):
        size = stats[k, cv2.CC_STAT_AREA]
        if size < 10: continue
        if np.max(textmap[labels==k]) < text_threshold: continue
        segmap = np.zeros(textmap.shape, dtype=np.uint8)
        segmap[labels==k] = 255
        segmap[np.logical_and(link_score==1, text_score==0)] = 0
        x, y = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP]
        w, h = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
        niter = int(np.sqrt(size * min(w,h) / (w*h)) * 2)
        sx, ex, sy, ey = x - niter, x + w + niter + 1, y - niter, y + h + niter + 1
        sx = max(0, sx); sy = max(0, sy)
        ex = min(img_w, ex); ey = min(img_h, ey)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(1+niter, 1+niter))
        segmap[sy:ey, sx:ex] = cv2.dilate(segmap[sy:ey, sx:ex], kernel)
        np_contours = np.roll(np.array(np.where(segmap!=0)),1,axis=0).transpose().reshape(-1,2)
        rectangle = cv2.minAreaRect(np_contours)
        box = cv2.boxPoints(rectangle)
        box_w, box_h = np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2])
        box_ratio = max(box_w,box_h) / (min(box_w,box_h) + 1e-5)
        if abs(1 - box_ratio) <= 0.1:
            l, r = min(np_contours[:,0]), max(np_contours[:,0])
            t, b = min(np_contours[:,1]), max(np_contours[:,1])
            box = np.array([[l, t], [r, t], [r, b], [l, b]], dtype=np.float32)
        startidx = box.sum(axis=1).argmin()
        box = np.roll(box, 4-startidx, 0)
        det.append(box)
        mapper.append(k)
    return det, labels, mapper


def getPoly_core(boxes, labels, mapper, linkmap):
    # configs
    num_cp = 5
    max_len_ratio = 0.7
    # expand_ratio = 1.45 # This was in EasyOCR's original but not directly used in the provided snippet's logic flow
    max_r = 2.0
    step_r = 0.2

    polys = []
    for k, box in enumerate(boxes):
        # size filter for small instance
        w = int(np.linalg.norm(box[0] - box[1]) + 0.5)
        h = int(np.linalg.norm(box[1] - box[2]) + 0.5)
        if w < 10 or h < 10:
            polys.append(None)
            continue

        # warp image
        tar = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        M = cv2.getPerspectiveTransform(box, tar)
        word_label = cv2.warpPerspective(labels, M, (w, h), flags=cv2.INTER_NEAREST)
        try:
            Minv = np.linalg.inv(M)
        except np.linalg.LinAlgError: # Catch singular matrix error
            polys.append(None)
            continue

        # binarization for selected label
        cur_label = mapper[k]
        word_label[word_label != cur_label] = 0
        word_label[word_label > 0] = 1

        """ Polygon generation """
        # Initialize lists to store contour points
        cp_top_list = []
        cp_bot_list = []
        cp_checked = False # Flag to ensure at least one point was found

        for i in range(num_cp):
            # Calculate current column, ensure it's within word_label bounds
            curr = int(w / (num_cp - 1) * i) if num_cp > 1 else 0
            if curr == w and w > 0: curr -= 1 # Adjust if at the very edge

            if not (0 <= curr < word_label.shape[1]): # Check if curr is a valid column index
                continue

            top_cnt_pts = np.where(word_label[:, curr] != 0)[0]
            
            if not top_cnt_pts.size > 0: # If no points found in this column
                continue
            else:
                cp_checked = True
            
            # Add the top-most and bottom-most points for this column
            cp_top_list.append(np.array([curr, top_cnt_pts[0]], dtype=np.int32))
            cp_bot_list.append(np.array([curr, top_cnt_pts[-1]], dtype=np.int32))
        
        # If not enough contour points were found (e.g., less than num_cp as in EasyOCR's logic)
        if not cp_checked or len(cp_top_list) < num_cp or len(cp_bot_list) < num_cp:
            # This condition implies that if points from all 'num_cp' slices aren't found,
            # the polygon isn't considered valid. Adjust if fewer points are acceptable.
            polys.append(None)
            continue
        
        # Convert lists of points to NumPy arrays
        cp_top = np.array(cp_top_list).reshape(-1, 2)
        cp_bot = np.array(cp_bot_list).reshape(-1, 2)

        # Now, cp_top and cp_bot are guaranteed to be NumPy arrays if this point is reached.
        # Proceed with polygon refinement
        final_poly = None # To store the polygon after checking link conditions

        for r_val in np.arange(0.5, max_r, step_r): # Loop for link checks (from EasyOCR)
            top_link_pts = 0
            if cp_top.shape[0] >= 2: # Need at least 2 points to form segments
                for idx_pt in range(cp_top.shape[0] - 1):
                    pt_1, pt_2 = cp_top[idx_pt], cp_top[idx_pt + 1]
                    mid_pt_x, mid_pt_y = int((pt_1[0] + pt_2[0]) / 2), int((pt_1[1] + pt_2[1]) / 2)
                    # Boundary check for mid_pt accessing linkmap
                    if 0 <= mid_pt_y < linkmap.shape[0] and 0 <= mid_pt_x < linkmap.shape[1]:
                        if linkmap[mid_pt_y, mid_pt_x] == 1:
                            top_link_pts += 1
            
            bot_link_pts = 0
            if cp_bot.shape[0] >= 2:
                for idx_pt in range(cp_bot.shape[0] - 1):
                    pt_1, pt_2 = cp_bot[idx_pt], cp_bot[idx_pt + 1]
                    mid_pt_x, mid_pt_y = int((pt_1[0] + pt_2[0]) / 2), int((pt_1[1] + pt_2[1]) / 2)
                    if 0 <= mid_pt_y < linkmap.shape[0] and 0 <= mid_pt_x < linkmap.shape[1]:
                        if linkmap[mid_pt_y, mid_pt_x] == 1:
                            bot_link_pts += 1

            # If too many points on the contour are part of link areas
            if top_link_pts > max_len_ratio * cp_top.shape[0] or \
               bot_link_pts > max_len_ratio * cp_bot.shape[0]:
                final_poly = None 
                break # Discard this polygon candidate and break from r_val refinement loop

            # Ideal case: no link points on the contour
            if top_link_pts == 0 and bot_link_pts == 0:
                poly_coords = np.concatenate((cp_top, np.flip(cp_bot, axis=0)), axis=0)
                final_poly = cv2.perspectiveTransform(np.array([poly_coords], dtype=np.float32), Minv)[0]
                break # Found a good polygon, break from r_val refinement loop
            
            # If the loop finishes without breaking, final_poly's last state is used.
            # (EasyOCR has more complex patch-based refinement here, omitted for now if not the root cause)

        polys.append(final_poly) # Append the polygon (or None if not found/valid)
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

def draw_boxes_on_image(image, boxes_or_polys, color=(0, 255, 0), thickness=2):
    img_copy = image.copy()
    for item in boxes_or_polys:
        if item is None: continue
        poly = np.array(item).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(img_copy, [poly], isClosed=True, color=color, thickness=thickness)
    return img_copy

# --- Main Inference Function ---
def perform_inference(net, image_path, text_threshold, link_threshold, low_text, cuda, poly,
                      canvas_size=1280, mag_ratio=1.5, show_time=False, 
                      output_dir=None, viz=False):
    if show_time: start_time = time.time()
    try:
        image = cv2.imread(image_path)
        if image is None: raise IOError(f"Could not read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except Exception as e: print(f"Error loading image {image_path}: {e}"); return None, None, None
    img_resized, target_ratio, size_heatmap = resize_aspect_ratio(image, canvas_size, cv2.INTER_LINEAR, mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio
    x = normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)
    if cuda: x = x.cuda()
    with torch.no_grad(): y, feature = net(x)
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()
    if show_time: print(f"Preprocessing & Inference: {time.time() - start_time:.4f}s"); proc_time_start = time.time()
    boxes, polys = getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)
    boxes = adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None and k < len(boxes): polys[k] = boxes[k] # Fallback to box if poly is None
    if show_time: print(f"Post-processing: {time.time() - proc_time_start:.4f}s")
    if viz and output_dir:
        filename, _ = os.path.splitext(os.path.basename(image_path))
        mask_file = os.path.join(output_dir, filename + '_mask.jpg')
        heatmap_viz = np.hstack((score_text, score_link))
        heatmap_viz = cv2.cvtColor(np.uint8(np.clip(heatmap_viz,0,1)*255), cv2.COLOR_GRAY2BGR)
        cv2.imwrite(mask_file, heatmap_viz)
    return boxes, polys, image

def load_labels(labels_path: str) -> dict:
    """Load labels from CSV into a dictionary: {img_name: text}"""
    labels = {}
    with open(labels_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_name = os.path.splitext(row['file'])[0]  # Remove .jpg if present
            labels[img_name] = row['text']
    return labels

def split_into_words(text: str) -> List[str]:
    """Split text into words (adjust delimiter as needed for your language)"""
    # For your example, split by space (assuming your language uses space as word separator)
    return text.split()

def extract_word_region(image: np.ndarray, box_or_poly, padding=5) -> np.ndarray:
    """Extract region from image based on detected box or polygon, with optional padding."""
    if isinstance(box_or_poly, list):
        box_or_poly = np.array(box_or_poly)
    if box_or_poly is None:
        return None
    x, y, w, h = cv2.boundingRect(box_or_poly.astype(np.int32))
    # Apply padding
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(image.shape[1] - x, w + 2*padding)
    h = min(image.shape[0] - y, h + 2*padding)
    return image[y:y+h, x:x+w]

def main(args):
    # --- Load labels ---
    labels = load_labels(args.labels_path)
    # --- Create output directories ---
    os.makedirs(args.output_images_dir, exist_ok=True)
    # --- Prepare new labels file ---
    new_labels_path = os.path.join(args.output_dir, 'labels.csv')
    with open(new_labels_path, 'w', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['file', 'text'])

    # --- Load CRAFT model (as in your demo.py) ---
    net = CRAFT()
    checkpoint = torch.load(args.trained_model, map_location='cuda' if torch.cuda.is_available() else 'cpu')
    if 'craft' in checkpoint:
        model_state_dict = checkpoint['craft']
    elif 'model' in checkpoint:
        model_state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        model_state_dict = checkpoint['state_dict']
    else:
        raise KeyError(f"Could not find model weights in checkpoint. Expected keys like 'craft', 'model', or 'state_dict'. Found: {list(checkpoint.keys())}")
    net.load_state_dict(copyStateDict(model_state_dict))
    if torch.cuda.is_available():
        net = net.cuda()
    net.eval()

    # --- Process each image ---
    for img_name, text in labels.items():
        img_path = os.path.join(args.images_dir, f"{img_name}.jpg")
        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            continue

        # --- Detect text regions ---
        detected_boxes, detected_polys, original_image = perform_inference(
            net, img_path, args.text_threshold, args.link_threshold, args.low_text,
            torch.cuda.is_available(), args.poly, args.canvas_size, args.mag_ratio
        )
        if detected_polys is None or original_image is None:
            print(f"No text detected in {img_name}")
            continue

        # --- Split sentence into words ---
        words = split_into_words(text)
        if len(words) == 0:
            continue

        # --- For each word, extract and save region ---
        # NOTE: This assumes detected_polys are in left-to-right order and match the words!
        # If your script does not guarantee this, you need a way to match detected regions to words.
        # For now, this is a basic implementation.
        # (In practice, you may need a more advanced matching algorithm.)
        if len(detected_polys) != len(words):
            print(f"Warning: Number of detected regions ({len(detected_polys)}) does not match number of words ({len(words)}) in {img_name}")
            continue

        for i, (word, poly) in enumerate(zip(words, detected_polys)):
            if poly is None:
                if i < len(detected_boxes) and detected_boxes[i] is not None:
                    poly = detected_boxes[i]
                else:
                    print(f"No region found for word {i} in {img_name}")
                    continue

            # --- Extract and save word region ---
            word_region = extract_word_region(original_image, poly)
            if word_region is None or word_region.size == 0:
                print(f"Could not extract region for word {i} in {img_name}")
                continue

            # --- Save new image ---
            new_img_name = f"{img_name}_word{i+1}.jpg"
            new_img_path = os.path.join(args.output_images_dir, new_img_name)
            cv2.imwrite(new_img_path, cv2.cvtColor(word_region, cv2.COLOR_RGB2BGR))

            # --- Append to new labels file ---
            with open(new_labels_path, 'a', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([new_img_name, word])

    print(f"Dataset generation complete. New images saved to {args.output_images_dir}")
    print(f"New labels saved to {new_labels_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate word-level dataset from sentence-level images using CRAFT text detection.')
    parser.add_argument('--images_dir', required=True, help='Directory containing original images (e.g., dataset/images/)')
    parser.add_argument('--labels_path', required=True, help='Path to original labels CSV (e.g., dataset/labels.csv)')
    parser.add_argument('--output_dir', required=True, help='Directory to save new dataset and labels')
    parser.add_argument('--output_images_dir', default='images', help='Subdirectory in output_dir for new images')
    parser.add_argument('--trained_model', required=True, help='Path to CRAFT model checkpoint (.pth)')
    parser.add_argument('--text_threshold', type=float, default=0.7, help='Text confidence threshold')
    parser.add_argument('--low_text', type=float, default=0.4, help='Text low_text threshold')
    parser.add_argument('--link_threshold', type=float, default=0.4, help='Link confidence threshold')
    parser.add_argument('--canvas_size', type=int, default=1280, help='Image size for inference')
    parser.add_argument('--mag_ratio', type=float, default=1.5, help='Image magnification ratio')
    parser.add_argument('--poly', action='store_true', help='Enable polygon type detection')
    args = parser.parse_args()

    # Ensure output_images_dir is within output_dir
    args.output_images_dir = os.path.join(args.output_dir, args.output_images_dir)
    os.makedirs(args.output_images_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    main(args)
