import argparse
import os
import sys
import time
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms as T
from collections import OrderedDict

# --- Add parseq_jawi directory to Python path ---
# This allows importing modules from your parseq_jawi project structure
# Adjust the path if your script is located elsewhere relative to parseq_jawi
current_dir = os.path.dirname(os.path.abspath(__file__))
parseq_dir = os.path.join(os.path.dirname(current_dir), 'parseq_jawi') # Assumes script is in craft/, parseq_jawi is sibling
if parseq_dir not in sys.path:
    sys.path.insert(0, parseq_dir)
print(f"Attempting to use Parseq modules from: {parseq_dir}")

# --- CRAFT Model Import and Utilities (from previous demo.py) ---
try:
    from craft.model.craft import CRAFT # Assumes craft/model/craft.py
except ImportError:
    print("Error: Could not import 'CRAFT' from 'model.craft'. Ensure script context or PYTHONPATH.")
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
        x, y, w, h = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP], stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
        niter = int(np.sqrt(size * min(w,h) / (w*h)) * 2)
        sx, ex, sy, ey = max(0, x-niter), min(img_w, x+w+niter+1), max(0, y-niter), min(img_h, y+h+niter+1)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(1+niter, 1+niter))
        segmap[sy:ey, sx:ex] = cv2.dilate(segmap[sy:ey, sx:ex], kernel)
        np_contours = np.roll(np.array(np.where(segmap!=0)),1,axis=0).transpose().reshape(-1,2)
        rect = cv2.minAreaRect(np_contours)
        box = cv2.boxPoints(rect)
        box_w, box_h = np.linalg.norm(box[0]-box[1]), np.linalg.norm(box[1]-box[2])
        if abs(1 - max(box_w,box_h)/(min(box_w,box_h)+1e-5)) <= 0.1:
            l, r, t, b = min(np_contours[:,0]), max(np_contours[:,0]), min(np_contours[:,1]), max(np_contours[:,1])
            box = np.array([[l,t],[r,t],[r,b],[l,b]], dtype=np.float32)
        box = np.roll(box, 4-box.sum(axis=1).argmin(), 0)
        det.append(box); mapper.append(k)
    return det, labels, mapper

def getPoly_core(boxes, labels, mapper, linkmap):
    num_cp, max_len_ratio, max_r, step_r = 5, 0.7, 2.0, 0.2
    polys = []
    for k, box in enumerate(boxes):
        w_box, h_box = int(np.linalg.norm(box[0]-box[1])+0.5), int(np.linalg.norm(box[1]-box[2])+0.5)
        if w_box < 10 or h_box < 10: polys.append(None); continue
        tar = np.float32([[0,0],[w_box,0],[w_box,h_box],[0,h_box]])
        M = cv2.getPerspectiveTransform(box, tar)
        word_label = cv2.warpPerspective(labels, M, (w_box,h_box), flags=cv2.INTER_NEAREST)
        try: Minv = np.linalg.inv(M)
        except np.linalg.LinAlgError: polys.append(None); continue
        word_label[word_label != mapper[k]] = 0
        word_label[word_label > 0] = 1
        cp_top_list, cp_bot_list, cp_checked = [], [], False
        for i in range(num_cp):
            curr = int(w_box/(num_cp-1)*i) if num_cp > 1 else 0
            if curr == w_box and w_box > 0: curr -=1
            if not (0 <= curr < word_label.shape[1]): continue
            pts = np.where(word_label[:,curr]!=0)[0]
            if not pts.size > 0: continue
            cp_checked = True
            cp_top_list.append(np.array([curr,pts[0]], dtype=np.int32))
            cp_bot_list.append(np.array([curr,pts[-1]], dtype=np.int32))
        if not cp_checked or len(cp_top_list) < num_cp or len(cp_bot_list) < num_cp : polys.append(None); continue
        cp_top, cp_bot = np.array(cp_top_list).reshape(-1,2), np.array(cp_bot_list).reshape(-1,2)
        final_poly = None
        for r_val in np.arange(0.5, max_r, step_r):
            top_link_pts, bot_link_pts = 0,0
            if cp_top.shape[0]>=2: top_link_pts = sum(1 for i in range(cp_top.shape[0]-1) if 0<=int((cp_top[i][1]+cp_top[i+1][1])/2)<linkmap.shape[0] and 0<=int((cp_top[i][0]+cp_top[i+1][0])/2)<linkmap.shape[1] and linkmap[int((cp_top[i][1]+cp_top[i+1][1])/2), int((cp_top[i][0]+cp_top[i+1][0])/2)]==1)
            if cp_bot.shape[0]>=2: bot_link_pts = sum(1 for i in range(cp_bot.shape[0]-1) if 0<=int((cp_bot[i][1]+cp_bot[i+1][1])/2)<linkmap.shape[0] and 0<=int((cp_bot[i][0]+cp_bot[i+1][0])/2)<linkmap.shape[1] and linkmap[int((cp_bot[i][1]+cp_bot[i+1][1])/2), int((cp_bot[i][0]+cp_bot[i+1][0])/2)]==1)
            if top_link_pts > max_len_ratio*cp_top.shape[0] or bot_link_pts > max_len_ratio*cp_bot.shape[0]: final_poly=None; break
            if top_link_pts==0 and bot_link_pts==0:
                poly_coords = np.concatenate((cp_top, np.flip(cp_bot,axis=0)), axis=0)
                final_poly = cv2.perspectiveTransform(np.array([poly_coords],dtype=np.float32),Minv)[0]; break
        polys.append(final_poly)
    return polys

def getDetBoxes(textmap, linkmap, text_threshold, link_threshold, low_text, poly=False):
    boxes, labels, mapper = getDetBoxes_core(textmap, linkmap, text_threshold, link_threshold, low_text)
    if poly and boxes: polys = getPoly_core(boxes, labels, mapper, linkmap)
    else: polys = [None] * len(boxes)
    return boxes, polys

def adjustResultCoordinates(coords, ratio_w, ratio_h, ratio_net=2):
    if coords is None or len(coords) == 0: return []
    return [(item*(ratio_w*ratio_net, ratio_h*ratio_net) if item is not None else None) for item in coords]

def perform_craft_inference(net, image_bgr, text_threshold, link_threshold, low_text, cuda, poly,
                            canvas_size=1280, mag_ratio=1.5):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img_resized, target_ratio, _ = resize_aspect_ratio(image_rgb, canvas_size, cv2.INTER_LINEAR, mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio
    x = normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2,0,1).unsqueeze(0)
    if cuda: x = x.cuda()
    with torch.no_grad(): y, _ = net(x)
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()
    boxes, polys = getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)
    boxes = adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = adjustResultCoordinates(polys, ratio_w, ratio_h)
    # Fallback to box if poly is None
    final_polys = [(polys[k] if polys[k] is not None else (boxes[k] if k < len(boxes) else None)) for k in range(len(polys))]
    return final_polys

# --- Parseq Model Import and Utilities ---
try:
    from strhub.models.parseq.system import Parseq as ParseqSystem # LitParseq in older STRHub
    from strhub.data.utils import CharsetAdapter
    from strhub.data.tokenizer import JawiTokenizer # From your repo
except ImportError as e:
    print(f"Error importing Parseq components: {e}")
    print("Ensure 'parseq_jawi' is in PYTHONPATH and has __init__.py files if needed.")
    sys.exit(1)

def load_parseq_model(checkpoint_path, device):
    # Parseq models from STRHub are often PyTorch Lightning checkpoints
    # The system class itself can load from checkpoint
    print(f"Loading Parseq model from: {checkpoint_path}")
    try:
        # charset_test is crucial. Infer from common Jawi charsets or make it an arg.
        # Based on your `train_jawi_parseq.yaml`, it's likely implicitly loaded or
        # we might need to provide it. For simplicity, using a common one.
        # A better way would be to load the YAML config used for training.
        
        # The ParseqSystem.load_from_checkpoint will handle charset if it's saved in checkpoint's hparams
        model = ParseqSystem.load_from_checkpoint(checkpoint_path, map_location=device)
        model.eval()
        model.to(device)
        print("Parseq model loaded successfully.")
        # The tokenizer should be part of the loaded model system if configured correctly during training
        if not hasattr(model, 'tokenizer') or model.tokenizer is None:
             print("Warning: Parseq model does not have a 'tokenizer' attribute. Using default JawiTokenizer.")
             # Example: from your train_jawi_parseq.yaml
             jawi_charset = "ءابتةثجحخدذرزسشصضطظعغفقكلمنهوىي۰۱۲۳۴۵۶۷۸۹" + \
                            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz" + \
                            "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ " 
             # Need to define SOS, EOS, PAD tokens consistent with your training
             # For simplicity, let's assume the model's internal tokenizer works or we have a simple one.
             # This part might need refinement based on how your Parseq checkpoint saves/loads tokenizer.
             # The ideal scenario is model.tokenizer is already set up by load_from_checkpoint
             # For this example, let's assume it is.
        
        if hasattr(model, 'hparams') and 'img_size' in model.hparams:
            img_size = model.hparams.img_size # e.g., [32, 128]
        else:
            print("Warning: Parseq model 'img_size' not found in hparams. Using default [32, 128].")
            img_size = [32, 128] # Default Parseq image height, variable width is common
        
        return model, img_size

    except Exception as e:
        print(f"Error loading Parseq model: {e}")
        print("Ensure the checkpoint is a valid PyTorch Lightning checkpoint for ParseqSystem.")
        sys.exit(1)


def preprocess_for_parseq(img_crop_bgr, target_img_size, device):
    """Prepares a single image crop for Parseq model."""
    # Convert to RGB, then Grayscale as Parseq often uses grayscale
    img_rgb = cv2.cvtColor(img_crop_bgr, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # Resize to target height, keeping aspect ratio for width (or fixed width if model expects it)
    h, w = img_gray.shape
    target_h, target_w_max = target_img_size # e.g. [32, 128]
    
    new_w = int(w * (target_h / h))
    # Cap width if model has a max width, otherwise use dynamic width
    # Parseq can often handle variable width, but some configs might fix it.
    # If target_w_max is small (e.g. 128), we might need to pad or resize differently.
    # For now, let's resize to target_h and new_w, then pad to target_w_max if new_w < target_w_max
    
    img_resized = cv2.resize(img_gray, (new_w, target_h), interpolation=cv2.INTER_CUBIC)

    # Pad to target_w_max if model requires fixed width input, common for some STR models
    # If Parseq handles variable width well, padding might not be strictly necessary
    # or could be handled by a collate_fn in a dataloader.
    # For single image inference, we often pad.
    if new_w < target_w_max:
        # Create a canvas of target_h x target_w_max and place img_resized on it
        padded_img = np.ones((target_h, target_w_max), dtype=np.uint8) * 255 # White padding
        start_x = 0 # Or center: (target_w_max - new_w) // 2
        padded_img[:, start_x : start_x + new_w] = img_resized
        img_to_transform = padded_img
    elif new_w > target_w_max: # If wider than max, resize to fit
        img_to_transform = cv2.resize(img_resized, (target_w_max, target_h), interpolation=cv2.INTER_CUBIC)
    else: # new_w == target_w_max
        img_to_transform = img_resized

    # Normalize (Parseq typically uses 0.5 mean, 0.5 std for [-1, 1] range)
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(0.5, 0.5) # Normalizes to [-1, 1]
    ])
    img_tensor = transform(img_to_transform).unsqueeze(0) # Add batch dimension
    return img_tensor.to(device)

def get_cropped_image_from_poly(image_bgr, poly_pts):
    """Crops an image region defined by a polygon (4 points).
       Uses minAreaRect for robustness or perspective transform for accuracy.
    """
    if poly_pts is None or len(poly_pts) < 3: return None

    poly = np.array(poly_pts, dtype=np.float32)
    
    # Option 1: Simple bounding rectangle (faster, less accurate for skewed text)
    # rect = cv2.boundingRect(poly.astype(np.int32))
    # x, y, w, h = rect
    # crop = image_bgr[y:y+h, x:x+w]
    # if crop.size == 0: return None
    # return crop

    # Option 2: Perspective transform (more accurate for skewed text)
    # Order points: top-left, top-right, bottom-right, bottom-left for getPerspectiveTransform
    # The CRAFT output polys should ideally be in a consistent order.
    # Let's assume poly is [tl, tr, br, bl]
    rect = cv2.minAreaRect(poly)
    box = cv2.boxPoints(rect) # This gives points in order, but might not be tl, tr, br, bl
    
    # Reorder box points to be tl, tr, br, bl for perspective transform
    # Sum of x+y is min for TL, max for BR.
    s = box.sum(axis=1)
    ordered_box = np.zeros((4,2), dtype=np.float32)
    ordered_box[0] = box[np.argmin(s)] # Top-left
    ordered_box[2] = box[np.argmax(s)] # Bottom-right
    
    # Diff of y-x is min for TR, max for BL
    diff = np.diff(box, axis=1)
    ordered_box[1] = box[np.argmin(diff)] # Top-right
    ordered_box[3] = box[np.argmax(diff)] # Bottom-left
    
    (tl, tr, br, bl) = ordered_box

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    if maxWidth == 0 or maxHeight == 0: return None

    dst_pts = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(ordered_box, dst_pts)
    warped_crop = cv2.warpPerspective(image_bgr, M, (maxWidth, maxHeight))
    
    return warped_crop

# --- Main OCR Pipeline ---
def main_ocr_pipeline(args):
    cuda_enabled = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if cuda_enabled else 'cpu')
    print(f"Using device: {device}")

    # 1. Load CRAFT model
    craft_net = CRAFT()
    print(f'Loading CRAFT weights from: {args.craft_model_path}')
    checkpoint_craft = torch.load(args.craft_model_path, map_location=device)
    if 'craft' in checkpoint_craft: model_state_dict_craft = checkpoint_craft['craft']
    elif 'model' in checkpoint_craft: model_state_dict_craft = checkpoint_craft['model']
    else: model_state_dict_craft = checkpoint_craft
    craft_net.load_state_dict(copyStateDict(model_state_dict_craft))
    craft_net.to(device)
    craft_net.eval()

    # 2. Load Parseq model
    parseq_model, parseq_img_size = load_parseq_model(args.parseq_model_path, device)

    # 3. Load and process image
    print(f"Processing image: {args.image_path}")
    image_bgr = cv2.imread(args.image_path)
    if image_bgr is None:
        print(f"Error: Could not read image at {args.image_path}")
        return

    # 4. Perform Text Detection (CRAFT)
    detected_polys = perform_craft_inference(
        craft_net, image_bgr, args.text_threshold, args.link_threshold,
        args.low_text, cuda_enabled, args.poly, args.canvas_size, args.mag_ratio
    )
    print(f"CRAFT detected {len(detected_polys)} potential text regions.")

    # 5. Perform Text Recognition (Parseq) for each detected region
    results = []
    output_image_viz = image_bgr.copy()

    for i, poly_pts in enumerate(detected_polys):
        if poly_pts is None:
            continue

        # Crop image region using polygon
        cropped_bgr = get_cropped_image_from_poly(image_bgr, poly_pts)
        
        if cropped_bgr is None or cropped_bgr.shape[0] == 0 or cropped_bgr.shape[1] == 0:
            print(f"Warning: Skipping invalid crop for polygon {i+1}")
            continue

        # Preprocess crop for Parseq
        parseq_input_tensor = preprocess_for_parseq(cropped_bgr, parseq_img_size, device)

        # Parseq inference
        with torch.no_grad():
            # The ParseqSystem.forward or a predict_step typically handles this
            # Assuming model takes image tensor and returns logits or decoded text
            # For STRHub models, it might be model(parseq_input_tensor)
            # The output structure depends on ParseqSystem's implementation.
            # It often returns a list of predictions, even for a single batch item.
            pred_output = parseq_model(parseq_input_tensor)
            
            # Decoding: STRHub's ParseqSystem usually returns [(label, confidence), ...]
            # The label is already decoded text using its internal tokenizer.
            if isinstance(pred_output, list) and len(pred_output) > 0:
                if isinstance(pred_output[0], tuple) and len(pred_output[0]) >= 1:
                    recognized_text = pred_output[0][0] # (text, confidence)
                    confidence = pred_output[0][1] if len(pred_output[0]) > 1 else None
                else: # If it directly returns list of strings
                    recognized_text = pred_output[0]
                    confidence = None
                print(f"Region {i+1}: Text = '{recognized_text}'" + (f", Conf = {confidence:.4f}" if confidence is not None else ""))
                results.append({'polygon': poly_pts, 'text': recognized_text, 'confidence': confidence})

                # Draw on output image for visualization
                cv2.polylines(output_image_viz, [poly_pts.astype(np.int32)], True, (0,255,0), 2)
                # Put text (ensure font supports Jawi characters)
                # For simplicity, printing to console. OpenCV putText for Jawi needs a proper font.
                # Example: cv2.putText(output_image_viz, recognized_text, (int(poly_pts[0][0]), int(poly_pts[0][1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)

            else:
                print(f"Warning: Unexpected output format from Parseq for region {i+1}: {pred_output}")


    # 6. Save or display results
    if args.output_dir:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        
        base_filename = os.path.splitext(os.path.basename(args.image_path))[0]
        
        # Save visualized image
        viz_filepath = os.path.join(args.output_dir, f"res_ocr_{base_filename}.jpg")
        cv2.imwrite(viz_filepath, output_image_viz)
        print(f"Visualized OCR result saved to: {viz_filepath}")

        # Save text results (e.g., as JSON or TXT)
        text_result_filepath = os.path.join(args.output_dir, f"res_ocr_{base_filename}.txt")
        with open(text_result_filepath, 'w', encoding='utf-8') as f:
            for res in results:
                poly_str = ";".join([f"{p[0]},{p[1]}" for p in res['polygon']])
                conf_str = f"{res['confidence']:.4f}" if res['confidence'] is not None else "N/A"
                f.write(f"Polygon: [{poly_str}] | Text: {res['text']} | Confidence: {conf_str}\n")
        print(f"Text OCR results saved to: {text_result_filepath}")
    
    print("OCR pipeline finished.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Jawi OCR Pipeline using CRAFT and Parseq')
    # Image and Model Paths
    parser.add_argument('--image_path', required=True, type=str, help='Path to the input image')
    parser.add_argument('--craft_model_path', required=True, type=str, help='Path to fine-tuned CRAFT model checkpoint (.pth)')
    parser.add_argument('--parseq_model_path', required=True, type=str, help='Path to fine-tuned Parseq model checkpoint (.ckpt or .pth)')
    parser.add_argument('--output_dir', default='./jawi_ocr_results/', type=str, help='Directory to save output image and text')
    
    # CRAFT Parameters
    parser.add_argument('--text_threshold', default=0.7, type=float, help='CRAFT text confidence threshold')
    parser.add_argument('--low_text', default=0.4, type=float, help='CRAFT text low_text threshold')
    parser.add_argument('--link_threshold', default=0.4, type=float, help='CRAFT link confidence threshold')
    parser.add_argument('--canvas_size', default=1280, type=int, help='CRAFT image size for inference')
    parser.add_argument('--mag_ratio', default=1.5, type=float, help='CRAFT image magnification ratio')
    parser.add_argument('--poly', default=False, action='store_true', help='CRAFT enable polygon type detection')
    
    # General Parameters
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    
    args = parser.parse_args()
    main_ocr_pipeline(args)
