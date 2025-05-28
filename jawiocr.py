import os
import sys
import argparse
import time
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from collections import OrderedDict
from PIL import Image as PILImage # Use a different alias for PIL to avoid conflict with cv2.Image
from torchvision import transforms as TorchTransforms # Alias for torchvision transforms


# Add craft and parseq_jawi directories to sys.path
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
    from model.craft import CRAFT
except ImportError as e:
    print(f"Error importing 'CRAFT' from 'model.craft': {e}\nLooked in: {craft_module_dir}")
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
        niter = int(np.sqrt(size * min(w,h) / (w*h)) * 2) if w*h > 0 else 0
        sx, ex, sy, ey = max(0, x-niter), min(img_w, x+w+niter+1), max(0, y-niter), min(img_h, y+h+niter+1)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(1+niter, 1+niter))
        segmap[sy:ey, sx:ex] = cv2.dilate(segmap[sy:ey, sx:ex], kernel)
        np_contours = np.roll(np.array(np.where(segmap!=0)),1,axis=0).transpose().reshape(-1,2)
        if np_contours.size == 0: continue
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
    for k, box_pts in enumerate(boxes):
        w_box, h_box = int(np.linalg.norm(box_pts[0]-box_pts[1])+0.5), int(np.linalg.norm(box_pts[1]-box_pts[2])+0.5)
        if w_box < 10 or h_box < 10: polys.append(None); continue
        tar = np.float32([[0,0],[w_box,0],[w_box,h_box],[0,h_box]])
        M = cv2.getPerspectiveTransform(box_pts, tar)
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
            if cp_top.shape[0]>=2: top_link_pts = sum(1 for i_pt in range(cp_top.shape[0]-1) if 0<=int((cp_top[i_pt][1]+cp_top[i_pt+1][1])/2)<linkmap.shape[0] and 0<=int((cp_top[i_pt][0]+cp_top[i_pt+1][0])/2)<linkmap.shape[1] and linkmap[int((cp_top[i_pt][1]+cp_top[i_pt+1][1])/2), int((cp_top[i_pt][0]+cp_top[i_pt+1][0])/2)]==1)
            if cp_bot.shape[0]>=2: bot_link_pts = sum(1 for i_pt in range(cp_bot.shape[0]-1) if 0<=int((cp_bot[i_pt][1]+cp_bot[i_pt+1][1])/2)<linkmap.shape[0] and 0<=int((cp_bot[i_pt][0]+cp_bot[i_pt+1][0])/2)<linkmap.shape[1] and linkmap[int((cp_bot[i_pt][1]+cp_bot[i_pt+1][1])/2), int((cp_bot[i_pt][0]+cp_bot[i_pt+1][0])/2)]==1)
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
    final_polys = [(polys[k] if polys[k] is not None else (boxes[k] if k < len(boxes) and boxes[k] is not None else None)) for k in range(len(polys))]
    return final_polys

# --- Parseq Model Import and Utilities ---
try:
    # Corrected imports based on STRHub's read.py
    from strhub.models.utils import load_from_checkpoint as parseq_load_from_checkpoint
    from strhub.data.module import SceneTextDataModule
    # JawiTokenizer is still needed if the model's internal tokenizer isn't sufficient or needs specific setup
    # from strhub.data.tokenizer import JawiTokenizer # Keep if needed for manual setup
except ImportError as e:
    print(f"Error importing Parseq components: {e}\nLooked in: {parseq_module_dir}")
    sys.exit(1)

def load_parseq_model_strhub(checkpoint_path, device):
    print(f"Loading Parseq model using STRHub method from: {checkpoint_path}")
    try:
        # kwargs can be used to pass additional arguments if needed by your specific model variant
        # e.g., if your model needs `charset_test` to be explicitly passed at load time
        # kwargs = {'charset_test': "your jawi charset string here if model requires it"}
        kwargs = {} # Start with empty, add if necessary based on model's needs
        model = parseq_load_from_checkpoint(checkpoint_path, **kwargs).eval().to(device)
        
        print("Parseq model loaded successfully via STRHub.")
        
        # Get image transform from the model's hparams (as done in read.py)
        img_transform = SceneTextDataModule.get_transform(model.hparams.img_size)
        
        # The tokenizer should be part of the loaded model as `model.tokenizer`
        if not hasattr(model, 'tokenizer') or model.tokenizer is None:
            print("CRITICAL Error: Parseq model loaded via STRHub does not have a 'tokenizer'. "
                  "This is unexpected. Ensure your checkpoint and STRHub environment are correctly set up.")
            # If this happens, you might need to investigate why the tokenizer isn't part of the model
            # or attempt manual JawiTokenizer setup, though STRHub aims to handle this.
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
    
    # Convert BGR (OpenCV) to RGB PIL Image for STRHub's transform
    img_rgb_pil = PILImage.fromarray(cv2.cvtColor(img_crop_bgr, cv2.COLOR_BGR2RGB))
    
    # Apply the transform obtained from SceneTextDataModule
    img_tensor = parseq_transform(img_rgb_pil).unsqueeze(0) # Add batch dimension
    return img_tensor.to(device)


def get_cropped_image_from_poly(image_bgr, poly_pts):
    if poly_pts is None or len(poly_pts) < 3: return None
    poly = np.array(poly_pts, dtype=np.float32)
    rect = cv2.minAreaRect(poly)
    box = cv2.boxPoints(rect)
    width = int(rect[1][0])
    height = int(rect[1][1])
    angle = rect[2]
    if angle < -45:
        width, height = height, width
    if width == 0 or height == 0: return None
    src_pts = box.astype("float32")
    dst_pts = np.array([[0, height-1], [0, 0], [width-1, 0], [width-1, height-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped_crop = cv2.warpPerspective(image_bgr, M, (width, height))
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
    elif 'state_dict' in checkpoint_craft: model_state_dict_craft = checkpoint_craft['state_dict']
    else: model_state_dict_craft = checkpoint_craft
    craft_net.load_state_dict(copyStateDict(model_state_dict_craft))
    craft_net.to(device)
    craft_net.eval()

    # 2. Load Parseq model using STRHub's method
    parseq_model, parseq_img_transform = load_parseq_model_strhub(args.parseq_model_path, device)

    # 3. Load image
    print(f"Processing image: {args.image_path}")
    image_bgr_original = cv2.imread(args.image_path)
    if image_bgr_original is None:
        print(f"Error: Could not read image at {args.image_path}"); return

    # 4. Perform Text Detection (CRAFT)
    detected_polys = perform_craft_inference(
        craft_net, image_bgr_original, args.text_threshold, args.link_threshold,
        args.low_text, cuda_enabled, args.poly, args.canvas_size, args.mag_ratio
    )
    print(f"CRAFT detected {len(detected_polys)} potential text regions.")

    # 5. Perform Text Recognition (Parseq)
    results = []
    output_image_viz = image_bgr_original.copy()

    for i, poly_pts in enumerate(detected_polys):
        if poly_pts is None: continue
        cropped_bgr = get_cropped_image_from_poly(image_bgr_original, poly_pts)
        if cropped_bgr is None: print(f"Warning: Skipping invalid crop for polygon {i+1}"); continue

        parseq_input_tensor = preprocess_for_parseq_strhub(cropped_bgr, parseq_img_transform, device)
        if parseq_input_tensor is None: print(f"Warning: Skipping region {i+1} due to preprocessing error."); continue

        with torch.no_grad():
            logits = parseq_model(parseq_input_tensor) # model(image) returns logits
            probabilities = logits.softmax(-1)
            # model.tokenizer.decode(probabilities) returns (list_of_texts, list_of_confidences)
            pred_texts, pred_confs = parseq_model.tokenizer.decode(probabilities)
            
            if pred_texts and len(pred_texts) > 0:
                recognized_text = pred_texts[0]
                confidence_tensor = pred_confs[0] if pred_confs and len(pred_confs) > 0 else None
                
                # Convert confidence tensor to Python float before formatting
                confidence_float = None
                if confidence_tensor is not None:
                    if isinstance(confidence_tensor, torch.Tensor):
                        confidence_float = confidence_tensor.item() 
                    else: # If it's already a float (though STRHub usually gives tensor)
                        confidence_float = confidence_tensor

                print(f"Region {i+1}: Text = '{recognized_text}'" + 
                      (f", Conf = {confidence_float:.4f}" if confidence_float is not None else ""))
                
                # Store the float value of confidence in results
                results.append({'polygon': poly_pts, 'text': recognized_text, 
                                'confidence': confidence_float}) # Store float here
                
                cv2.polylines(output_image_viz, [poly_pts.astype(np.int32)], True, (0,255,0), 2)
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
                poly_str = ";".join([f"{int(p[0])},{int(p[1])}" for p in res['polygon']])
                # Confidence is now a float, so this formatting is fine
                conf_str = f"{res['confidence']:.4f}" if res['confidence'] is not None else "N/A"
                f.write(f"Polygon: [{poly_str}] | Text: {res['text']} | Confidence: {conf_str}\n")
        print(f"Text OCR results saved to: {text_result_filepath}")
    print("OCR pipeline finished.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Jawi OCR Pipeline using CRAFT and Parseq (STRHub method)')
    parser.add_argument('--image_path', required=True, type=str)
    parser.add_argument('--craft_model_path', required=True, type=str)
    parser.add_argument('--parseq_model_path', required=True, type=str, help="Parseq model checkpoint (STRHub format)")
    parser.add_argument('--output_dir', default='./jawi_ocr_results/', type=str)
    parser.add_argument('--text_threshold', default=0.7, type=float)
    parser.add_argument('--low_text', default=0.4, type=float)
    parser.add_argument('--link_threshold', default=0.4, type=float)
    parser.add_argument('--canvas_size', default=1280, type=int)
    parser.add_argument('--mag_ratio', default=1.5, type=float)
    parser.add_argument('--poly', default=False, action='store_true')
    parser.add_argument('--no_cuda', action='store_true')
    args = parser.parse_args()
    main_ocr_pipeline(args)
