import os
import sys
import argparse
import time
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from collections import OrderedDict
from PIL import Image as PILImage
from torchvision import transforms as TorchTransforms

# --- Path Setup ---
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
    print(f"Error importing 'CRAFT' from 'model.craft': {e}")
    print(f"Please ensure that '{os.path.join(craft_module_dir, 'model', 'craft.py')}' exists and is accessible.")
    print(f"Attempted to look in: {craft_module_dir}")
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
    return resized, ratio, (int(target_w32/2), int(target_h32/2)) 

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
        segmap[np.logical_and(link_score==1, text_score==0)] = 0 
        x, y = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP]
        w, h = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
        niter = int(np.sqrt(size * min(w,h) / (w*h)) * 2) if w*h > 0 else 0 
        sx, ex, sy, ey = max(0, x-niter), min(img_w, x+w+niter+1), max(0, y-niter), min(img_h, y+h+niter+1)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(1+niter, 1+niter))
        segmap[sy:ey, sx:ex] = cv2.dilate(segmap[sy:ey, sx:ex], kernel)
        np_contours = np.roll(np.array(np.where(segmap!=0)),1,axis=0).transpose().reshape(-1,2)
        if np_contours.size == 0: 
            continue
        rectangle = cv2.minAreaRect(np_contours)
        box_pts = cv2.boxPoints(rectangle)
        box_w, box_h = np.linalg.norm(box_pts[0] - box_pts[1]), np.linalg.norm(box_pts[1] - box_pts[2])
        if abs(1 - max(box_w,box_h)/(min(box_w,box_h)+1e-5)) <= 0.1: 
            l, r = min(np_contours[:,0]), max(np_contours[:,0])
            t, b = min(np_contours[:,1]), max(np_contours[:,1])
            box_pts = np.array([[l, t], [r, t], [r, b], [l, b]], dtype=np.float32)
        startidx = box_pts.sum(axis=1).argmin() 
        box_pts = np.roll(box_pts, 4-startidx, 0)
        det.append(box_pts)
        mapper.append(k)
    return det, labels, mapper

def getPoly_core(boxes, labels, mapper, linkmap):
    num_cp, max_len_ratio, max_r, step_r = 5, 0.7, 2.0, 0.2
    polys = []
    for k, box_pts_param in enumerate(boxes): 
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
        polys = [None] * len(boxes) 
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
    with torch.no_grad(): y, _ = net(x) 
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()
    boxes, polys = getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)
    boxes = adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = adjustResultCoordinates(polys, ratio_w, ratio_h)
    final_polys = []
    for k in range(max(len(polys) if polys else 0, len(boxes) if boxes else 0) ): 
        poly_item = polys[k] if polys and k < len(polys) else None
        box_item = boxes[k] if boxes and k < len(boxes) else None
        if poly_item is not None:
            final_polys.append(poly_item)
        elif box_item is not None:
            final_polys.append(box_item)
    return [p for p in final_polys if p is not None]


# --- Parseq Model Import and Utilities ---
try:
    from strhub.models.utils import load_from_checkpoint as parseq_load_from_checkpoint
    from strhub.data.module import SceneTextDataModule
except ImportError as e:
    print(f"Error importing Parseq components: {e}\nLooked in: {parseq_module_dir}")
    sys.exit(1)

def load_parseq_model_strhub(checkpoint_path, device):
    print(f"Loading Parseq model using STRHub method from: {checkpoint_path}")
    try:
        kwargs = {} 
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
    if img_crop_bgr is None or img_crop_bgr.shape[0] == 0 or img_crop_bgr.shape[1] == 0:
        return None
    img_rgb_pil = PILImage.fromarray(cv2.cvtColor(img_crop_bgr, cv2.COLOR_BGR2RGB))
    img_tensor = parseq_transform(img_rgb_pil).unsqueeze(0) 
    return img_tensor.to(device)


# --- Simple Per-Crop Orientation Correction ---
def simple_orientation_correction(image_crop_bgr: np.ndarray) -> np.ndarray:
    if image_crop_bgr is None or image_crop_bgr.shape[0] == 0 or image_crop_bgr.shape[1] == 0:
        return image_crop_bgr
    h, w = image_crop_bgr.shape[:2]
    if w < h: 
        return cv2.rotate(image_crop_bgr, cv2.ROTATE_90_CLOCKWISE)
    else:
        return image_crop_bgr

# --- Cropping Utility ---
def get_cropped_image_from_poly(image_bgr, poly_pts):
    if poly_pts is None or len(poly_pts) < 4: return None
    
    poly = np.array(poly_pts, dtype=np.float32)
    rect = cv2.minAreaRect(poly)  
    
    w_rect, h_rect = rect[1] 
    angle = rect[2]

    target_w, target_h = 0, 0
    if abs(angle) > 45: 
        target_w = int(h_rect)
        target_h = int(w_rect)
    else: 
        target_w = int(w_rect)
        target_h = int(h_rect)

    if target_w <= 0 or target_h <= 0:
        x_coords = poly[:, 0]; y_coords = poly[:, 1]
        xmin, xmax = np.min(x_coords), np.max(x_coords)
        ymin, ymax = np.min(y_coords), np.max(y_coords)
        target_w = int(xmax - xmin); target_h = int(ymax - ymin)
        if target_w <= 0 or target_h <= 0: return None

    dst_pts = np.array([[0, 0], [target_w - 1, 0], [target_w - 1, target_h - 1], [0, target_h - 1]], dtype="float32")
    
    s = poly.sum(axis=1)
    ordered_src_pts = np.zeros((4, 2), dtype="float32")
    ordered_src_pts[0] = poly[np.argmin(s)] 
    ordered_src_pts[2] = poly[np.argmax(s)] 
    
    remaining_indices = [i for i, pt in enumerate(poly) 
                         if not np.array_equal(pt, ordered_src_pts[0]) and \
                            not np.array_equal(pt, ordered_src_pts[2])]
    
    if len(remaining_indices) == 2:
        pt1 = poly[remaining_indices[0]]
        pt2 = poly[remaining_indices[1]]
        if pt1[1] < pt2[1] or (abs(pt1[1] - pt2[1]) < 1e-3 and pt1[0] > pt2[0]): 
            ordered_src_pts[1] = pt1 
            ordered_src_pts[3] = pt2 
        else:
            ordered_src_pts[1] = pt2 
            ordered_src_pts[3] = pt1 
    else: 
        print("Warning: Polygon point reordering for perspective transform might be incorrect. Using as-is.")
        ordered_src_pts = poly.astype("float32") 

    M = cv2.getPerspectiveTransform(ordered_src_pts, dst_pts)
    warped_crop = cv2.warpPerspective(image_bgr, M, (target_w, target_h))
    
    if warped_crop.shape[0] == 0 or warped_crop.shape[1] == 0: return None
    return warped_crop


# --- Main OCR Pipeline ---
def main_ocr_pipeline(args):
    cuda_enabled = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if cuda_enabled else 'cpu')
    print(f"Using device: {device}")

    debug_output_dir = ""
    if args.save_debug_crops:
        debug_output_dir = os.path.join(args.output_dir, "debug_crops")
        if not os.path.exists(debug_output_dir):
            os.makedirs(debug_output_dir)
            print(f"Created debug crops directory: {debug_output_dir}")

    print(f"Processing image: {args.image_path}")
    image_bgr_original = cv2.imread(args.image_path)
    if image_bgr_original is None:
        print(f"Error: Could not read image at {args.image_path}"); return
    base_image_filename = os.path.splitext(os.path.basename(args.image_path))[0]
    
    # Image fed to CRAFT is the original image without global orientation correction
    image_bgr_processed_for_craft = image_bgr_original.copy() 

    craft_net = CRAFT()
    print(f'Loading CRAFT weights from: {args.craft_model_path}')
    checkpoint_craft = torch.load(args.craft_model_path,map_location=device,weights_only=False)
    if 'craft' in checkpoint_craft: model_state_dict_craft=checkpoint_craft['craft']
    elif 'model' in checkpoint_craft: model_state_dict_craft=checkpoint_craft['model']
    elif 'state_dict' in checkpoint_craft: model_state_dict_craft=checkpoint_craft['state_dict']
    else: model_state_dict_craft=checkpoint_craft
    craft_net.load_state_dict(copyStateDict(model_state_dict_craft)); craft_net.to(device); craft_net.eval()

    parseq_model,parseq_img_transform = load_parseq_model_strhub(args.parseq_model_path,device)
    
    detected_polys_from_craft = perform_craft_inference(
        craft_net,image_bgr_processed_for_craft,args.text_threshold,args.link_threshold,
        args.low_text,cuda_enabled,args.poly,args.canvas_size,args.mag_ratio
    )
    print(f"CRAFT detected {len(detected_polys_from_craft)} text regions.")

    regions_with_x_coords_for_sort = []
    if detected_polys_from_craft:
        for poly_pts in detected_polys_from_craft:
            if poly_pts is not None and len(poly_pts)>0:
                moments = cv2.moments(poly_pts.astype(np.int32))
                center_x = int(moments["m10"]/moments["m00"]) if moments["m00"]!=0 else int(np.mean(poly_pts[:,0]))
                regions_with_x_coords_for_sort.append((center_x,poly_pts))
        regions_with_x_coords_for_sort.sort(key=lambda item:item[0],reverse=True)
        sorted_detected_polys = [item[1] for item in regions_with_x_coords_for_sort]
        print(f"Regions sorted R-L: {len(sorted_detected_polys)} regions.")
    else: sorted_detected_polys=[]

    results_data,recognized_text_snippets = [],[]
    output_image_viz = image_bgr_processed_for_craft.copy() # Visualize on the same image CRAFT saw

    for i, poly_pts in enumerate(sorted_detected_polys):
        # Cropping is done from the image that CRAFT processed
        cropped_bgr = get_cropped_image_from_poly(image_bgr_processed_for_craft, poly_pts)
        if cropped_bgr is None or cropped_bgr.shape[0]==0 or cropped_bgr.shape[1]==0:
            print(f"Skipping invalid crop for sorted region {i+1}"); continue
        
        if args.save_debug_crops:
            fn=f"{base_image_filename}_sregion_{i+1}_crop.png"; fp=os.path.join(debug_output_dir,fn)
            try: cv2.imwrite(fp,cropped_bgr)
            except Exception as e: print(f"Err save crop {fp}:{e}")
        
        crop_for_parseq = cropped_bgr
        if args.use_simple_orientation:
            corrected_bgr_crop = simple_orientation_correction(cropped_bgr)
            crop_for_parseq = corrected_bgr_crop
            if args.save_debug_crops and corrected_bgr_crop is not cropped_bgr:
                fn=f"{base_image_filename}_sregion_{i+1}_corrected.png"; fp=os.path.join(debug_output_dir,fn)
                try: cv2.imwrite(fp,crop_for_parseq)
                except Exception as e: print(f"Err save corrected {fp}:{e}")
        
        parseq_input_tensor = preprocess_for_parseq_strhub(crop_for_parseq,parseq_img_transform,device)
        if parseq_input_tensor is None: 
            print(f"Skipping sorted region {i+1} (Parseq preprocess fail)."); continue
        
        with torch.no_grad():
            logits = parseq_model(parseq_input_tensor)
            probabilities = logits.softmax(-1)
            pred_texts,pred_confs_tensors_list = parseq_model.tokenizer.decode(probabilities)
            if pred_texts and len(pred_texts)>0:
                recognized_text_segment = pred_texts[0]
                token_confidences = pred_confs_tensors_list[0] if pred_confs_tensors_list and len(pred_confs_tensors_list)>0 else None
                seq_conf = None
                if token_confidences is not None and isinstance(token_confidences,torch.Tensor):
                    seq_conf = token_confidences.mean().item() if token_confidences.numel()>1 else token_confidences.item()
                
                x_key = "N/A"
                if i < len(regions_with_x_coords_for_sort): # Check index validity
                     x_key = regions_with_x_coords_for_sort[i][0]

                conf_str = f"{seq_conf:.4f}" if seq_conf is not None else "N/A"
                print(f"SRegion {i+1}(X:{x_key}):Txt='{recognized_text_segment}',Conf={conf_str}")
                results_data.append({'orig_x':x_key,'poly':poly_pts,'text':recognized_text_segment,'conf':seq_conf})
                recognized_text_snippets.append(recognized_text_segment)
                cv2.polylines(output_image_viz,[poly_pts.astype(np.int32)],True,(0,0,255),2)
            else: print(f"No text decoded for sorted region {i+1}")
            
    final_rl_text = " ".join(recognized_text_snippets)
    print(f"\nFinal Combined R-L Text: {final_rl_text}\n")

    if args.output_dir:
        if not os.path.exists(args.output_dir): os.makedirs(args.output_dir)
        cv2.imwrite(os.path.join(args.output_dir,f"res_ocr_{base_image_filename}.jpg"),output_image_viz)
        print(f"Viz saved to: {os.path.join(args.output_dir,f'res_ocr_{base_image_filename}.jpg')}")
        with open(os.path.join(args.output_dir,f"res_ocr_{base_image_filename}.txt"),'w',encoding='utf-8') as f:
            f.write(f"Final Text (R-L): {final_rl_text}\n\nRegions (Sorted R-L):\n")
            for res in results_data:
                poly_s = ";".join([f"{int(p[0])},{int(p[1])}" for p in res['poly']]) if res['poly'] is not None else "N/A"
                conf_s = f"{res['conf']:.4f}" if res['conf'] is not None else "N/A"
                f.write(f"X-Key:{res['orig_x']}|Poly:[{poly_s}]|Txt:{res['text']}|Conf:{conf_s}\n")
        print(f"Txt results saved to: {os.path.join(args.output_dir,f'res_ocr_{base_image_filename}.txt')}")
    print("OCR pipeline finished.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Jawi OCR: CRAFT + Simple Orientation + Parseq + R-L Sort')
    parser.add_argument('--image_path', required=True, type=str, help='Path to the input image')
    parser.add_argument('--craft_model_path', required=True, type=str, help='Path to CRAFT model')
    parser.add_argument('--parseq_model_path', required=True, type=str, help='Path to Parseq model (STRHub)')
    parser.add_argument('--output_dir', default='./jawi_ocr_results/', type=str, help='Directory for results')
    
    parser.add_argument('--text_threshold', default=0.7, type=float, help='CRAFT: text confidence threshold')
    parser.add_argument('--low_text', default=0.4, type=float, help='CRAFT: text low_text threshold')
    parser.add_argument('--link_threshold', default=0.4, type=float, help='CRAFT: link confidence threshold')
    parser.add_argument('--canvas_size', default=1280, type=int, help='CRAFT: image size for inference')
    parser.add_argument('--mag_ratio', default=1.5, type=float, help='CRAFT: image magnification ratio')
    parser.add_argument('--poly', default=False, action='store_true', help='CRAFT: enable polygon type detection')
    
    parser.add_argument('--use_simple_orientation', action='store_true', help='Enable W<H per-crop orientation correction')
    parser.add_argument('--save_debug_crops', action='store_true', help='Save intermediate cropped images')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    
    args = parser.parse_args()
    main_ocr_pipeline(args)
