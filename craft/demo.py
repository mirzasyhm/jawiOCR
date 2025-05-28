import argparse
import os
import time
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from collections import OrderedDict

# Attempt to import CRAFT model from EasyOCR's structure
try:
    from model.craft import CRAFT
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
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
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

    # magnify image size
    target_size = mag_ratio * max(height, width)

    # set original image size
    if target_size > square_size:
        target_size = square_size
    
    ratio = target_size / max(height, width)    

    target_h, target_w = int(height * ratio), int(width * ratio)
    proc = cv2.resize(img, (target_w, target_h), interpolation = interpolation)


    # make canvas and paste image
    target_h32, target_w32 = target_h, target_w
    if target_h % 32 != 0:
        target_h32 = target_h + (32 - target_h % 32)
    if target_w % 32 != 0:
        target_w32 = target_w + (32 - target_w % 32)
    resized = np.zeros((target_h32, target_w32, channel), dtype=np.uint8)
    resized[0:target_h, 0:target_w, :] = proc
    target_h, target_w = target_h32, target_w32

    size_heatmap = (int(target_w/2), int(target_h/2))

    return resized, ratio, size_heatmap

def getDetBoxes_core(textmap, linkmap, text_threshold, link_threshold, low_text):
    # prepare data
    linkmap = linkmap.copy()
    textmap = textmap.copy()
    img_h, img_w = textmap.shape

    """ labeling method """
    ret, text_score = cv2.threshold(textmap, low_text, 1, 0)
    ret, link_score = cv2.threshold(linkmap, link_threshold, 1, 0)

    text_score_comb = np.clip(text_score + link_score, 0, 1)
    nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(text_score_comb.astype(np.uint8), connectivity=4)

    det = []
    mapper = []
    for k in range(1,nLabels):
        # size filtering
        size = stats[k, cv2.CC_STAT_AREA]
        if size < 10: continue

        # thresholding
        if np.max(textmap[labels==k]) < text_threshold: continue

        # make segmentation map
        segmap = np.zeros(textmap.shape, dtype=np.uint8)
        segmap[labels==k] = 255
        segmap[np.logical_and(link_score==1, text_score==0)] = 0   # remove link area
        x, y = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP]
        w, h = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
        niter = int(np.sqrt(size * min(w,h) / (w*h)) * 2)
        sx, ex, sy, ey = x - niter, x + w + niter + 1, y - niter, y + h + niter + 1
        # boundary check
        sx = max(0, sx); sy = max(0, sy)
        ex = min(img_w, ex); ey = min(img_h, ey)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(1+niter, 1+niter))
        segmap[sy:ey, sx:ex] = cv2.dilate(segmap[sy:ey, sx:ex], kernel)

        # make box
        np_contours = np.roll(np.array(np.where(segmap!=0)),1,axis=0).transpose().reshape(-1,2)
        rectangle = cv2.minAreaRect(np_contours)
        box = cv2.boxPoints(rectangle)

        # align diamond-shape
        w, h = np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2])
        box_ratio = max(w,h) / (min(w,h) + 1e-5)
        if abs(1 - box_ratio) <= 0.1:
            l, r = min(np_contours[:,0]), max(np_contours[:,0])
            t, b = min(np_contours[:,1]), max(np_contours[:,1])
            box = np.array([[l, t], [r, t], [r, b], [l, b]], dtype=np.float32)

        # make clock-wise order
        startidx = box.sum(axis=1).argmin()
        box = np.roll(box, 4-startidx, 0)
        box = np.array(box)

        det.append(box)
        mapper.append(k)

    return det, labels, mapper

def getPoly_core(boxes, labels, mapper, linkmap):
    # configs
    num_cp = 5
    max_len_ratio = 0.7
    expand_ratio = 1.45
    max_r = 2.0
    step_r = 0.2

    polys = []  
    for k,box in enumerate(boxes):
        # size filter for small instance
        w, h = int(np.linalg.norm(box[0]-box[1])+0.5), int(np.linalg.norm(box[1]-box[2])+0.5)
        if w < 10 or h < 10:
            polys.append(None); continue

        # warp image
        tar = np.float32([[0,0],[w,0],[w,h],[0,h]])
        M = cv2.getPerspectiveTransform(box, tar)
        word_label = cv2.warpPerspective(labels, M, (w,h), flags=cv2.INTER_NEAREST)
        try:
            Minv = np.linalg.inv(M)
        except:
            polys.append(None); continue

        # binarization for selected label
        cur_label = mapper[k]
        word_label[word_label != cur_label] = 0
        word_label[word_label > 0] = 1

        """ Polygon generation """
        # find top/bottom contours
        cp_checked = False
        for i in range(num_cp):
            curr = int(w / (num_cp-1) * i)
            if curr == w: curr -= 1
            top_cnt_pts = np.where(word_label[:,curr]!=0)[0]
            if len(top_cnt_pts) == 0: continue
            else: cp_checked = True
            top_cp = np.array([curr, top_cnt_pts[0]], dtype=np.int32).reshape(1,2)
            bot_cp = np.array([curr, top_cnt_pts[-1]], dtype=np.int32).reshape(1,2)
            if i == 0: 
                cp_top, cp_bot = top_cp, bot_cp
            else:
                cp_top = np.concatenate((cp_top, top_cp), axis=0)
                cp_bot = np.concatenate((cp_bot, bot_cp), axis=0)
        
        if not cp_checked: # if not valid top/bottom contour => use box as polygon
            polys.append(None); continue
            
        # an add-hoc post-processing for removing border connected sequences
        # check if num_cp = 5
        if len(cp_top) < 5 or len(cp_bot) < 5: 
            polys.append(None); continue

        # check if any contour points contains link area
        for r in np.arange(0.5, max_r, step_r):
            # count points on link pixels
            top_link_pts = 0
            for i in range(len(cp_top)-1):
                pt_1 = cp_top[i]; pt_2 = cp_top[i+1]
                mid_pt_x = int((pt_1[0] + pt_2[0])/2); mid_pt_y = int((pt_1[1] + pt_2[1])/2)
                if linkmap[mid_pt_y, mid_pt_x] == 1: top_link_pts +=1
            bot_link_pts = 0
            for i in range(len(cp_bot)-1):
                pt_1 = cp_bot[i]; pt_2 = cp_bot[i+1]
                mid_pt_x = int((pt_1[0] + pt_2[0])/2); mid_pt_y = int((pt_1[1] + pt_2[1])/2)
                if linkmap[mid_pt_y, mid_pt_x] == 1: bot_link_pts +=1

            # if too many link points > max_len_ratio * num_cp --> skip
            if top_link_pts > max_len_ratio * num_cp or bot_link_pts > max_len_ratio * num_cp:
                polys.append(None); break
            
            else: # check if all contour points are not on link area
                # if there is no link points (i.e. safe points)
                if top_link_pts == 0 and bot_link_pts == 0:
                    poly = np.concatenate((cp_top, np.flip(cp_bot,axis=0)), axis=0)
                    poly = cv2.perspectiveTransform(np.array([poly], dtype=np.float32), Minv)[0]
                    polys.append(poly)
                    break
                # if there is any link points but not too many
                elif top_link_pts <= max_len_ratio * num_cp and bot_link_pts <= max_len_ratio * num_cp:
                    # if not reach the end of expand_ratio range > remove border connected sequences by a score map
                    # get patch image
                    # add padding for patch image (prevent image cropping)
                    left = int(np.min(box[:,0])); right = int(np.max(box[:,0]))
                    top = int(np.min(box[:,1])); bottom = int(np.max(box[:,1]))
                    
                    patch_w = right-left+1; patch_h = top-bottom+1
                    if patch_w < 10 or patch_h < 10: polys.append(None); break # skip if patch is too small
                    
                    patch_img = np.zeros((patch_h, patch_w), dtype=np.uint8)
                    patch_box = box.copy()
                    patch_box[:,0] -= left; patch_box[:,1] -= top;
                    
                    cv2.fillPoly(patch_img, [patch_box.astype(np.int32)], 1) # create mask for patch image
                    
                    # create another mask for link area (NOTE: link area is removed from the mask)
                    patch_linkmap = linkmap[top:bottom+1, left:right+1]
                    patch_img[patch_linkmap==1]=0
                    
                    # find the largest connected component for the mask
                    _, components = cv2.connectedComponents(patch_img, connectivity=4)
                    
                    # if there is no connected component found
                    if components.max() == 0: polys.append(None); break
                    
                    # get the largest component
                    patch_img = (components == (np.bincount(components.flat)[1:].argmax() + 1)).astype(np.uint8)

                    # if the largest component is too small
                    if patch_img.sum() < 10: polys.append(None); break

                    # get contour from the largest component mask
                    contours, _ = cv2.findContours(patch_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    # if there is no contour or too many contours
                    if len(contours) != 1: polys.append(None); break
                    
                    poly = contours[0] # get the only one contour
                    
                    # if the contour is too small (i.e. num of points < 4)
                    if len(poly) < 4 : polys.append(None); break
                    
                    poly = poly.reshape(-1,2) # reshape to (N,2)
                    poly[:,0] += left; poly[:,1] += top # add offset
                    
                    polys.append(poly)
                    break
            
            # if not break yet > continue to next expand_ratio
            if r + step_r >= max_r: polys.append(None); break # if reach the end of expand_ratio range

    return polys

def getDetBoxes(textmap, linkmap, text_threshold, link_threshold, low_text, poly=False):
    boxes, labels, mapper = getDetBoxes_core(textmap, linkmap, text_threshold, link_threshold, low_text)

    if poly and boxes: # Use condition 'boxes' to avoid error when boxes is empty
        polys = getPoly_core(boxes, labels, mapper, linkmap)
    else:
        polys = [None] * len(boxes)

    return boxes, polys

def adjustResultCoordinates(polys, ratio_w, ratio_h, ratio_net=2):
    if len(polys) > 0:
        polys = np.array(polys, dtype=object) # Use dtype=object for lists of arrays of different lengths
        for k in range(len(polys)):
            if polys[k] is not None:
                polys[k] *= (ratio_w * ratio_net, ratio_h * ratio_net)
    return polys

def draw_boxes_on_image(image, boxes_or_polys, color=(0, 255, 0), thickness=2):
    """Draws bounding boxes or polygons on the image."""
    img_copy = image.copy()
    for item in boxes_or_polys:
        if item is None:
            continue
        # Ensure it's a NumPy array of int32
        poly = np.array(item).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(img_copy, [poly], isClosed=True, color=color, thickness=thickness)
    return img_copy


# --- Main Inference Function (adapted from test_net) ---
def perform_inference(net, image_path, text_threshold, link_threshold, low_text, cuda, poly,
                      canvas_size=1280, mag_ratio=1.5, show_time=False, 
                      output_dir=None, viz=False):
    """
    Performs text detection on a single image.
    """
    if show_time:
        start_time = time.time()

    # Read image
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image from {image_path}")
            return None, None
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Assuming model expects RGB
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None, None

    img_resized, target_ratio, size_heatmap = resize_aspect_ratio(image, canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # Preprocessing
    x = normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = x.unsqueeze(0)                          # [c, h, w] to [b, c, h, w]
    
    if cuda:
        x = x.cuda()

    # Forward pass
    with torch.no_grad():
        y, feature = net(x)

    # Make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()

    if show_time:
        print(f"Preprocessing and Inference time: {time.time() - start_time:.4f}s")
        proc_time_start = time.time()

    # Post-processing
    boxes, polys = getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # Coordinate adjustment
    # ratio_net = 2 if the model output is half input size, common in CRAFT
    boxes = adjustResultCoordinates(boxes, ratio_w, ratio_h, ratio_net=2)
    polys = adjustResultCoordinates(polys, ratio_w, ratio_h, ratio_net=2)
    
    for k in range(len(polys)):
        if polys[k] is None:
            polys[k] = boxes[k]

    if show_time:
        print(f"Post-processing time: {time.time() - proc_time_start:.4f}s")

    # Visualization (optional, can be handled outside)
    if viz and output_dir:
        # save score text images
        filename, file_ext = os.path.splitext(os.path.basename(image_path))
        mask_file = os.path.join(output_dir, filename + '_mask.jpg')
        
        # Render results on image
        render_img = score_text.copy()
        render_img = np.hstack((render_img, score_link))
        ret_score_text = cv2.cvtColor(np.uint8(np.clip(render_img,0,1)*255), cv2.COLOR_GRAY2BGR)
        cv2.imwrite(mask_file, ret_score_text)

    return boxes, polys, image # Return original image for drawing

# --- Main Execution ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CRAFT Text Detection Demo for EasyOCR fine-tuned model')
    parser.add_argument('--trained_model', required=True, type=str, help='Path to fine-tuned CRAFT model checkpoint (.pth)')
    parser.add_argument('--image_path', required=True, type=str, help='Path to the input image')
    parser.add_argument('--output_dir', default='./demo_results/', type=str, help='Directory to save output image with detected text')
    
    # Parameters from your test config (or common defaults)
    parser.add_argument('--text_threshold', default=0.7, type=float, help='Text confidence threshold')
    parser.add_argument('--low_text', default=0.4, type=float, help='Text low_text threshold')
    parser.add_argument('--link_threshold', default=0.4, type=float, help='Link confidence threshold')
    parser.add_argument('--canvas_size', default=1280, type=int, help='Image size for inference')
    parser.add_argument('--mag_ratio', default=1.5, type=float, help='Image magnification ratio')
    parser.add_argument('--poly', default=False, action='store_true', help='Enable polygon type detection')
    parser.add_argument('--no_cuda', default=False, action='store_true', help='Disable CUDA')
    parser.add_argument('--show_time', default=False, action='store_true', help='Show processing time')
    parser.add_argument('--viz', default=False, action='store_true', help='Visualize heatmaps')


    args = parser.parse_args()

    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # CUDA setup
    cuda_enabled = not args.no_cuda and torch.cuda.is_available()
    if cuda_enabled:
        print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
        cudnn.benchmark = False # As per EasyOCR's eval.py
    else:
        print("CUDA not available or disabled. Using CPU.")
        if not args.no_cuda and not torch.cuda.is_available():
            print("Note: --no_cuda was not specified, but CUDA is not available on this system.")

    # Load the CRAFT model
    net = CRAFT()
    print(f'Loading weights from checkpoint: {args.trained_model}')
    
    if cuda_enabled:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model)))
        net = net.cuda()
        # net = torch.nn.DataParallel(net) # If trained with DataParallel and not handled in copyStateDict
    else:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model, map_location='cpu')))
    
    net.eval()

    # Perform inference
    print(f"Processing image: {args.image_path}")
    detected_boxes, detected_polys, original_image = perform_inference(
        net,
        args.image_path,
        args.text_threshold,
        args.link_threshold,
        args.low_text,
        cuda_enabled,
        args.poly,
        args.canvas_size,
        args.mag_ratio,
        args.show_time,
        args.output_dir if args.viz else None,
        args.viz
    )

    if detected_polys is not None and original_image is not None:
        # Draw detected polygons (or boxes if polys are None) on the original image
        # Use polys if available, otherwise boxes
        items_to_draw = []
        for i, poly_item in enumerate(detected_polys):
            if poly_item is not None:
                items_to_draw.append(poly_item)
            elif detected_boxes[i] is not None: # Fallback to box if poly is None
                 items_to_draw.append(detected_boxes[i])
        
        # Convert original_image back to BGR for OpenCV saving
        original_image_bgr = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
        result_image = draw_boxes_on_image(original_image_bgr, items_to_draw)

        # Save the result image
        filename = os.path.basename(args.image_path)
        output_filepath = os.path.join(args.output_dir, f"res_{filename}")
        cv2.imwrite(output_filepath, result_image)
        print(f"Result image saved to: {output_filepath}")
    else:
        print("Text detection failed or no text found.")

    print("Demo finished.")

