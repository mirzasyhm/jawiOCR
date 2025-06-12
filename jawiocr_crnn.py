# jawiocr_crnn.py

import os
import sys
import argparse
import time
import cv2
import json
import math
import numpy as np
from collections import OrderedDict

import torch
import torch.backends.cudnn as cudnn
from PIL import Image as PILImage
from torchvision import transforms as TorchTransforms

# TensorFlow imports for optional orientation model
try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing import image as tf_keras_image
    from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_preprocess_input
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("Warning: TensorFlow not found. Custom orientation model functionality will be disabled.")


# --- Path Setup ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))
# Assumes 'craft' and 'crnn' directories are in the same folder as this script
craft_module_dir = os.path.join(current_script_dir, 'craft')
crnn_module_dir = os.path.join(current_script_dir, 'crnn')

if craft_module_dir not in sys.path: sys.path.insert(0, craft_module_dir)
if crnn_module_dir not in sys.path: sys.path.insert(0, crnn_module_dir)


# --- CRAFT Model Import and Utilities ---
# These functions are for text detection and are adapted from the provided scripts [1, 2].
try:
    from model import CRAFT
except ImportError as e:
    print(f"Error importing 'CRAFT' from 'model.craft': {e}\n"
          f"Ensure the 'craft' submodule is structured correctly and accessible from sys.path.")
    sys.exit(1)

def copyStateDict(state_dict):
    """Handles model state dictionaries saved with 'module.' prefix."""
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v
    return new_state_dict

def normalizeMeanVariance(in_img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
    """Normalizes an image for CRAFT model input."""
    img = in_img.copy().astype(np.float32)
    img -= np.array([mean[0] * 255.0, mean[1] * 255.0, mean[2] * 255.0], dtype=np.float32)
    img /= np.array([variance[0] * 255.0, variance[1] * 255.0, variance[2] * 255.0], dtype=np.float32)
    return img

def resize_aspect_ratio(img, square_size, interpolation, mag_ratio=1.):
    """Resizes an image while maintaining aspect ratio for CRAFT."""
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

def getDetBoxes(textmap, linkmap, text_threshold, link_threshold, low_text):
    """Extracts bounding boxes from CRAFT's text and link score maps."""
    linkmap_vis, textmap_vis = linkmap.copy(), textmap.copy()
    img_h, img_w = textmap.shape
    ret, text_score = cv2.threshold(textmap_vis, low_text, 1, 0)
    ret, link_score = cv2.threshold(linkmap_vis, link_threshold, 1, 0)
    text_score_comb = np.clip(text_score + link_score, 0, 1)
    nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(text_score_comb.astype(np.uint8), connectivity=4)
    det = []
    for k in range(1, nLabels):
        size = stats[k, cv2.CC_STAT_AREA]
        if size < 10: continue
        if np.max(textmap_vis[labels == k]) < text_threshold: continue
        segmap = np.zeros(textmap.shape, dtype=np.uint8)
        segmap[labels == k] = 255
        segmap[np.logical_and(link_score == 1, text_score == 0)] = 0
        x, y = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP]
        w, h = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
        niter = int(math.sqrt(size * min(w, h) / (w * h))) * 2 if w > 0 and h > 0 else 0
        sx, ex, sy, ey = x - niter, x + w + niter + 1, y - niter, y + h + niter + 1
        sx, ex, sy, ey = max(0, sx), min(img_w, ex), max(0, sy), min(img_h, ey)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1 + niter, 1 + niter))
        segmap[sy:ey, sx:ex] = cv2.dilate(segmap[sy:ey, sx:ex], kernel)
        np_contours = np.roll(np.array(np.where(segmap != 0)), 1, axis=0).transpose().reshape(-1, 2)
        if np_contours.size == 0: continue
        rectangle = cv2.minAreaRect(np_contours)
        box = cv2.boxPoints(rectangle)
        w_box, h_box = np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2])
        if abs(1 - max(w_box, h_box) / (min(w_box, h_box) + 1e-5)) <= 0.1:
            l, t = (min(np_contours[:, 0]), min(np_contours[:, 1]))
            r, b = (max(np_contours[:, 0]), max(np_contours[:, 1]))
            box = np.array([[l, t], [r, t], [r, b], [l, b]], dtype=np.float32)
        startidx = box.sum(axis=1).argmin()
        box = np.roll(box, 4 - startidx, 0)
        det.append(box)
    return det

def perform_craft_inference(net, image_bgr, text_threshold, link_threshold, low_text, cuda, canvas_size=1280, mag_ratio=1.5):
    """Runs a full text detection pass with CRAFT."""
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img_resized, target_ratio, _ = resize_aspect_ratio(image_rgb, canvas_size, cv2.INTER_LINEAR, mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio
    x = normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)
    if cuda: x = x.cuda()
    with torch.no_grad():
        y, _ = net(x)
    score_text = y[0, :, :, 0].cpu().data.numpy()
    score_link = y[0, :, :, 1].cpu().data.numpy()
    boxes = getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text)
    # Scale coordinates back to original image size
    final_polys = []
    for box in boxes:
        poly = np.array(box).astype(np.int32)
        # CRAFT maps are 1/2 size of network input; scale by 2 and then by inverse resize ratio
        scaled_poly = (poly * 2 / target_ratio)
        final_polys.append(scaled_poly)
    return final_polys


# --- CRNN Model Import and Utilities ---
# These functions are for text recognition using the CRNN model [2].
try:
    from model import CRNN
except ImportError:
    print("Error: Could not import CRNN from model.py. Ensure the 'crnn' directory and its 'model.py' are in the Python path.")
    sys.exit(1)

# Define CRNN default parameters
CRNN_IMG_HEIGHT = 32
CRNN_IMG_WIDTH = 128
CRNN_NUM_CHANNELS = 1 # CRNN model expects grayscale images

def load_crnn_model(model_path, alphabet_path, device):
    """Loads a pre-trained CRNN model and its associated alphabet."""
    print(f"Loading CRNN model from: {model_path}")
    print(f"Loading alphabet from: {alphabet_path}")
    if not os.path.exists(alphabet_path):
        print(f"CRITICAL Error: Alphabet file '{alphabet_path}' not found.")
        sys.exit(1)
    try:
        with open(alphabet_path, 'r', encoding='utf-8') as f:
            alphabet_chars = json.load(f)
    except Exception as e:
        print(f"CRITICAL Error: Could not load or parse alphabet file: {e}")
        sys.exit(1)

    n_class = len(alphabet_chars) + 1  # +1 for CTC blank character
    model = CRNN(imgH=CRNN_IMG_HEIGHT, nc=CRNN_NUM_CHANNELS, nclass=n_class, nh=256)
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"CRITICAL Error loading CRNN state_dict: {e}")
        sys.exit(1)

    model = model.to(device).eval()
    
    # Transformation pipeline for CRNN input images
    transform = TorchTransforms.Compose([
        TorchTransforms.ToPILImage(),
        TorchTransforms.Resize((CRNN_IMG_HEIGHT, CRNN_IMG_WIDTH)),
        TorchTransforms.Grayscale(num_output_channels=CRNN_NUM_CHANNELS),
        TorchTransforms.ToTensor(),
        TorchTransforms.Normalize(mean=[0.5], std=[0.5])
    ])
    print("CRNN model, alphabet, and transform loaded successfully.")
    return model, transform, alphabet_chars

def preprocess_for_crnn(img_crop_bgr, crnn_transform, device):
    """Preprocesses a single image crop for CRNN inference."""
    if img_crop_bgr is None or img_crop_bgr.shape[0] == 0 or img_crop_bgr.shape[1] == 0:
        return None
    img_tensor = crnn_transform(img_crop_bgr).unsqueeze(0)
    return img_tensor.to(device)

def decode_crnn_output(log_probs, alphabet, beam_size=20):
    """Decodes CRNN raw output using a CTC beam search decoder."""
    # This function requires torchaudio to be installed.
    try:
        from torchaudio.models.decoder import ctc_decoder
    except ImportError:
        print("Error: torchaudio is required for decoding. Please install it (`pip install torchaudio`).")
        return "DECODER_ERROR", 0.0

    log_probs_for_decoder = log_probs.permute(1, 0, 2) # T, N, C -> required by decoder
    blank_token = "-"
    # Ensure alphabet for decoder does not contain the blank token itself
    decoder_alphabet = [char for char in alphabet if char != blank_token]
    
    decoder = ctc_decoder(
        lexicon=None,
        tokens=[blank_token] + decoder_alphabet,
        beam_size=beam_size,
        blank_token=blank_token,
        sil_token=blank_token,
        nbest=1,
        log_add=True
    )

    hypotheses = decoder(log_probs_for_decoder.cpu())
    if not hypotheses or not hypotheses[0]:
        return "", 0.0

    best_hypothesis = hypotheses[0][0]
    confidence = math.exp(best_hypothesis.score)
    text = "".join(decoder.idxs_to_tokens(best_hypothesis.tokens))
    
    return text, confidence


# --- Image Utilities (Rotation, Cropping, Orientation) ---
# These utility functions are adapted from the provided scripts [1, 2].
def rotate_image_cv(image_cv, angle_degrees):
    """Rotates an OpenCV image by 90, 180, or 270 degrees."""
    if angle_degrees == 90: return cv2.rotate(image_cv, cv2.ROTATE_90_CLOCKWISE)
    if angle_degrees == 180: return cv2.rotate(image_cv, cv2.ROTATE_180)
    if angle_degrees == 270: return cv2.rotate(image_cv, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image_cv

def load_custom_orientation_model_keras(model_path):
    """Loads a custom Keras model for page orientation detection."""
    if not TF_AVAILABLE:
        print("Cannot load Keras model: TensorFlow is not installed.")
        return None
    print(f"Loading Keras orientation model from {model_path}...")
    try:
        model = tf.keras.models.load_model(model_path)
        print("Keras orientation model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading Keras orientation model: {e}")
        return None

def get_custom_page_orientation(orientation_model, image_bgr, class_names, target_img_size=(224, 224)):
    """Predicts the orientation of a full page image."""
    if not TF_AVAILABLE or orientation_model is None:
        return "0_degrees", 1.0
    pil_img = PILImage.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)).resize(target_img_size, PILImage.Resampling.LANCZOS)
    img_array = tf_keras_image.img_to_array(pil_img)
    if img_array.shape[-1] == 1:
        img_array = tf.image.grayscale_to_rgb(tf.convert_to_tensor(img_array)).numpy()
    preprocessed_img = resnet50_preprocess_input(np.expand_dims(img_array, axis=0).copy())
    predictions = orientation_model.predict(preprocessed_img, verbose=0)
    predicted_index = np.argmax(predictions[0])
    confidence = np.max(predictions[0])
    return class_names[predicted_index], confidence

def simple_orientation_correction(image_crop_bgr):
    """Corrects orientation for individual crops that are taller than they are wide."""
    if image_crop_bgr is None or image_crop_bgr.shape[0] == 0 or image_crop_bgr.shape[1] == 0:
        return image_crop_bgr, "invalid_crop"
    h, w = image_crop_bgr.shape[:2]
    if w < h and w > 0:
        return cv2.rotate(image_crop_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE), "rotated_90_ccw"
    return image_crop_bgr, "no_rotation"

def get_cropped_image_from_poly(image_bgr, poly_pts):
    """Crops and rectifies a quadrilateral region from an image."""
    if poly_pts is None or len(poly_pts) != 4: return None
    poly = np.asarray(poly_pts, dtype=np.float32)
    target_w = int(round(max(np.linalg.norm(poly[1] - poly[0]), np.linalg.norm(poly[2] - poly[3]))))
    target_h = int(round(max(np.linalg.norm(poly[3] - poly[0]), np.linalg.norm(poly[2] - poly[1]))))
    if target_w <= 0 or target_h <= 0: return None
    dst_pts = np.array([[0, 0], [target_w-1, 0], [target_w-1, target_h-1], [0, target_h-1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(poly, dst_pts)
    warped_crop = cv2.warpPerspective(image_bgr, M, (target_w, target_h))
    return warped_crop if warped_crop.size else None


# --- Main OCR Pass Function ---
def run_ocr_pass(image_bgr, base_fname, pass_name, args, craft_net, crnn_model, crnn_transform, crnn_alphabet, device, debug_dir):
    """Runs a complete OCR pass: detection, sorting, and recognition."""
    print(f"\n--- Starting OCR Pass: {pass_name} ---")
    
    detected_polys = perform_craft_inference(
        craft_net, image_bgr, args.text_threshold, args.link_threshold,
        args.low_text, (device.type == 'cuda'), args.canvas_size, args.mag_ratio
    )
    print(f"{pass_name} - CRAFT detected {len(detected_polys)} regions.")

    regions = []
    for poly in detected_polys:
        center_x = int(np.mean(poly[:, 0]))
        regions.append({'x': center_x, 'poly': poly})
    regions.sort(key=lambda item: item['x'], reverse=True) # Sort Right-to-Left

    results_data = []
    text_snippets = []
    
    for i, region in enumerate(regions):
        cropped_bgr = get_cropped_image_from_poly(image_bgr, region['poly'])
        if cropped_bgr is None: continue

        crop_for_crnn = cropped_bgr
        if args.use_simple_orientation:
            crop_for_crnn, _ = simple_orientation_correction(cropped_bgr)

        crnn_input = preprocess_for_crnn(crop_for_crnn, crnn_transform, device)
        if crnn_input is None: continue
        
        with torch.no_grad():
            raw_preds = crnn_model(crnn_input)
            log_probs = raw_preds.log_softmax(2)
            text_seg, conf = decode_crnn_output(log_probs, crnn_alphabet, args.beam_size)
            
            if text_seg:
                print(f"{pass_name} Region {i+1}: Text='{text_seg}', Conf={conf:.4f}")
                results_data.append({'poly': region['poly'], 'text': text_seg, 'conf': conf})
                text_snippets.append(text_seg)
            
    avg_confidence = np.mean([res['conf'] for res in results_data if res['conf'] is not None]) if results_data else 0.0
    final_text = " ".join(text_snippets)
    print(f"{pass_name} - Avg Confidence: {avg_confidence:.4f}, Combined Text: {final_text}")
    return final_text, avg_confidence, results_data


# --- Main OCR Pipeline ---
def main_ocr_pipeline(args):
    """Orchestrates the entire OCR process from image loading to final output."""
    cuda_enabled = not args.no_cuda and torch.cuda.is_available()
    pytorch_device = torch.device('cuda' if cuda_enabled else 'cpu')
    print(f"Using PyTorch device: {pytorch_device}")

    # Load CRAFT text detector
    craft_net = CRAFT()
    print(f'Loading CRAFT model from: {args.craft_model_path}')
    ckpt_craft = torch.load(args.craft_model_path, map_location=pytorch_device)
    craft_net.load_state_dict(copyStateDict(ckpt_craft))
    craft_net.to(pytorch_device).eval()

    # Load CRNN text recognizer
    crnn_model, crnn_transform, crnn_alphabet = load_crnn_model(
        args.crnn_model_path, args.alphabet_path, pytorch_device
    )

    # Load optional Keras orientation model
    orientation_model = None
    if args.custom_orientation_model_path:
        orientation_model = load_custom_orientation_model_keras(args.custom_orientation_model_path)

    # Process image
    print(f"Processing image: {args.image_path}")
    image_bgr_original = cv2.imread(args.image_path)
    if image_bgr_original is None:
        print(f"Error: Could not read image at {args.image_path}")
        return

    # Initial Page Orientation Correction
    image_for_pass1 = image_bgr_original.copy()
    if orientation_model:
        class_names = args.orientation_class_names.split(',')
        pred_class, pred_conf = get_custom_page_orientation(orientation_model, image_bgr_original, class_names)
        print(f"Global Page Orientation Prediction: {pred_class} (Conf: {pred_conf:.2f})")
        if pred_conf * 100 >= args.orientation_confidence_threshold:
            if pred_class == "90_degrees": image_for_pass1 = rotate_image_cv(image_for_pass1, 270)
            elif pred_class == "270_degrees": image_for_pass1 = rotate_image_cv(image_for_pass1, 90)

    # OCR Pass 1
    final_text_pass1, avg_conf_pass1, results_data_pass1 = run_ocr_pass(
        image_for_pass1, os.path.basename(args.image_path), "Pass1", args, 
        craft_net, crnn_model, crnn_transform, crnn_alphabet, pytorch_device, ""
    )

    # Conditional OCR Pass 2 (180-degree rotation)
    chosen_text = final_text_pass1
    chosen_results = results_data_pass1
    chosen_pass_name = "Pass1"
    
    if avg_conf_pass1 * 100 < args.rerun_180_threshold:
        print(f"\nPass 1 conf ({avg_conf_pass1*100:.2f}%) is below threshold ({args.rerun_180_threshold}%). "
              f"Re-running with 180-deg rotation.")
        image_for_pass2 = rotate_image_cv(image_for_pass1, 180)
        
        final_text_pass2, avg_conf_pass2, results_data_pass2 = run_ocr_pass(
            image_for_pass2, os.path.basename(args.image_path), "Pass2_180_Rot", args,
            craft_net, crnn_model, crnn_transform, crnn_alphabet, pytorch_device, ""
        )
        
        if avg_conf_pass2 > avg_conf_pass1:
            print("Pass 2 (180-deg rotated) confidence is higher. Using Pass 2 results.")
            chosen_text = final_text_pass2
            chosen_results = results_data_pass2
            chosen_pass_name = "Pass2_180_Rot"
        else:
            print("Pass 1 confidence is higher. Sticking with Pass 1 results.")

    # Output results
    print(f"\n--- Final Chosen Result (from {chosen_pass_name}) ---")
    print(f"Final Combined Right-to-Left Text: {chosen_text}\n")
    
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        base_fname = os.path.splitext(os.path.basename(args.image_path))[0]
        txt_filepath = os.path.join(args.output_dir, f"res_{base_fname}.txt")
        with open(txt_filepath, 'w', encoding='utf-8') as f:
            f.write(f"Final Combined Text (R-L):\n{chosen_text}\n\n")
            f.write("Individual Detections:\n")
            for res in chosen_results:
                f.write(f"  - Text: {res['text']}, Confidence: {res['conf']:.4f}\n")
        print(f"Final text results saved to: {txt_filepath}")

    print("OCR pipeline finished.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Jawi OCR with CRAFT and CRNN')
    
    # Model and Path Arguments
    parser.add_argument('--image_path', required=True, type=str, help="Path to the input image file.")
    parser.add_argument('--craft_model_path', required=True, type=str, help="Path to the pre-trained CRAFT model (.pth).")
    parser.add_argument('--crnn_model_path', required=True, type=str, help="Path to the pre-trained CRNN model state_dict (.pth).")
    parser.add_argument('--alphabet_path', required=True, type=str, help="Path to the alphabet.json file for the CRNN model.")
    parser.add_argument('--custom_orientation_model_path', type=str, default=None, help="Optional path to a Keras page orientation model (.h5).")
    parser.add_argument('--output_dir', default='./jawi_ocr_crnn_results/', type=str, help="Directory to save output text files.")
    
    # CRAFT Parameters
    parser.add_argument('--text_threshold', default=0.7, type=float, help="Text confidence threshold for CRAFT.")
    parser.add_argument('--low_text', default=0.4, type=float, help="Text low-bound score for CRAFT.")
    parser.add_argument('--link_threshold', default=0.4, type=float, help="Link confidence threshold for CRAFT.")
    parser.add_argument('--canvas_size', default=1280, type=int, help="Maximum image size for CRAFT processing.")
    parser.add_argument('--mag_ratio', default=1.5, type=float, help="Image magnification ratio for CRAFT.")
    
    # CRNN Parameters
    parser.add_argument('--beam_size', type=int, default=20, help="Beam size for the CTC Beam Search Decoder.")

    # Pipeline Control Arguments
    parser.add_argument('--use_simple_orientation', action='store_true', help="Enable simple per-crop orientation correction.")
    parser.add_argument('--rerun_180_threshold', type=float, default=85.0, help="Confidence threshold below which a 180-degree re-run is triggered.")
    parser.add_argument('--orientation_class_names', type=str, default='0_degrees,180_degrees,270_degrees,90_degrees', help="Class names for the Keras orientation model.")
    parser.add_argument('--orientation_confidence_threshold', type=float, default=75.0, help="Confidence threshold for applying global page rotation.")
    
    # System Arguments
    parser.add_argument('--no_cuda', action='store_true', help="Disable CUDA, forcing CPU usage.")
    
    args = parser.parse_args()
    main_ocr_pipeline(args)
