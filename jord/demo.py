# demo.py
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_preprocess_input
import matplotlib.pyplot as plt
from PIL import Image # For image rotation
import sys
import os

# --- Configuration ---
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMAGE_SIZE = (IMG_HEIGHT, IMG_WIDTH)
MODEL_PATH = 'best_jawi_orientation_resnet50.keras'

CLASS_NAMES = ['0_degrees', '180_degrees', '270_degrees', '90_degrees'] # VERIFY THIS ORDER!
TARGET_180_CLASS_NAME = '180_degrees' # The string name for 180 degrees class
CONFIDENCE_THRESHOLD_FOR_REPREDICT = 70.0 # Percentage
ROTATION_ANGLE_FOR_REPREDICT = 45 # Degrees

INPUT_IMAGE_COLOR_MODE_FOR_LOADING = 'rgb'

PREDICTION_OUTPUT_DIR = "predictions_v2" # Changed output dir name
if not os.path.exists(PREDICTION_OUTPUT_DIR):
    os.makedirs(PREDICTION_OUTPUT_DIR)
    print(f"Created directory: {PREDICTION_OUTPUT_DIR}")
# --- End Configuration ---

def preprocess_image_pil(pil_image, target_size, color_mode_for_model='rgb'):
    """
    Preprocesses a PIL Image object for ResNet50 model prediction.
    Ensures image is converted to RGB for the model if needed.
    """
    # Ensure image is in RGB format if the model expects it
    if color_mode_for_model == 'rgb' and pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')
    elif color_mode_for_model == 'grayscale' and pil_image.mode != 'L':
        pil_image = pil_image.convert('L')


    # Resize (Pillow's resize is used here, could also use tf.image.resize after converting to tensor)
    # Using thumbnail to maintain aspect ratio and then padding or cropping could be more robust,
    # but for simplicity, direct resize is used.
    # ResNet50 expects specific input size.
    # img_resized = pil_image.resize(target_size, Image.Resampling.LANCZOS) # High quality resampling

    # Keras image.img_to_array expects a PIL image, let's convert back after rotation
    # then use our existing keras-based preprocessing chain.
    # The target_size is mainly for the initial load_img if used.
    # Here, the PIL image is already loaded. We just need to make it a numpy array.
    
    img_array = image.img_to_array(pil_image) # Converts PIL image to NumPy array

    # If original input was grayscale and converted to RGB, or if input PIL was L mode and model needs RGB
    if color_mode_for_model == 'rgb' and img_array.shape[-1] == 1: # Check if it has 1 channel after conversion
        img_array = tf.image.grayscale_to_rgb(tf.convert_to_tensor(img_array)).numpy()

    if color_mode_for_model == 'rgb' and img_array.shape[-1] != 3:
        print(f"Error: Rotated image array must have 3 channels for ResNet50. Found shape: {img_array.shape}")
        return None
    elif color_mode_for_model == 'grayscale' and img_array.shape[-1] !=1:
        print(f"Error: Rotated image array must have 1 channel for grayscale model. Found shape: {img_array.shape}")
        return None


    img_array_batched = np.expand_dims(img_array, axis=0)
    preprocessed_img_array = resnet50_preprocess_input(img_array_batched.copy())
    
    return preprocessed_img_array


def load_and_preprocess_initial(img_path):
    """Loads and preprocesses an image from path for the first prediction."""
    try:
        img_pil = Image.open(img_path) # Load with PIL to keep original for rotation
    except FileNotFoundError:
        print(f"Error: Image file not found at {img_path}")
        return None, None
    except Exception as e:
        print(f"Error loading image {img_path} with PIL: {e}")
        return None, None

    # Resize the PIL image to the target input size of the model *before* any preprocessing
    # This ensures consistency.
    img_pil_resized = img_pil.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)

    processed_for_model = preprocess_image_pil(img_pil_resized, IMAGE_SIZE, 'rgb') # Assuming model always wants RGB

    if processed_for_model is None:
        return None, None # Preprocessing failed

    return img_pil_resized, processed_for_model # Return original PIL (resized) and processed numpy array


def predict_orientation(model, img_path_to_predict, show_and_save_image=True):
    # --- First Prediction ---
    original_pil_image_resized, processed_image_pass1 = load_and_preprocess_initial(img_path_to_predict)
    
    if original_pil_image_resized is None or processed_image_pass1 is None:
        return

    predictions_pass1 = model.predict(processed_image_pass1)
    predicted_index_pass1 = np.argmax(predictions_pass1[0])
    confidence_pass1 = np.max(predictions_pass1[0]) * 100
    
    try:
        predicted_class_pass1 = CLASS_NAMES[predicted_index_pass1]
    except IndexError:
        print(f"Error (Pass 1): Predicted index {predicted_index_pass1} out of range for CLASS_NAMES.")
        return

    final_predicted_class = predicted_class_pass1
    final_confidence = confidence_pass1
    prediction_stage = "Pass 1"

    # --- Check for Second Prediction ---
    needs_second_pass = False
    if predicted_class_pass1 == TARGET_180_CLASS_NAME:
        needs_second_pass = True
        print(f"Info: First prediction is '{TARGET_180_CLASS_NAME}'. Proceeding to 2nd pass with 45-degree rotation.")
    elif confidence_pass1 < CONFIDENCE_THRESHOLD_FOR_REPREDICT:
        needs_second_pass = True
        print(f"Info: First prediction confidence ({confidence_pass1:.2f}%) is below threshold ({CONFIDENCE_THRESHOLD_FOR_REPREDICT}%). Proceeding to 2nd pass with 45-degree rotation.")

    if needs_second_pass:
        try:
            # Rotate the *original resized* PIL image by 45 degrees
            # fillcolor='white' or 'black' can be used if images have consistent background
            rotated_pil_image = original_pil_image_resized.rotate(ROTATION_ANGLE_FOR_REPREDICT, expand=True, fillcolor=(255,255,255)) # White fill
            
            # The rotated image might have different dimensions due to 'expand=True'.
            # We need to resize it again to the model's input size.
            rotated_pil_image_resized = rotated_pil_image.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)

            processed_image_pass2 = preprocess_image_pil(rotated_pil_image_resized, IMAGE_SIZE, 'rgb')

            if processed_image_pass2 is not None:
                predictions_pass2 = model.predict(processed_image_pass2)
                predicted_index_pass2 = np.argmax(predictions_pass2[0])
                confidence_pass2 = np.max(predictions_pass2[0]) * 100
                
                try:
                    predicted_class_pass2 = CLASS_NAMES[predicted_index_pass2]
                    print(f"Info (Pass 2 on 45deg rotated): Predicted '{predicted_class_pass2}' with {confidence_pass2:.2f}% confidence.")
                    # Decision: Use the second prediction as the final one if a second pass was made.
                    # You could add more sophisticated logic here, e.g., only if confidence_pass2 > confidence_pass1
                    final_predicted_class = predicted_class_pass2
                    final_confidence = confidence_pass2
                    prediction_stage = "Pass 2 (after 45deg rot)"
                except IndexError:
                    print(f"Error (Pass 2): Predicted index {predicted_index_pass2} out of range for CLASS_NAMES.")
                    # Stick with pass 1 result if pass 2 errors
            else:
                print("Warning: Preprocessing failed for the rotated image. Sticking with Pass 1 prediction.")
        except Exception as e:
            print(f"Error during 2nd pass rotation/prediction: {e}. Sticking with Pass 1 prediction.")


    base_filename = os.path.basename(img_path_to_predict)
    filename_no_ext, ext = os.path.splitext(base_filename)
    save_filename = f"{filename_no_ext}_pred_{final_predicted_class.replace('_','-')}_{prediction_stage.replace(' ','_')}.png"
    save_path = os.path.join(PREDICTION_OUTPUT_DIR, save_filename)

    print(f"\n--- Final Prediction for: {base_filename} ({prediction_stage}) ---")
    print(f"Predicted Orientation: {final_predicted_class}")
    print(f"Confidence: {final_confidence:.2f}%")

    if show_and_save_image:
        try:
            # Display the original image that was passed to the demo
            img_display_original = Image.open(img_path_to_predict)
            
            plt.figure(figsize=(7, 7))
            plt.imshow(img_display_original)
            title_text = f"Input: {base_filename}\nFinal Pred: {final_predicted_class} ({final_confidence:.2f}%) [{prediction_stage}]"
            if needs_second_pass and prediction_stage.startswith("Pass 2"):
                title_text += f"\n(Pass 1 was: {predicted_class_pass1} @ {confidence_pass1:.2f}%)"
            plt.title(title_text, fontsize=9)
            plt.axis('off')
            
            plt.savefig(save_path)
            print(f"Prediction figure saved to: {save_path}")
            
            plt.show()
            plt.close()

        except Exception as e:
            print(f"Could not display or save image figure for {img_path_to_predict}: {e}")

def main():
    # ... (model loading, arg parsing - same as before) ...
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        return
    
    print(f"Loading model from {MODEL_PATH}...")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    if len(sys.argv) < 2:
        print("\nUsage: python demo.py <path_to_image1> [path_to_image2 ...]")
        return

    image_paths = sys.argv[1:]
    for img_path in image_paths:
        if not os.path.exists(img_path):
            print(f"Skipping: Image file not found at {img_path}")
            continue
        predict_orientation(model, img_path, show_and_save_image=True)

if __name__ == '__main__':
    print(f"TensorFlow version: {tf.__version__}")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus: print(f"Found GPUs: {gpus}")
    else: print("No GPU found by TensorFlow.")
    main()
