# demo.py
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_preprocess_input
import matplotlib.pyplot as plt
import sys
import os

# --- Configuration ---
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMAGE_SIZE = (IMG_HEIGHT, IMG_WIDTH)
MODEL_PATH = 'best_jawi_orientation_resnet50_old.keras' # Ensure this is your correct model path

CLASS_NAMES = ['0_degrees', '180_degrees', '270_degrees', '90_degrees'] # VERIFY THIS ORDER!

INPUT_IMAGE_COLOR_MODE_FOR_LOADING = 'rgb' # 'rgb' or 'grayscale'

# Directory to save prediction images
PREDICTION_OUTPUT_DIR = "predictions"
if not os.path.exists(PREDICTION_OUTPUT_DIR):
    os.makedirs(PREDICTION_OUTPUT_DIR)
    print(f"Created directory: {PREDICTION_OUTPUT_DIR}")

# --- End Configuration ---

def preprocess_image(img_path):
    try:
        img = image.load_img(
            img_path,
            target_size=IMAGE_SIZE,
            color_mode=INPUT_IMAGE_COLOR_MODE_FOR_LOADING
        )
    except FileNotFoundError:
        print(f"Error: Image file not found at {img_path}")
        return None
    except Exception as e:
        print(f"Error loading image {img_path}: {e}")
        return None

    img_array = image.img_to_array(img)

    # Corrected logic: If input images are specified as grayscale, convert to pseudo-RGB
    if INPUT_IMAGE_COLOR_MODE_FOR_LOADING == 'grayscale':
        if img_array.shape[-1] == 1:
            img_array = tf.image.grayscale_to_rgb(tf.convert_to_tensor(img_array)).numpy()

    if img_array.shape[-1] != 3:
        print(f"Error: Image array must have 3 channels for ResNet50. Found shape: {img_array.shape} for {img_path}")
        return None

    img_array_batched = np.expand_dims(img_array, axis=0)
    preprocessed_img_array = resnet50_preprocess_input(img_array_batched.copy())
    
    return preprocessed_img_array

def predict_orientation(model, img_path_to_predict, show_and_save_image=True):
    processed_image = preprocess_image(img_path_to_predict)
    if processed_image is None:
        return

    predictions = model.predict(processed_image)
    predicted_index = np.argmax(predictions[0])
    
    try:
        predicted_class = CLASS_NAMES[predicted_index]
    except IndexError:
        print(f"Error: Predicted index {predicted_index} is out of range for CLASS_NAMES.")
        return

    confidence = np.max(predictions[0]) * 100

    base_filename = os.path.basename(img_path_to_predict)
    filename_no_ext, ext = os.path.splitext(base_filename)
    save_filename = f"{filename_no_ext}_prediction.png"
    save_path = os.path.join(PREDICTION_OUTPUT_DIR, save_filename)


    print(f"\n--- Prediction for: {base_filename} ---")
    print(f"Predicted Orientation: {predicted_class}")
    print(f"Confidence: {confidence:.2f}%")

    if show_and_save_image:
        try:
            # Load the original image again for display (not the preprocessed one)
            img_display = image.load_img(img_path_to_predict)
            
            plt.figure(figsize=(7, 7)) # Slightly larger figure
            plt.imshow(img_display)
            plt.title(f"Input: {base_filename}\nPredicted: {predicted_class} ({confidence:.2f}%)", fontsize=10)
            plt.axis('off')
            
            # Save the figure
            plt.savefig(save_path)
            print(f"Prediction figure saved to: {save_path}")
            
            plt.show() # Attempt to show it inline as well
            plt.close() # Close the figure to free memory after showing/saving

        except Exception as e:
            print(f"Could not display or save image figure for {img_path_to_predict}: {e}")

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        return
    
    print(f"Loading model from {MODEL_PATH}...")
    try:
        # When loading models with custom components or mixed precision,
        # sometimes you might need to pass custom_objects or ensure the policy is set.
        # For standard ResNet50 from applications and global mixed precision, it's often fine.
        # If issues, ensure mixed precision policy is set before loading:
        # tf.keras.mixed_precision.set_global_policy('mixed_float16')
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
    
    # Optional: Set mixed precision policy if model was saved with it and loading fails
    # try:
    #     policy = tf.keras.mixed_precision.Policy('mixed_float16')
    #     tf.keras.mixed_precision.set_global_policy(policy)
    #     print("Global mixed precision policy set to 'mixed_float16'.")
    # except Exception as e:
    #     print(f"Could not set mixed precision policy: {e}")

    main()
