# demo.py
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_preprocess_input # Import ResNet50's preprocessor
import matplotlib.pyplot as plt
import sys
import os

# --- Configuration ---
IMG_HEIGHT = 224 # MUST MATCH ResNet50's expected input
IMG_WIDTH = 224  # MUST MATCH ResNet50's expected input
IMAGE_SIZE = (IMG_HEIGHT, IMG_WIDTH)
MODEL_PATH = 'best_jawi_orientation_resnet50_old.keras' # Or your ResNet50 model path

# CLASS_NAMES - CRITICAL: Verify this matches your training subfolder order (alphabetical)
CLASS_NAMES = ['0_degrees', '180_degrees', '270_degrees', '90_degrees'] # ADJUST IF NECESSARY!

# Whether your original input images for the demo are grayscale or RGB.
# The model ultimately needs 3 channels.
INPUT_IMAGE_COLOR_MODE_FOR_LOADING = 'rgb' # Use 'grayscale' if your test images are grayscale,
                                         # then we'll convert to pseudo-RGB.
                                         # Use 'rgb' if test images are already color.

# --- End Configuration ---

def preprocess_image(img_path):
    """
    Loads and preprocesses an image for ResNet50 model prediction.
    """
    try:
        # Load image, ensuring it's the correct size for ResNet50
        img = image.load_img(
            img_path,
            target_size=IMAGE_SIZE,
            color_mode=INPUT_IMAGE_COLOR_MODE_FOR_LOADING # Load as specified
        )
    except FileNotFoundError:
        print(f"Error: Image file not found at {img_path}")
        return None
    except Exception as e:
        print(f"Error loading image {img_path}: {e}")
        return None

    img_array = image.img_to_array(img) # Converts to NumPy array

    # If original input was grayscale, convert to pseudo-RGB (3 channels)
    if INPUT_IMAGE_COLOR_MODE_FOR_LOADING == 'rgb':
        if img_array.shape[-1] == 1: # Check if it has 1 channel
             # img_array = np.repeat(img_array, 3, axis=-1) # Manual repeat
            img_array = tf.image.grayscale_to_rgb(tf.convert_to_tensor(img_array)).numpy() # Using TF utility

    # Ensure img_array is (height, width, 3) before resnet50_preprocess_input
    if img_array.shape[-1] != 3:
        print(f"Error: Image array does not have 3 channels after potential conversion. Shape: {img_array.shape}")
        return None

    # Add batch dimension (model expects batch_size, height, width, channels)
    img_array_batched = np.expand_dims(img_array, axis=0)
    
    # Apply ResNet50 specific preprocessing
    # This function expects pixel values in the range [0, 255] and 3 channels.
    # It will scale/normalize them as ResNet50 expects (typically to -1 to 1 or similar).
    preprocessed_img_array = resnet50_preprocess_input(img_array_batched.copy()) # Use .copy() if any in-place ops
    
    return preprocessed_img_array

def predict_orientation(model, img_path_to_predict, show_image=True):
    """
    Predicts the orientation of an image and optionally displays it.
    """
    processed_image = preprocess_image(img_path_to_predict)
    if processed_image is None:
        return

    # Sanity check the shape before prediction
    # print(f"Shape of processed image going into model.predict: {processed_image.shape}") # Should be (1, 224, 224, 3)

    predictions = model.predict(processed_image)
    
    predicted_index = np.argmax(predictions[0])
    
    try:
        predicted_class = CLASS_NAMES[predicted_index]
    except IndexError:
        print(f"Error: Predicted index {predicted_index} is out of range for CLASS_NAMES.")
        return

    confidence = np.max(predictions[0]) * 100

    print(f"\n--- Prediction for: {os.path.basename(img_path_to_predict)} ---")
    print(f"Predicted Orientation: {predicted_class}")
    print(f"Confidence: {confidence:.2f}%")

    if show_image:
        try:
            img_display = image.load_img(img_path_to_predict)
            plt.figure(figsize=(6, 6))
            plt.imshow(img_display)
            plt.title(f"Predicted: {predicted_class} ({confidence:.2f}%)")
            plt.axis('off')
            plt.show()
        except Exception as e:
            print(f"Could not display image: {e}")

def main():
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
        predict_orientation(model, img_path, show_image=True)

if __name__ == '__main__':
    print(f"TensorFlow version: {tf.__version__}")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus: print(f"Found GPUs: {gpus}")
    else: print("No GPU found by TensorFlow.")
    main()
