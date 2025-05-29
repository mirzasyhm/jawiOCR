# demo.py
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import sys
import os

# --- Configuration ---
# These should match your training configuration
IMG_HEIGHT = 128
IMG_WIDTH = 128
IMAGE_SIZE = (IMG_HEIGHT, IMG_WIDTH)
MODEL_PATH = 'best_jawi_orientation_classifier_mixed_precision.keras' # Or 'jawi_orientation_classifier_mixed_precision.keras'

# !!! IMPORTANT: CLASS NAMES !!!
# This list MUST match the order of subdirectories in your 'train' folder,
# sorted alphabetically, as used by image_dataset_from_directory.
# Example: If your folders were '0_degrees', '90_degrees', '180_degrees', '270_degrees'
# then the order would be: ['0_degrees', '180_degrees', '270_degrees', '90_degrees'] (alphabetical sort)
# Please verify this order based on your 'jawi_orientation_dataset/train/' subfolders.
# You can check by running: print(sorted(os.listdir('jawi_orientation_dataset/train')))
CLASS_NAMES = ['0_degrees', '180_degrees', '270_degrees', '90_degrees'] # ADJUST THIS IF NECESSARY!

# If you trained with grayscale images, set this to 'grayscale'
COLOR_MODE = 'grayscale' # or 'grayscale' if you trained with grayscale images

# --- End Configuration ---

def preprocess_image(img_path):
    """
    Loads and preprocesses an image for model prediction.
    """
    try:
        img = image.load_img(
            img_path,
            target_size=IMAGE_SIZE,
            color_mode=COLOR_MODE
        )
    except FileNotFoundError:
        print(f"Error: Image file not found at {img_path}")
        return None
    except Exception as e:
        print(f"Error loading image {img_path}: {e}")
        return None

    img_array = image.img_to_array(img)
    
    # Normalize pixel values to [0, 1] if your model expects this
    # This was done in your training script with Rescaling(1./255)
    img_array = img_array / 255.0
    
    # Add batch dimension (model expects batch_size, height, width, channels)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_orientation(model, img_path_to_predict, show_image=True):
    """
    Predicts the orientation of an image and optionally displays it.
    """
    processed_image = preprocess_image(img_path_to_predict)
    if processed_image is None:
        return

    # Make predictions
    # If using mixed precision, predictions are usually float32
    predictions = model.predict(processed_image)
    
    # Get the index of the class with the highest probability
    predicted_index = np.argmax(predictions[0])
    
    # Get the predicted class name and confidence
    try:
        predicted_class = CLASS_NAMES[predicted_index]
    except IndexError:
        print(f"Error: Predicted index {predicted_index} is out of range for CLASS_NAMES (length {len(CLASS_NAMES)}).")
        print("Please ensure CLASS_NAMES in demo.py matches your model's output classes and order.")
        return

    confidence = np.max(predictions[0]) * 100 # Convert to percentage

    print(f"\n--- Prediction for: {os.path.basename(img_path_to_predict)} ---")
    print(f"Predicted Orientation: {predicted_class}")
    print(f"Confidence: {confidence:.2f}%")

    if show_image:
        try:
            img_display = image.load_img(img_path_to_predict) # Load original for display
            plt.figure(figsize=(6, 6))
            plt.imshow(img_display)
            plt.title(f"Predicted: {predicted_class} ({confidence:.2f}%)")
            plt.axis('off')
            plt.show()
        except Exception as e:
            print(f"Could not display image: {e}")

def main():
    # Load the trained model
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        print("Please ensure the model path is correct and the model file exists.")
        return
    
    print(f"Loading model from {MODEL_PATH}...")
    try:
        # If your model was saved with mixed precision and uses custom layers not automatically handled,
        # you might need to provide custom_objects. For standard Keras layers, it's usually fine.
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Get image path from command line arguments
    if len(sys.argv) < 2:
        print("\nUsage: python demo.py <path_to_image1> [path_to_image2 ...]")
        # Example for testing if no argument is provided:
        # Create a dummy image file named 'dummy_test_image.png' in the same directory
        # or replace with a path to an actual test image.
        # print("\nNo image path provided. You can test with a placeholder like:")
        # print("python demo.py your_image.jpg")
        return

    image_paths = sys.argv[1:]
    for img_path in image_paths:
        if not os.path.exists(img_path):
            print(f"Skipping: Image file not found at {img_path}")
            continue
        predict_orientation(model, img_path, show_image=True)

if __name__ == '__main__':
    # Check TensorFlow version and GPU availability (optional, for info)
    print(f"TensorFlow version: {tf.__version__}")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"Found GPUs: {gpus}")
    else:
        print("No GPU found by TensorFlow.")
    
    main()
