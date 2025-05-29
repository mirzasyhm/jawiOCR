# data_preprocessing.py
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.layers import Rescaling
import os

IMG_WIDTH = 224 # Default, will be overridden by training.py for ResNet50
IMG_HEIGHT = 224 # Default, will be overridden by training.py for ResNet50
BATCH_SIZE = 32
NUM_CLASSES = 4

def create_datasets(img_size, batch_size, train_dir_path, val_dir_path, test_dir_path=None):
    # ... (rest of the function remains unchanged from your previous version)
    print(f"Loading training images from: {train_dir_path}")
    train_dataset = image_dataset_from_directory(
        train_dir_path,
        labels='inferred',
        label_mode='int',
        image_size=img_size,
        interpolation='nearest',
        batch_size=batch_size,
        shuffle=True,
        seed=42
    )

    print(f"Loading validation images from: {val_dir_path}")
    validation_dataset = image_dataset_from_directory(
        val_dir_path,
        labels='inferred',
        label_mode='int',
        image_size=img_size,
        interpolation='nearest',
        batch_size=batch_size,
        shuffle=False,
        seed=42
    )

    class_names = train_dataset.class_names
    print(f"Class names found: {class_names}")

    # Normalization will be handled by ResNet50's preprocess_input
    # So, the Rescaling(1./255) layer can be removed if preprocess_input is used
    # However, keeping it for consistency before preprocess_input is also fine,
    # as preprocess_input typically handles scaling to -1 to 1 or 0 to 255 based on mode.
    # For this example, we'll apply ResNet's specific preprocessing later.
    # We'll still do basic 0-1 scaling here for consistency if preprocess_input wasn't used.
    
    normalization_layer = Rescaling(1./255) # This scales to [0,1]
    train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y),
                                      num_parallel_calls=tf.data.AUTOTUNE)
    validation_dataset = validation_dataset.map(lambda x, y: (normalization_layer(x), y),
                                                num_parallel_calls=tf.data.AUTOTUNE)


    train_dataset = train_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    validation_dataset = validation_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    test_dataset = None
    if test_dir_path and os.path.exists(test_dir_path):
        print(f"Loading test images from: {test_dir_path}")
        test_dataset = image_dataset_from_directory(
            test_dir_path,
            labels='inferred',
            label_mode='int',
            image_size=img_size,
            interpolation='nearest',
            batch_size=batch_size,
            shuffle=False
        )
        test_dataset = test_dataset.map(lambda x, y: (normalization_layer(x), y), # Apply 0-1 scaling here too
                                        num_parallel_calls=tf.data.AUTOTUNE)
        test_dataset = test_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_dataset, validation_dataset, test_dataset, class_names
