# data_preprocessing.py
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
# Removed: from tensorflow.keras.layers import Rescaling
import os

IMG_WIDTH = 224  # Default, will be overridden by training.py for ResNet50
IMG_HEIGHT = 224 # Default, will be overridden by training.py for ResNet50
BATCH_SIZE = 256
NUM_CLASSES = 4

def create_datasets(img_size, batch_size, train_dir_path, val_dir_path, test_dir_path=None):
    print(f"Loading training images from: {train_dir_path}")
    train_dataset = image_dataset_from_directory(
        train_dir_path,
        labels='inferred',
        label_mode='int',
        color_mode='rgb',  # Explicitly set to RGB (default)
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
        color_mode='rgb',  # Explicitly set to RGB (default)
        image_size=img_size,
        interpolation='nearest',
        batch_size=batch_size,
        shuffle=False,
        seed=42
    )

    class_names = train_dataset.class_names
    print(f"Class names found: {class_names}")

    # REMOVED Rescaling layer. Images will remain in [0, 255] range.
    # ResNet50's preprocess_input will be applied in training.py
    # normalization_layer = Rescaling(1./255)
    # train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y),
    #                                   num_parallel_calls=tf.data.AUTOTUNE)
    # validation_dataset = validation_dataset.map(lambda x, y: (normalization_layer(x), y),
    #                                             num_parallel_calls=tf.data.AUTOTUNE)

    # Optimize data loading pipeline
    train_dataset = train_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    validation_dataset = validation_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    test_dataset = None
    if test_dir_path and os.path.exists(test_dir_path):
        print(f"Loading test images from: {test_dir_path}")
        test_dataset = image_dataset_from_directory(
            test_dir_path,
            labels='inferred',
            label_mode='int',
            color_mode='rgb',  # Explicitly set to RGB (default)
            image_size=img_size,
            interpolation='nearest',
            batch_size=batch_size,
            shuffle=False
        )
        # REMOVED Rescaling for test_dataset as well
        # test_dataset = test_dataset.map(lambda x, y: (normalization_layer(x), y),
        #                                 num_parallel_calls=tf.data.AUTOTUNE)
        test_dataset = test_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_dataset, validation_dataset, test_dataset, class_names
