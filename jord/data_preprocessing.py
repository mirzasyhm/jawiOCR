# data_preprocessing.py
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.layers import Rescaling
import os

# Constants for dataset loading
# These can be imported or overridden in training.py if needed,
# but good to have defaults here.
IMG_WIDTH = 128
IMG_HEIGHT = 128
BATCH_SIZE = 64
NUM_CLASSES = 4


def create_datasets(img_size, batch_size, train_dir_path, val_dir_path, test_dir_path=None):
    """
    Loads and preprocesses the training, validation, and optional test datasets.
    """
    print(f"Loading training images from: {train_dir_path}")
    train_dataset = image_dataset_from_directory(
        train_dir_path,
        labels='inferred',
        label_mode='int',
        image_size=img_size,
        color_mode='grayscale',
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
        color_mode='grayscale',
        interpolation='nearest',
        batch_size=batch_size,
        shuffle=False,
        seed=42
    )

    class_names = train_dataset.class_names
    print(f"Class names found: {class_names}")
    # It's good practice to ensure NUM_CLASSES matches what's found in the data
    # This can be done in the training script where NUM_CLASSES is definitively set.
    # if len(class_names) != NUM_CLASSES:
    #     raise ValueError(f"Expected {NUM_CLASSES} classes, but found {len(class_names)} in training data: {class_names}")


    normalization_layer = Rescaling(1./255)
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
            color_mode='grayscale',
            interpolation='nearest',
            batch_size=batch_size,
            shuffle=False # No shuffle for test data
        )
        test_dataset = test_dataset.map(lambda x, y: (normalization_layer(x), y),
                                        num_parallel_calls=tf.data.AUTOTUNE)
        test_dataset = test_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_dataset, validation_dataset, test_dataset, class_names
