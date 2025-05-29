# training.py
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import mixed_precision # Import mixed precision

# Import functions from your other files
from data_preprocessing import create_datasets
from cnn import create_cnn_model

# --- Configuration Parameters for A100 GPU Optimization ---
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMAGE_SIZE = (IMG_HEIGHT, IMG_WIDTH)
# Increase BATCH_SIZE for powerful GPUs like A100. Test for optimal value.
# Ensure it's a multiple of 8, or ideally 64 or 128 for A100.
BATCH_SIZE = 64 # Start with a larger batch size, e.g., 64, 128, or 256
NUM_CLASSES = 4
EPOCHS = 20
LEARNING_RATE = 0.001

# --- Mixed Precision Policy ---
# Use 'mixed_float16' for NVIDIA GPUs with Tensor Cores (like A100)
# This can significantly speed up training and reduce memory usage.
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
print(f"Using mixed precision policy: {policy.name}")


# Dataset directory paths
BASE_DIR = 'jawi_orientation_dataset'
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
VALIDATION_DIR = os.path.join(BASE_DIR, 'validation')
TEST_DIR = os.path.join(BASE_DIR, 'test')

# Model save paths
MODEL_SAVE_PATH = 'jawi_orientation_classifier_mixed_precision.keras'
BEST_MODEL_SAVE_PATH = 'best_jawi_orientation_classifier_mixed_precision.keras'


def main():
    print(f"TensorFlow version: {tf.__version__}")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Found GPUs: {gpus}")
        except RuntimeError as e:
            print(e)
    else:
        print("No GPU found, training on CPU.")


    if not os.path.exists(TRAIN_DIR) or not os.path.exists(VALIDATION_DIR):
        print(f"Error: Training ('{TRAIN_DIR}') or validation ('{VALIDATION_DIR}') directory not found.")
        return

    INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, 1)

    test_dir_to_pass = None
    if os.path.exists(TEST_DIR) and os.listdir(TEST_DIR):
         test_dir_to_pass = TEST_DIR

    train_ds, val_ds, test_ds, class_names_loaded = create_datasets(
        IMAGE_SIZE, BATCH_SIZE, TRAIN_DIR, VALIDATION_DIR, test_dir_to_pass
    )

    print(f"Loaded class names: {class_names_loaded}")
    actual_num_classes = len(class_names_loaded)
    if actual_num_classes != NUM_CLASSES:
        print(f"Warning: NUM_CLASSES config is {NUM_CLASSES}, but found {actual_num_classes} classes. Using {actual_num_classes}.")
        # global NUM_CLASSES # If NUM_CLASSES was global
        # NUM_CLASSES = actual_num_classes # This would require NUM_CLASSES to be global or passed around

    model = create_cnn_model(INPUT_SHAPE, actual_num_classes) # Use actual number of classes
    model.summary()

    # For mixed precision, the optimizer does not need to be wrapped if using model.fit()
    # Loss scaling is handled automatically by TensorFlow when using model.fit() and a global policy.
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    # If NOT using model.fit (i.e. custom training loop), you would need LossScaleOptimizer:
    # optimizer = mixed_precision.LossScaleOptimizer(optimizer)

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model_checkpoint_cb = ModelCheckpoint(
        filepath=BEST_MODEL_SAVE_PATH,
        save_best_only=True,
        monitor='val_accuracy',
        mode='max',
        verbose=1
    )
    early_stopping_cb = EarlyStopping(
        monitor='val_loss',
        patience=10,
        verbose=1,
        restore_best_weights=True
    )
    reduce_lr_cb = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=0.00001,
        verbose=1
    )
    callbacks_list = [model_checkpoint_cb, early_stopping_cb, reduce_lr_cb]

    print(f"\n--- Starting Model Training with Batch Size: {BATCH_SIZE} ---")
    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
        callbacks=callbacks_list
    )
    print("--- Model Training Finished ---")

    model.save(MODEL_SAVE_PATH)
    print(f"Final model saved to {MODEL_SAVE_PATH}")

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.suptitle('Model Training History', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('training_history_mixed_precision.png')
    print("Training history plot saved to training_history_mixed_precision.png")
    plt.show()

    if test_ds:
        print("\n--- Evaluating on Test Set ---")
        model_to_evaluate = model
        if os.path.exists(BEST_MODEL_SAVE_PATH):
            print(f"Loading best model from {BEST_MODEL_SAVE_PATH} for test evaluation.")
            try:
                # When loading a model saved with mixed precision,
                # custom objects might sometimes be needed if layers were custom,
                # but for standard layers and global policy, it's often direct.
                model_to_evaluate = tf.keras.models.load_model(BEST_MODEL_SAVE_PATH)
            except Exception as e:
                print(f"Could not load best model due to: {e}. Using the final model for evaluation.")
        
        test_loss, test_accuracy = model_to_evaluate.evaluate(test_ds)
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
    else:
        print("\nNo test set provided or test set is empty. Skipping test set evaluation.")

if __name__ == '__main__':
    main()
