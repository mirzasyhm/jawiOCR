# training.py
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import mixed_precision
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_preprocess_input

from data_preprocessing import create_datasets
from cnn import create_resnet50_model # Updated import

# --- Configuration Parameters for ResNet50 ---
IMG_WIDTH = 224  # ResNet50 typically uses 224x224
IMG_HEIGHT = 224
IMAGE_SIZE = (IMG_HEIGHT, IMG_WIDTH)
BATCH_SIZE = 64  # Can often use a larger batch size with A100 for pre-trained models
NUM_CLASSES = 4
INITIAL_EPOCHS = 15   # Epochs for training only the top layers
FINE_TUNE_EPOCHS = 15 # Epochs for fine-tuning (if enabled)
LEARNING_RATE = 0.001 # Initial learning rate
FINE_TUNE_LR = 0.00001 # Very low learning rate for fine-tuning
FINE_TUNE_AT_LAYERS = 20 # Number of layers from the end of ResNet50 to unfreeze for fine-tuning
                        # Set to 0 to disable fine-tuning phase & keep base frozen.

# Mixed Precision Policy
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
print(f"Using mixed precision policy: {policy.name}")

BASE_DIR = 'jawi_orientation_dataset'
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
VALIDATION_DIR = os.path.join(BASE_DIR, 'val')
TEST_DIR = os.path.join(BASE_DIR, 'test')

MODEL_SAVE_PATH = 'jawi_orientation_resnet50.keras'
BEST_MODEL_SAVE_PATH = 'best_jawi_orientation_resnet50.keras'

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
        print(f"Error: Training or validation directory not found.")
        return

    # 1. Load and Preprocess Data
    INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, 3)
    test_dir_to_pass = TEST_DIR if os.path.exists(TEST_DIR) and os.listdir(TEST_DIR) else None

    # Load datasets using the original create_datasets function
    train_ds, val_ds, test_ds, class_names_loaded = create_datasets(
        IMAGE_SIZE, BATCH_SIZE, TRAIN_DIR, VALIDATION_DIR, test_dir_to_pass
    )
    
    # Apply ResNet50 specific preprocessing
    # Note: resnet50_preprocess_input expects pixel values in [0, 255] range if using default mode.
    # Our create_datasets already scales to [0,1]. So, we'll first undo that scaling (multiply by 255)
    # before applying resnet50_preprocess_input.
    # Alternatively, one could remove Rescaling(1./255) from create_datasets.
    def unscale_and_preprocess_resnet(image, label):
        image = image * 255.0 # Rescale from [0,1] back to [0,255]
        return resnet50_preprocess_input(image), label

    train_ds = train_ds.map(unscale_and_preprocess_resnet, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(unscale_and_preprocess_resnet, num_parallel_calls=tf.data.AUTOTUNE)
    if test_ds:
        test_ds = test_ds.map(unscale_and_preprocess_resnet, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Re-apply prefetch after mapping
    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    if test_ds:
      test_ds = test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    actual_num_classes = len(class_names_loaded)
    print(f"Loaded class names: {class_names_loaded}, Num classes: {actual_num_classes}")

    # --- Phase 1: Train only the top custom layers ---
    print("\n--- Phase 1: Training Top Layers ---")
    model = create_resnet50_model(INPUT_SHAPE, actual_num_classes, fine_tune_at=0) # All base layers frozen
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks_list_phase1 = [
        ModelCheckpoint(filepath=BEST_MODEL_SAVE_PATH, save_best_only=True, monitor='val_accuracy', mode='max', verbose=1),
        EarlyStopping(monitor='val_loss', patience=7, verbose=1, restore_best_weights=True), # Shorter patience for initial phase
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001, verbose=1)
    ]

    history_phase1 = model.fit(
        train_ds,
        epochs=INITIAL_EPOCHS,
        validation_data=val_ds,
        callbacks=callbacks_list_phase1
    )

    # Load the best weights from phase 1 for fine-tuning
    if os.path.exists(BEST_MODEL_SAVE_PATH):
        print(f"Loading best weights from Phase 1: {BEST_MODEL_SAVE_PATH}")
        model.load_weights(BEST_MODEL_SAVE_PATH)

    # --- Phase 2: Fine-tuning (optional) ---
    total_epochs_run = len(history_phase1.history['loss'])

    if FINE_TUNE_AT_LAYERS > 0 and FINE_TUNE_EPOCHS > 0:
        print("\n--- Phase 2: Fine-tuning ResNet50 Base Layers ---")
        # Re-create the model with some base layers unfrozen
        # Or, more efficiently, unfreeze layers in the existing model object
        
        # Unfreeze layers in the existing model
        base_model = model.layers[1] # Assuming base_model is the second layer after InputLayer wrapper by Model()
        base_model.trainable = True
        
        # Freeze all layers initially
        for layer in base_model.layers:
            layer.trainable = False
        
        # Unfreeze layers from `FINE_TUNE_AT_LAYERS` onwards (from the top)
        if FINE_TUNE_AT_LAYERS > 0 and FINE_TUNE_AT_LAYERS < len(base_model.layers):
            print(f"Fine-tuning ResNet50: Unfreezing top {FINE_TUNE_AT_LAYERS} layers of the base model for fine-tuning.")
            for layer in base_model.layers[-FINE_TUNE_AT_LAYERS:]:
                if not isinstance(layer, tf.keras.layers.BatchNormalization): # Keep BN layers frozen
                    layer.trainable = True
        else:
             print("Warning: FINE_TUNE_AT_LAYERS value out of range for base model. Fine-tuning with current trainable status.")


        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=FINE_TUNE_LR), # Use a very low learning rate
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        model.summary() # Show summary again to see trainable params change

        callbacks_list_phase2 = [
            ModelCheckpoint(filepath=BEST_MODEL_SAVE_PATH, save_best_only=True, monitor='val_accuracy', mode='max', verbose=1), # Continue saving best model
            EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=4, min_lr=0.000001, verbose=1) # Even smaller min_lr
        ]
        
        print(f"Continuing training from epoch {total_epochs_run}")
        history_phase2 = model.fit(
            train_ds,
            epochs=total_epochs_run + FINE_TUNE_EPOCHS,
            initial_epoch=total_epochs_run,
            validation_data=val_ds,
            callbacks=callbacks_list_phase2
        )
        
        # Combine histories for plotting
        for key in history_phase1.history:
            history_phase1.history[key].extend(history_phase2.history[key])
        history = history_phase1 # Now history contains combined results
        
    else:
        print("\nSkipping fine-tuning phase.")
        history = history_phase1

    print("--- Model Training (including fine-tuning if any) Finished ---")
    
    # Save the final model (which should be the best model if EarlyStopping restored weights)
    model.save(MODEL_SAVE_PATH)
    print(f"Final model saved to {MODEL_SAVE_PATH}")

    # Plotting (using combined history)
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right'); plt.title('Accuracy'); plt.xlabel('Epoch'); plt.ylabel('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right'); plt.title('Loss'); plt.xlabel('Epoch'); plt.ylabel('Loss')
    
    plt.suptitle('ResNet50 Model Training History', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('training_history_resnet50.png')
    print("Training history plot saved to training_history_resnet50.png")
    plt.show()

    if test_ds:
        print("\n--- Evaluating on Test Set (with best model weights) ---")
        if os.path.exists(BEST_MODEL_SAVE_PATH):
            print(f"Loading best model from {BEST_MODEL_SAVE_PATH} for test evaluation.")
            model.load_weights(BEST_MODEL_SAVE_PATH) # Ensure best weights are loaded
        
        test_loss, test_accuracy = model.evaluate(test_ds)
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")

if __name__ == '__main__':
    main()
