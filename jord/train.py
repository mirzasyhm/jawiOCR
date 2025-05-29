# training.py
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import mixed_precision
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_preprocess_input

from data_preprocessing import create_datasets
from cnn import create_resnet50_model # Assuming cnn.py has create_resnet50_model

# --- Configuration Parameters for ResNet50 ---
IMG_WIDTH = 224
IMG_HEIGHT = 224
IMAGE_SIZE = (IMG_HEIGHT, IMG_WIDTH)
BATCH_SIZE = 64
NUM_CLASSES = 4
INITIAL_EPOCHS = 2   # Epochs for training only the top layers
FINE_TUNE_EPOCHS = 2 # Epochs for fine-tuning (if enabled)
LEARNING_RATE = 0.001 # Initial learning rate
FINE_TUNE_LR = 0.00001 # Very low learning rate for fine-tuning
# Number of layers from the end of ResNet50 base to unfreeze for fine-tuning
# For example, ResNet50 has 175 layers (approx, excluding Input).
# Setting to 20 means unfreezing roughly the last conv block.
FINE_TUNE_AT_LAYERS = 20
# Set to 0 to disable fine-tuning phase & keep base frozen for the whole training.

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
    # ... (GPU setup, directory checks, data loading, Phase 1 training - all same as previous correct version) ...
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

    INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, 3) # RGB images
    test_dir_to_pass = TEST_DIR if os.path.exists(TEST_DIR) and os.listdir(TEST_DIR) else None

    train_ds, val_ds, test_ds, class_names_loaded = create_datasets(
        IMAGE_SIZE, BATCH_SIZE, TRAIN_DIR, VALIDATION_DIR, test_dir_to_pass
    )
    
    def preprocess_for_resnet(image, label):
        return resnet50_preprocess_input(image), label

    train_ds = train_ds.map(preprocess_for_resnet, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.map(preprocess_for_resnet, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    if test_ds:
        test_ds = test_ds.map(preprocess_for_resnet, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

    actual_num_classes = len(class_names_loaded)
    print(f"Loaded class names: {class_names_loaded}, Num classes: {actual_num_classes}")

    print("\n--- Phase 1: Training Top Layers ---")
    # fine_tune_at=0 is passed to ensure base_model.trainable = False inside create_resnet50_model
    model = create_resnet50_model(INPUT_SHAPE, actual_num_classes, fine_tune_at=0)
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    callbacks_list_phase1 = [
        ModelCheckpoint(filepath=BEST_MODEL_SAVE_PATH, save_best_only=True, monitor='val_accuracy', mode='max', verbose=1),
        EarlyStopping(monitor='val_loss', patience=7, verbose=1, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001, verbose=1)
    ]
    history_phase1 = model.fit(
        train_ds,
        epochs=INITIAL_EPOCHS,
        validation_data=val_ds,
        callbacks=callbacks_list_phase1
    )

    if os.path.exists(BEST_MODEL_SAVE_PATH):
        print(f"Loading best weights from Phase 1: {BEST_MODEL_SAVE_PATH}")
        model.load_weights(BEST_MODEL_SAVE_PATH)

    total_epochs_run = len(history_phase1.history['loss'])

    if FINE_TUNE_AT_LAYERS > 0 and FINE_TUNE_EPOCHS > 0:
        print("\n--- Phase 2: Fine-tuning ResNet50 Base Layers ---")
        
        # *** Get the ResNet50 base model by its explicit name ***
        # This name ("resnet50_base_application") must match the name given in cnn.py
        resnet_base_for_tuning = model.get_layer("resnet50_base_application")
        
        if resnet_base_for_tuning is None: # Should not happen if names match
            print("CRITICAL Error: Could not find the ResNet50 base model layer named 'resnet50_base_application'.")
            print("Skipping fine-tuning.")
            history = history_phase1
        else:
            print(f"Found ResNet50 base layer for fine-tuning: {resnet_base_for_tuning.name}")
            resnet_base_for_tuning.trainable = True # Allow the ResNet50 block to have trainable weights
            
            # Freeze all layers within the ResNet50 base model initially for precise unfreezing
            for layer in resnet_base_for_tuning.layers:
                layer.trainable = False
            
            # Unfreeze layers from `FINE_TUNE_AT_LAYERS` onwards (from the top of ResNet50)
            num_base_layers = len(resnet_base_for_tuning.layers)
            if FINE_TUNE_AT_LAYERS > 0 and FINE_TUNE_AT_LAYERS <= num_base_layers:
                print(f"Unfreezing top {FINE_TUNE_AT_LAYERS} layers of the {resnet_base_for_tuning.name} (total {num_base_layers} layers).")
                for layer_idx in range(num_base_layers - FINE_TUNE_AT_LAYERS, num_base_layers):
                    layer_to_unfreeze = resnet_base_for_tuning.layers[layer_idx]
                    if not isinstance(layer_to_unfreeze, tf.keras.layers.BatchNormalization):
                        layer_to_unfreeze.trainable = True
                        # print(f"  Unfrozen: {layer_to_unfreeze.name}")
                    else:
                        print(f"  Keeping BatchNormalization layer {layer_to_unfreeze.name} frozen.")
            elif FINE_TUNE_AT_LAYERS > num_base_layers:
                 print(f"Warning: FINE_TUNE_AT_LAYERS ({FINE_TUNE_AT_LAYERS}) is > total layers in {resnet_base_for_tuning.name} ({num_base_layers}). Unfreezing all non-BN layers in base.")
                 for layer_to_unfreeze in resnet_base_for_tuning.layers:
                     if not isinstance(layer_to_unfreeze, tf.keras.layers.BatchNormalization):
                        layer_to_unfreeze.trainable = True
            else: # FINE_TUNE_AT_LAYERS is 0 (already handled by outer if) or invalid negative
                print(f"Warning: FINE_TUNE_AT_LAYERS is {FINE_TUNE_AT_LAYERS}. No layers in ResNet base will be specifically unfrozen by count (base trainable status: {resnet_base_for_tuning.trainable}).")

            # Re-compile the model for these modifications to take effect
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=FINE_TUNE_LR), # Use a very low learning rate
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            model.summary() # Show summary again to verify trainable parameters

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
            for key in history_phase1.history: # Should be present
                if key in history_phase2.history:
                     history_phase1.history[key].extend(history_phase2.history[key])
            history = history_phase1 # Now history contains combined results
    else: # FINE_TUNE_AT_LAYERS is 0 or FINE_TUNE_EPOCHS is 0
        print("\nSkipping fine-tuning phase based on configuration (FINE_TUNE_AT_LAYERS or FINE_TUNE_EPOCHS is 0).")
        history = history_phase1

    # ... (rest of the saving, plotting, and evaluation code remains the same) ...
    print("--- Model Training (including fine-tuning if any) Finished ---")
    
    # Make sure 'history' variable is defined in all paths
    if 'history' not in locals():
        history = history_phase1 # Fallback if fine-tuning somehow skipped without setting history

    model.save(MODEL_SAVE_PATH)
    print(f"Final model saved to {MODEL_SAVE_PATH}")

    acc = history.history.get('accuracy', []) # Use .get for safety
    val_acc = history.history.get('val_accuracy', [])
    loss = history.history.get('loss', [])
    val_loss = history.history.get('val_loss', [])
    
    if not acc: # If history is empty for some reason
        print("Warning: Training history is empty. Skipping plot.")
    else:
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
            # It's often better to load weights into the existing model structure
            # rather than model = tf.keras.models.load_model() if you've made structural changes
            # to `trainable` attributes that might not be fully saved/restored by load_model.
            model.load_weights(BEST_MODEL_SAVE_PATH) 
        
        test_loss, test_accuracy = model.evaluate(test_ds)
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")

if __name__ == '__main__':
    main()
