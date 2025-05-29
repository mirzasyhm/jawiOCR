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
INITIAL_EPOCHS = 2
FINE_TUNE_EPOCHS = 2
LEARNING_RATE = 0.001
FINE_TUNE_LR = 0.00001
FINE_TUNE_AT_LAYERS = 20

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

    INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, 3) # RGB images
    test_dir_to_pass = TEST_DIR if os.path.exists(TEST_DIR) and os.listdir(TEST_DIR) else None

    train_ds, val_ds, test_ds, class_names_loaded = create_datasets(
        IMAGE_SIZE, BATCH_SIZE, TRAIN_DIR, VALIDATION_DIR, test_dir_to_pass
    )
    
    # Apply ResNet50 specific preprocessing.
    # Images are now directly from image_dataset_from_directory in [0, 255] range.
    def preprocess_for_resnet(image, label):
        # image is already tf.float32 in [0, 255] from image_dataset_from_directory
        return resnet50_preprocess_input(image), label

    train_ds = train_ds.map(preprocess_for_resnet, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(preprocess_for_resnet, num_parallel_calls=tf.data.AUTOTUNE)
    if test_ds:
        test_ds = test_ds.map(preprocess_for_resnet, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Re-apply prefetch after mapping
    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    if test_ds:
      test_ds = test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    actual_num_classes = len(class_names_loaded)
    print(f"Loaded class names: {class_names_loaded}, Num classes: {actual_num_classes}")

    print("\n--- Phase 1: Training Top Layers ---")
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
        
        resnet_base_for_tuning = None
        for layer_in_model in model.layers:
            # Default name for ResNet50 from tf.keras.applications is 'resnet50'
            # Or, if you named it in cnn.py (e.g., name="resnet50_base_application_model") use that name.
            if layer_in_model.name == 'resnet50': # Check for default name
                resnet_base_for_tuning = layer_in_model
                break
            # Alternative check if you used a custom name like "resnet50_base_application_model"
            # elif layer_in_model.name == "resnet50_base_application_model":
            #     resnet_base_for_tuning = layer_in_model
            #     break
        
        if resnet_base_for_tuning is None:
            print("Error: Could not find the ResNet50 base model layer for fine-tuning by common names.")
            # Attempt to get the base model assuming it's the second layer (after InputLayer if functional API was built that way)
            # This is less robust; naming the layer in cnn.py is preferred.
            if len(model.layers) > 1 and isinstance(model.layers[1], tf.keras.Model) and "resnet" in model.layers[1].name.lower():
                print("Attempting to use model.layers[1] as ResNet base.")
                resnet_base_for_tuning = model.layers[1]
            else:
                print("Could not reliably identify ResNet base layer. Skipping fine-tuning.")

        if resnet_base_for_tuning: # Proceed only if found
            print(f"Found ResNet50 base layer for fine-tuning: {resnet_base_for_tuning.name}")
            resnet_base_for_tuning.trainable = True
            
            for layer in resnet_base_for_tuning.layers:
                layer.trainable = False # Freeze all first
            
            if FINE_TUNE_AT_LAYERS > 0 and FINE_TUNE_AT_LAYERS <= len(resnet_base_for_tuning.layers):
                print(f"Unfreezing top {FINE_TUNE_AT_LAYERS} layers of the ResNet50 base model.")
                for layer_idx in range(len(resnet_base_for_tuning.layers) - FINE_TUNE_AT_LAYERS, len(resnet_base_for_tuning.layers)):
                    layer_to_unfreeze = resnet_base_for_tuning.layers[layer_idx]
                    if not isinstance(layer_to_unfreeze, tf.keras.layers.BatchNormalization):
                        layer_to_unfreeze.trainable = True
                    else:
                        print(f"Keeping BatchNormalization layer {layer_to_unfreeze.name} frozen.")
            elif FINE_TUNE_AT_LAYERS > len(resnet_base_for_tuning.layers):
                 print(f"Warning: FINE_TUNE_AT_LAYERS too large. Unfreezing all non-BN layers in ResNet base.")
                 for layer_to_unfreeze in resnet_base_for_tuning.layers:
                     if not isinstance(layer_to_unfreeze, tf.keras.layers.BatchNormalization):
                        layer_to_unfreeze.trainable = True
            else:
                print("Warning: FINE_TUNE_AT_LAYERS is 0. No layers in ResNet base will be unfrozen.")

            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=FINE_TUNE_LR),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            model.summary()

            callbacks_list_phase2 = [
                ModelCheckpoint(filepath=BEST_MODEL_SAVE_PATH, save_best_only=True, monitor='val_accuracy', mode='max', verbose=1),
                EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=4, min_lr=0.000001, verbose=1)
            ]
            
            print(f"Continuing training from epoch {total_epochs_run}")
            history_phase2 = model.fit(
                train_ds,
                epochs=total_epochs_run + FINE_TUNE_EPOCHS,
                initial_epoch=total_epochs_run,
                validation_data=val_ds,
                callbacks=callbacks_list_phase2
            )
            
            for key in history_phase1.history:
                history_phase1.history[key].extend(history_phase2.history[key])
            history = history_phase1
        else: # resnet_base_for_tuning was not found
             print("Skipping fine-tuning phase as ResNet base layer was not identified.")
             history = history_phase1 # Use only phase 1 history

    else: # FINE_TUNE_AT_LAYERS is 0 or FINE_TUNE_EPOCHS is 0
        print("\nSkipping fine-tuning phase based on configuration.")
        history = history_phase1

    # ... (rest of the saving, plotting, and evaluation code remains the same) ...
    print("--- Model Training (including fine-tuning if any) Finished ---")
    
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
            model.load_weights(BEST_MODEL_SAVE_PATH)
        
        test_loss, test_accuracy = model.evaluate(test_ds)
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")

if __name__ == '__main__':
    main()
