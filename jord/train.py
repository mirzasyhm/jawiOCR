# training.py
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import mixed_precision
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_preprocess_input

from data_preprocessing import create_datasets
from cnn import create_resnet50_model # This now returns (model, base_model_instance)

# --- Configuration Parameters for ResNet50 ---
IMG_WIDTH = 224
IMG_HEIGHT = 224
IMAGE_SIZE = (IMG_HEIGHT, IMG_WIDTH)
BATCH_SIZE = 256
NUM_CLASSES = 4
INITIAL_EPOCHS = 25
FINE_TUNE_EPOCHS = 25
LEARNING_RATE = 0.0001
FINE_TUNE_LR = 0.00001
FINE_TUNE_AT_LAYERS = 20 # Number of layers from the END of ResNet50 base to unfreeze

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

    INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, 3)
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
    # Get both the full model and the reference to the ResNet50 base
    # initial_base_frozen=True ensures base_model_instance.trainable = False for this phase
    model, resnet50_base_ref = create_resnet50_model(INPUT_SHAPE, actual_num_classes, initial_base_frozen=True)
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
        print(f"Loading best weights from Phase 1 into current model structure: {BEST_MODEL_SAVE_PATH}")
        model.load_weights(BEST_MODEL_SAVE_PATH) # Load weights into the existing model object

    total_epochs_run = len(history_phase1.history['loss'])

    if FINE_TUNE_AT_LAYERS > 0 and FINE_TUNE_EPOCHS > 0:
        print("\n--- Phase 2: Fine-tuning ResNet50 Base Layers ---")
        
        # Now use the direct reference 'resnet50_base_ref'
        if resnet50_base_ref is None: # Should not happen with the changes in cnn.py
            print("CRITICAL Error: resnet50_base_ref is None. Skipping fine-tuning.")
            history = history_phase1 # Ensure history is defined
        else:
            print(f"Preparing to fine-tune base model: {resnet50_base_ref.name}")
            # Make the entire ResNet50 base block trainable
            resnet50_base_ref.trainable = True
            
            # Freeze all layers *within* the ResNet50 base model initially for precise unfreezing
            for layer in resnet50_base_ref.layers:
                layer.trainable = False # This ensures a clean slate before unfreezing specific layers
            
            num_internal_base_layers = len(resnet50_base_ref.layers)
            if FINE_TUNE_AT_LAYERS > 0 and FINE_TUNE_AT_LAYERS <= num_internal_base_layers:
                print(f"Unfreezing top {FINE_TUNE_AT_LAYERS} layers of {resnet50_base_ref.name} (total {num_internal_base_layers} internal layers).")
                # Unfreeze from the end of the resnet50_base_ref.layers list
                for layer_idx in range(num_internal_base_layers - FINE_TUNE_AT_LAYERS, num_internal_base_layers):
                    layer_to_unfreeze = resnet50_base_ref.layers[layer_idx]
                    if not isinstance(layer_to_unfreeze, tf.keras.layers.BatchNormalization):
                        layer_to_unfreeze.trainable = True
                        # print(f"  Unfrozen: {layer_to_unfreeze.name}")
                    else:
                        print(f"  Keeping BatchNormalization layer {layer_to_unfreeze.name} frozen.")
            elif FINE_TUNE_AT_LAYERS > num_internal_base_layers:
                 print(f"Warning: FINE_TUNE_AT_LAYERS ({FINE_TUNE_AT_LAYERS}) is > total internal layers in {resnet50_base_ref.name} ({num_internal_base_layers}). Unfreezing all non-BN internal layers.")
                 for layer_to_unfreeze in resnet50_base_ref.layers:
                     if not isinstance(layer_to_unfreeze, tf.keras.layers.BatchNormalization):
                        layer_to_unfreeze.trainable = True
            else: # FINE_TUNE_AT_LAYERS is 0 (already handled by outer if) or invalid negative
                print(f"Warning: FINE_TUNE_AT_LAYERS is {FINE_TUNE_AT_LAYERS}. No internal layers in {resnet50_base_ref.name} will be specifically unfrozen by count.")

            # Re-compile the *overall model* for these modifications to take effect
            # The 'model' object's graph includes the resnet50_base_ref with its updated layer trainability
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=FINE_TUNE_LR),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            print(f"Model recompiled for fine-tuning. Summary of the overall model '{model.name}':")
            model.summary() # Show summary again to verify trainable parameters

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
                if key in history_phase2.history:
                     history_phase1.history[key].extend(history_phase2.history[key])
            history = history_phase1
    else:
        print("\nSkipping fine-tuning phase based on configuration (FINE_TUNE_AT_LAYERS or FINE_TUNE_EPOCHS is 0).")
        history = history_phase1

    if 'history' not in locals(): # Should always be defined by now
        history = history_phase1

    model.save(MODEL_SAVE_PATH)
    print(f"Final model saved to {MODEL_SAVE_PATH}")

    # Plotting (ensure history dictionary is not empty)
    acc = history.history.get('accuracy', [])
    val_acc = history.history.get('val_accuracy', [])
    loss = history.history.get('loss', [])
    val_loss = history.history.get('val_loss', [])
    
    if not acc:
        print("Warning: Training history is empty. Skipping plot generation.")
    else:
        epochs_range = range(len(acc))
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy'); plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right'); plt.title('Accuracy'); plt.xlabel('Epoch'); plt.ylabel('Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss'); plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right'); plt.title('Loss'); plt.xlabel('Epoch'); plt.ylabel('Loss')
        
        plt.suptitle('ResNet50 Model Training History', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig('training_history_resnet50.png')
        print("Training history plot saved to training_history_resnet50.png")
        plt.show()

    if test_ds:
        print("\n--- Evaluating on Test Set (with best model weights from current structure) ---")
        if os.path.exists(BEST_MODEL_SAVE_PATH):
            print(f"Loading best weights from {BEST_MODEL_SAVE_PATH} into current model structure for test evaluation.")
            model.load_weights(BEST_MODEL_SAVE_PATH)
        
        test_loss, test_accuracy = model.evaluate(test_ds)
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")

if __name__ == '__main__':
    main()
