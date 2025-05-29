# cnn.py
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
# from tensorflow.keras import mixed_precision # Not strictly needed here but fine if output layer uses it

def create_resnet50_model(input_shape, num_classes, fine_tune_at=0):
    """
    Creates a model using ResNet50 as a base for transfer learning.
    """
    # Load ResNet50 pre-trained on ImageNet, without the top classification layer
    # *** ADD A NAME TO THE BASE MODEL ***
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape,
        name="resnet50_base_application" # Explicit name for the ResNet50 application model
    )

    # Freeze or unfreeze layers based on fine_tune_at (logic from previous version)
    if fine_tune_at == 0:
        base_model.trainable = False
        print("All base ResNet50 layers frozen.")
    else:
        base_model.trainable = True
        for layer in base_model.layers: # Freeze all internal layers first
            layer.trainable = False
        
        if fine_tune_at > 0 and fine_tune_at <= len(base_model.layers):
            print(f"Preparing for fine-tuning: Top {fine_tune_at} layers of the base model will be unfreezable.")
            # The actual unfreezing based on this count will happen in training.py
            # Here, we just ensure the base_model itself is trainable if fine_tune_at > 0
            # The individual layer unfreezing logic based on count is better handled in training.py
            # before compiling for the fine-tuning phase.
            # For now, `base_model.trainable = True` means the container is trainable,
            # but its internal layers are still individually frozen by the loop above.
            # This will be correctly handled in training.py phase 2.
        elif fine_tune_at > len(base_model.layers):
            print(f"Warning: fine_tune_at ({fine_tune_at}) is > ResNet50 layers ({len(base_model.layers)}). All base layers trainable.")
        else: # fine_tune_at < 0 (invalid)
            print("Warning: fine_tune_at is negative. All base layers frozen.")
            base_model.trainable = False


    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax', dtype='float32')(x) # Ensure float32 for mixed precision

    # Create the overall model
    # *** GIVE A NAME TO THE OVERALL MODEL TOO, FOR CLARITY ***
    model = Model(inputs=base_model.input, outputs=predictions, name="jawi_orientation_resnet_classifier")
    
    print(f"--- Model created: {model.name} ---")
    print(f"Base model ({base_model.name}) trainable: {base_model.trainable}")
    if base_model.trainable: # Only print internal layer counts if base is potentially trainable
        trainable_base_layers = sum([1 for layer in base_model.layers if layer.trainable])
        print(f"Initially trainable layers *within* {base_model.name}: {trainable_base_layers} (will be adjusted for fine-tuning)")
    
    return model
