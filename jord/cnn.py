# cnn.py
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras import mixed_precision # For mixed precision output layer

def create_resnet50_model(input_shape, num_classes, fine_tune_at=0):
    """
    Creates a model using ResNet50 as a base for transfer learning.

    Args:
        input_shape: Tuple, shape of input images (height, width, channels).
        num_classes: Integer, number of output classes.
        fine_tune_at: Integer, number of layers from the end of the base model
                      to unfreeze for fine-tuning. If 0, all base layers frozen.
                      If > 0, layers from 'fine_tune_at' onwards are unfrozen.
                      A common strategy is to unfreeze layers after a certain block.
                      For ResNet50, common blocks are 'conv5_block1_out', 'conv4_block1_out', etc.
                      If an integer is passed, it means unfreeze the top `fine_tune_at` layers.
    Returns:
        A Keras Model instance.
    """
    # Load ResNet50 pre-trained on ImageNet, without the top classification layer
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,  # Exclude the final Dense layer of ResNet50
        input_shape=input_shape
    )

    # Freeze the layers of the base model initially
    if fine_tune_at == 0:
        base_model.trainable = False
        print("All base ResNet50 layers frozen.")
    else:
        base_model.trainable = True
        # Freeze all layers up to a certain point, and unfreeze the rest
        # This requires knowing layer names or counting layers.
        # A simpler approach for ResNet50 is to unfreeze later blocks.
        # For this example, we'll use a simpler integer-based unfreezing.
        # Freeze all layers initially
        for layer in base_model.layers:
            layer.trainable = False
        
        # Unfreeze layers from `fine_tune_at` onwards (from the top)
        if fine_tune_at > 0 and fine_tune_at < len(base_model.layers):
            print(f"Fine-tuning ResNet50: Unfreezing top {fine_tune_at} layers of the base model.")
            for layer in base_model.layers[-fine_tune_at:]:
                 # Be cautious unfreezing BatchNormalization layers when fine-tuning
                 # with small batch sizes and few samples per class.
                 # Often, it's recommended to keep them frozen initially.
                if not isinstance(layer, tf.keras.layers.BatchNormalization):
                    layer.trainable = True
        else:
            print("Warning: fine_tune_at value is out of range or 0. All base layers remain frozen or all unfrozen if fine_tune_at makes base_model.trainable=True.")


    # Add custom classification layers on top of ResNet50
    x = base_model.output
    x = GlobalAveragePooling2D()(x)  # Converts features to a vector per image
    
    # Add a few dense layers for classification
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x) # Good to add BN before Dropout
    x = Dropout(0.5)(x)
    
    # Output layer (ensure dtype is float32 for mixed precision compatibility)
    predictions = Dense(num_classes, activation='softmax', dtype='float32')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    
    print(f"Total layers in base_model: {len(base_model.layers)}")
    trainable_layers = sum([1 for layer in base_model.layers if layer.trainable])
    print(f"Trainable layers in base_model after setup: {trainable_layers}")
    
    return model
