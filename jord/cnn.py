# cnn.py
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization

def create_resnet50_model(input_shape, num_classes, initial_base_frozen=True):
    """
    Creates a model using ResNet50 as a base for transfer learning.

    Args:
        input_shape: Tuple, shape of input images (height, width, channels).
        num_classes: Integer, number of output classes.
        initial_base_frozen: Boolean, if True, the base_model is frozen initially.
                             Set to False if you intend to fine-tune from the start
                             (less common for a two-phase approach).
    Returns:
        A tuple: (Keras Model instance for the full model, Keras Model instance for the base ResNet50 model)
    """
    # Load ResNet50 pre-trained on ImageNet, without the top classification layer
    # Give the ResNet50 application model itself a name for clarity, though it's not directly used by model.get_layer()
    base_model_instance = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape,
        name="resnet50_application_base" # Name of the ResNet50 tf.keras.Model object
    )

    # Set initial trainability for the entire base_model block for Phase 1
    if initial_base_frozen:
        base_model_instance.trainable = False
        print(f"Base model '{base_model_instance.name}' set to non-trainable for initial phase.")
    else:
        base_model_instance.trainable = True
        print(f"Base model '{base_model_instance.name}' set to trainable for initial phase.")

    # Add custom classification layers on top
    x = base_model_instance.output # Get output tensor from the base model
    x = GlobalAveragePooling2D(name="custom_global_avg_pool")(x)
    x = Dense(256, activation='relu', name="custom_dense_1")(x)
    x = BatchNormalization(name="custom_batch_norm")(x)
    x = Dropout(0.5, name="custom_dropout_1")(x)
    predictions_output = Dense(num_classes, activation='softmax', dtype='float32', name="custom_output_softmax")(x)

    # Create the overall model
    full_model = Model(inputs=base_model_instance.input, outputs=predictions_output, name="jawi_orientation_classifier_with_resnet50")
    
    print(f"--- Full model created: {full_model.name} ---")
    print(f"  Using base model: {base_model_instance.name}")
    print(f"  Base model initial trainable status: {base_model_instance.trainable}")
    
    # Return both the full model and the reference to the base ResNet50 model
    return full_model, base_model_instance
