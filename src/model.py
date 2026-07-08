# ==============================================================================
# SkinCareAI - Dermatological Screening Assistant
# Model Architecture Module
# ==============================================================================

# ==============================================================================
# Imports
# ==============================================================================
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ==============================================================================
# Constants
# ==============================================================================
IMAGE_SIZE  = (224, 224, 3)
NUM_CLASSES = 7

CLASS_NAMES = [
    "akiec",  # Actinic Keratoses & Intraepithelial Carcinoma
    "bcc",    # Basal Cell Carcinoma
    "bkl",    # Benign Keratosis-like Lesions
    "df",     # Dermatofibroma
    "mel",    # Melanoma
    "nv",     # Melanocytic Nevi
    "vasc",   # Vascular Lesions
]

CLASS_FULL_NAMES = {
    "akiec": "Actinic Keratoses / Intraepithelial Carcinoma",
    "bcc":   "Basal Cell Carcinoma",
    "bkl":   "Benign Keratosis-like Lesion",
    "df":    "Dermatofibroma",
    "mel":   "Melanoma",
    "nv":    "Melanocytic Nevi (Mole)",
    "vasc":  "Vascular Lesion",
}

# Triage urgency mapping
# Red = requires urgent dermatologist review
# Amber = monitor / schedule appointment
# Green = likely benign, routine check
TRIAGE_MAP = {
    "mel":   ("URGENT",  "#e74c3c", "⚠️ Possible Melanoma detected. Seek immediate dermatologist evaluation."),
    "bcc":   ("URGENT",  "#e74c3c", "⚠️ Possible Basal Cell Carcinoma. Schedule urgent dermatology appointment."),
    "akiec": ("URGENT",  "#e74c3c", "⚠️ Possible Actinic Keratosis / Carcinoma. Urgent dermatologist review advised."),
    "bkl":   ("MONITOR", "#f39c12", "🔶 Benign Keratosis detected. Monitor for changes and schedule a routine check."),
    "df":    ("MONITOR", "#f39c12", "🔶 Dermatofibroma detected. Generally benign — schedule a routine follow-up."),
    "nv":    ("LOW RISK","#27ae60", "✅ Melanocytic Nevi (mole) detected. Appears benign. Routine self-monitoring recommended."),
    "vasc":  ("LOW RISK","#27ae60", "✅ Vascular lesion detected. Typically benign. Consult a dermatologist if growing."),
}


# ==============================================================================
# Model Builder
# ==============================================================================
def build_model(num_classes=NUM_CLASSES, dropout_rate=0.3, learning_rate=1e-4):
    """
    Build a transfer-learning model based on ResNet50 pretrained on ImageNet.
    Architecture mirrors Pneumonia_Detection's ResNet50 approach, adapted for
    7-class HAM10000 skin lesion classification.

    Architecture:
        Input Image (224×224×3)
                │
        ResNet50 (ImageNet Weights, frozen)
                │
        GlobalAveragePooling2D
                │
        Dropout (0.3)
                │
        Dense (256, ReLU)
                │
        Dropout (0.3)
                │
        Dense (7, Softmax)

    Args:
        num_classes   (int):   Number of output classes. Default 7.
        dropout_rate  (float): Dropout probability for regularisation.
        learning_rate (float): Adam optimiser learning rate.

    Returns:
        model (keras.Model): Compiled Keras model ready for training.
    """
    # -------------------------------------------------------------------------
    # Base Model — ResNet50 (ImageNet Weights, no top)
    # -------------------------------------------------------------------------
    base_model = keras.applications.ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=IMAGE_SIZE,
    )

    # -------------------------------------------------------------------------
    # Freeze Base — only train the new classification head initially
    # -------------------------------------------------------------------------
    base_model.trainable = False

    # -------------------------------------------------------------------------
    # Classification Head
    # -------------------------------------------------------------------------
    inputs = keras.Input(shape=IMAGE_SIZE)

    x = base_model(inputs, training=False)

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dropout(dropout_rate)(x)

    x = layers.Dense(256, activation="relu")(x)

    x = layers.Dropout(dropout_rate)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    # -------------------------------------------------------------------------
    # Assemble Model
    # -------------------------------------------------------------------------
    model = keras.Model(inputs, outputs, name="SkinCareAI_ResNet50")

    # -------------------------------------------------------------------------
    # Compile
    # -------------------------------------------------------------------------
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


# ==============================================================================
# Fine-Tuning Unfreeze
# ==============================================================================
def unfreeze_model(model, unfreeze_from_layer=143, learning_rate=1e-5):
    """
    Unfreeze the top layers of ResNet50 for fine-tuning after initial head training.
    ResNet50 has 175 layers; unfreezing from layer 143 exposes the last conv block.

    Args:
        model             (keras.Model): Previously compiled model.
        unfreeze_from_layer (int):       Layer index to start unfreezing from.
        learning_rate     (float):       Lower LR for fine-tuning.

    Returns:
        model (keras.Model): Recompiled model with partial base unfrozen.
    """
    # -------------------------------------------------------------------------
    # Locate and Unfreeze Base Model Layers
    # -------------------------------------------------------------------------
    base_model = model.layers[1]   # ResNet50 is the second layer after Input
    base_model.trainable = True

    for layer in base_model.layers[:unfreeze_from_layer]:
        layer.trainable = False

    # -------------------------------------------------------------------------
    # Recompile with Lower Learning Rate
    # -------------------------------------------------------------------------
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


# ==============================================================================
# Model Info Utility
# ==============================================================================
def print_model_info(model):
    """
    Print a clean model summary and parameter counts.

    Args:
        model (keras.Model): The compiled Keras model.
    """
    print("\n" + "=" * 70)
    print("  SkinCareAI Model Architecture")
    print("=" * 70)
    model.summary()

    total_params     = model.count_params()
    trainable_params = sum(
        tf.size(w).numpy() for w in model.trainable_weights
    )

    print("\n" + "-" * 40)
    print(f"  Total Parameters     : {total_params:,}")
    print(f"  Trainable Parameters : {trainable_params:,}")
    print(f"  Frozen Parameters    : {total_params - trainable_params:,}")
    print("-" * 40)
    print(f"\n  Classes ({len(CLASS_NAMES)}):")
    for i, cls in enumerate(CLASS_NAMES):
        print(f"    [{i}] {cls:<8} — {CLASS_FULL_NAMES[cls]}")
    print("=" * 70 + "\n")
