# ==============================================================================
# SkinCareAI - Dermatological Screening Assistant
# Data Loader Module
# ==============================================================================

# ==============================================================================
# Imports
# ==============================================================================
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras


# ==============================================================================
# Constants
# ==============================================================================
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

CLASS_NAMES = [
    "akiec",  # Actinic Keratoses & Intraepithelial Carcinoma
    "bcc",    # Basal Cell Carcinoma
    "bkl",    # Benign Keratosis-like Lesions
    "df",     # Dermatofibroma
    "mel",    # Melanoma
    "nv",     # Melanocytic Nevi
    "vasc",   # Vascular Lesions
]

# ImageNet normalization parameters
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# HAM10000 class distribution weights (inverse frequency for imbalance handling)
# Approximate counts: nv~67%, mel~11%, bkl~11%, bcc~5%, akiec~3%, vasc~1.5%, df~1%
CLASS_WEIGHTS = {
    0: 6.8,   # akiec
    1: 4.5,   # bcc
    2: 2.0,   # bkl
    3: 16.0,  # df
    4: 2.0,   # mel
    5: 0.3,   # nv
    6: 11.0,  # vasc
}


# ==============================================================================
# Preprocessing Functions
# ==============================================================================
def preprocess_image_train(image, label):
    """
    Apply training-time augmentations and ImageNet normalization.
    Mirrors ResNet50 preprocessing convention from Pneumonia_Detection.
    """
    # -------------------------------------------------------------------------
    # Resize & Cast
    # -------------------------------------------------------------------------
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32)

    # -------------------------------------------------------------------------
    # Augmentations
    # -------------------------------------------------------------------------
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    image = tf.image.random_saturation(image, lower=0.9, upper=1.1)
    image = tf.image.random_hue(image, max_delta=0.05)

    # Random rotation — use tf.image.rot90 (safe inside tf.data.map)
    # Randomly rotate by 0°, 90°, 180°, or 270°
    k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
    image = tf.image.rot90(image, k=k)

    # -------------------------------------------------------------------------
    # ImageNet Normalization (ResNet50 preprocessing)
    # -------------------------------------------------------------------------
    image = tf.keras.applications.resnet.preprocess_input(image)

    return image, label


def preprocess_image_val(image, label):
    """
    Apply validation-time preprocessing only (no augmentation).
    Resize → center crop → normalize.
    """
    # -------------------------------------------------------------------------
    # Resize & Cast
    # -------------------------------------------------------------------------
    image = tf.image.resize(image, [256, 256])
    image = tf.cast(image, tf.float32)

    # -------------------------------------------------------------------------
    # Center Crop to 224×224
    # -------------------------------------------------------------------------
    image = tf.image.resize_with_crop_or_pad(image, IMAGE_SIZE[0], IMAGE_SIZE[1])

    # -------------------------------------------------------------------------
    # ImageNet Normalization (ResNet50 preprocessing)
    # -------------------------------------------------------------------------
    image = tf.keras.applications.resnet.preprocess_input(image)

    return image, label


def preprocess_single_image(image_array):
    """
    Preprocess a single raw numpy image array for inference in the Streamlit app.
    Accepts H×W×3 uint8 array, returns 1×224×224×3 float32 tensor.
    """
    # -------------------------------------------------------------------------
    # Resize & Cast
    # -------------------------------------------------------------------------
    image = tf.image.resize(image_array, IMAGE_SIZE)
    image = tf.cast(image, tf.float32)

    # -------------------------------------------------------------------------
    # ImageNet Normalization
    # -------------------------------------------------------------------------
    image = tf.keras.applications.resnet.preprocess_input(image)

    # -------------------------------------------------------------------------
    # Add Batch Dimension
    # -------------------------------------------------------------------------
    image = tf.expand_dims(image, axis=0)

    return image


# ==============================================================================
# Dataset Builder
# ==============================================================================
def build_dataset(data_dir, subset="train", batch_size=BATCH_SIZE):
    """
    Build a tf.data.Dataset from a directory structured as:
        data_dir/
            train/
                akiec/  bcc/  bkl/  df/  mel/  nv/  vasc/
            val/
                akiec/  bcc/  bkl/  df/  mel/  nv/  vasc/

    Args:
        data_dir (str): Root directory containing train/ and val/ subdirs.
        subset   (str): "train" or "val".
        batch_size (int): Number of images per batch.

    Returns:
        dataset (tf.data.Dataset): Ready-to-train batched and prefetched dataset.
        class_names (list): Ordered list of class name strings.
        num_classes (int): Number of distinct classes found.
    """
    subset_dir = os.path.join(data_dir, subset)

    if not os.path.exists(subset_dir):
        raise FileNotFoundError(
            f"Dataset directory not found: {subset_dir}\n"
            f"Please populate dataset/train/ and dataset/val/ with HAM10000 "
            f"class subdirectories: {CLASS_NAMES}"
        )

    # -------------------------------------------------------------------------
    # Load from Directory
    # -------------------------------------------------------------------------
    raw_dataset = keras.utils.image_dataset_from_directory(
        subset_dir,
        labels="inferred",
        label_mode="int",
        class_names=CLASS_NAMES,
        color_mode="rgb",
        batch_size=None,           # unbatched so we can map augmentations
        image_size=IMAGE_SIZE,
        shuffle=(subset == "train"),
        seed=42,
    )

    class_names = CLASS_NAMES
    num_classes = len(CLASS_NAMES)

    # -------------------------------------------------------------------------
    # Apply Transforms
    # -------------------------------------------------------------------------
    if subset == "train":
        dataset = raw_dataset.map(preprocess_image_train, num_parallel_calls=AUTOTUNE)
        dataset = dataset.shuffle(buffer_size=1000, seed=42)
    else:
        dataset = raw_dataset.map(preprocess_image_val, num_parallel_calls=AUTOTUNE)
        dataset = dataset.cache()  # Cache validation preprocessed images in RAM

    # -------------------------------------------------------------------------
    # Batch & Prefetch
    # -------------------------------------------------------------------------
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)

    return dataset, class_names, num_classes


# ==============================================================================
# Dataset Info Utility
# ==============================================================================
def get_dataset_info(data_dir):
    """
    Print class distribution statistics for each dataset split.

    Args:
        data_dir (str): Root directory containing train/ and val/ subdirs.
    """
    print("\n" + "=" * 70)
    print("  HAM10000 Dataset Information")
    print("=" * 70)

    for split in ["train", "val"]:
        split_dir = os.path.join(data_dir, split)
        if not os.path.exists(split_dir):
            print(f"\n[{split.upper()}] Directory not found: {split_dir}")
            continue

        print(f"\n[{split.upper()}] Split:")
        print("-" * 40)
        total = 0
        for cls in CLASS_NAMES:
            cls_dir = os.path.join(split_dir, cls)
            if os.path.exists(cls_dir):
                count = len([
                    f for f in os.listdir(cls_dir)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))
                ])
                total += count
                print(f"  {cls:<8} : {count:>5} images")
            else:
                print(f"  {cls:<8} : [directory missing]")
        print(f"  {'TOTAL':<8} : {total:>5} images")

    print("=" * 70 + "\n")
