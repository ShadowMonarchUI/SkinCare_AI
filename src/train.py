# ==============================================================================
# SkinCareAI - Dermatological Screening Assistant
# Training Pipeline
# ==============================================================================

# ==============================================================================
# Imports
# ==============================================================================
import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import tensorflow as tf
from tensorflow import keras

# Local modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import (
    build_dataset,
    get_dataset_info,
    CLASS_NAMES,
    CLASS_WEIGHTS,
)
from src.model import build_model, unfreeze_model, print_model_info


# ==============================================================================
# Argument Parser
# ==============================================================================
def parse_args():
    """
    Parse command-line arguments for the training pipeline.
    """
    parser = argparse.ArgumentParser(
        description="SkinCareAI - HAM10000 Skin Lesion Training Pipeline"
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="dataset",
        help="Root directory containing train/ and val/ splits (default: dataset)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs for initial head training (default: 30)",
    )
    parser.add_argument(
        "--fine_tune_epochs",
        type=int,
        default=10,
        help="Number of additional fine-tuning epochs with unfrozen top layers (default: 10)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training (default: 32)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Initial learning rate for head training (default: 1e-4)",
    )
    parser.add_argument(
        "--fine_tune_lr",
        type=float,
        default=1e-5,
        help="Learning rate for fine-tuning phase (default: 1e-5)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help="Dropout rate in classification head (default: 0.3)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/skin_model.keras",
        help="Output path for best model weights (default: models/skin_model.keras)",
    )
    parser.add_argument(
        "--no_fine_tune",
        action="store_true",
        help="Skip fine-tuning phase and only train the classification head",
    )
    parser.add_argument(
        "--fine_tune_only",
        action="store_true",
        help="Skip Phase 1 and run fine-tuning only on an existing saved model",
    )

    return parser.parse_args()


# ==============================================================================
# Callbacks
# ==============================================================================
def build_callbacks(model_output_path):
    """
    Construct the Keras callback suite for training.

    Includes:
      - ModelCheckpoint  : saves best val_accuracy weights
      - EarlyStopping    : patience=10 on val_accuracy
      - ReduceLROnPlateau: halves LR on 3-epoch val_loss plateau
      - CSVLogger        : writes epoch-by-epoch metrics to training_log.csv

    Args:
        model_output_path (str): Path to save best model (.keras).

    Returns:
        list of keras.callbacks.Callback instances.
    """
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=model_output_path,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1,
    )

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=10,
        restore_best_weights=True,
        verbose=1,
    )

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1,
    )

    csv_logger = keras.callbacks.CSVLogger(
        "training_log.csv",
        append=True,
    )

    return [checkpoint, early_stop, reduce_lr, csv_logger]


# ==============================================================================
# Training History Plot
# ==============================================================================
def plot_training_history(history, fine_tune_history=None, output_path="training_history.png"):
    """
    Generate and save a training history plot showing loss and accuracy curves
    for both the head-training and optional fine-tuning phases.

    Args:
        history           (History): Keras History from head-training phase.
        fine_tune_history (History): Keras History from fine-tuning phase (optional).
        output_path       (str):     File path for the saved PNG.
    """
    acc      = history.history["accuracy"]
    val_acc  = history.history["val_accuracy"]
    loss     = history.history["loss"]
    val_loss = history.history["val_loss"]

    if fine_tune_history:
        acc     += fine_tune_history.history["accuracy"]
        val_acc += fine_tune_history.history["val_accuracy"]
        loss    += fine_tune_history.history["loss"]
        val_loss+= fine_tune_history.history["val_loss"]

    epochs_range = range(1, len(acc) + 1)
    fine_tune_start = len(history.history["accuracy"]) if fine_tune_history else None

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("SkinCareAI — Training History", fontsize=14, fontweight="bold")

    # -------------------------------------------------------------------------
    # Accuracy Plot
    # -------------------------------------------------------------------------
    axes[0].plot(epochs_range, acc,     label="Train Accuracy",      color="#3498db")
    axes[0].plot(epochs_range, val_acc, label="Validation Accuracy",  color="#e74c3c")
    if fine_tune_start:
        axes[0].axvline(
            x=fine_tune_start + 0.5,
            linestyle="--",
            color="#f39c12",
            label="Fine-Tuning Start",
        )
    axes[0].set_title("Model Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend(loc="lower right")
    axes[0].grid(alpha=0.3)

    # -------------------------------------------------------------------------
    # Loss Plot
    # -------------------------------------------------------------------------
    axes[1].plot(epochs_range, loss,     label="Train Loss",     color="#3498db")
    axes[1].plot(epochs_range, val_loss, label="Validation Loss", color="#e74c3c")
    if fine_tune_start:
        axes[1].axvline(
            x=fine_tune_start + 0.5,
            linestyle="--",
            color="#f39c12",
            label="Fine-Tuning Start",
        )
    axes[1].set_title("Model Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend(loc="upper right")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\n  [OK] Training history saved -> {output_path}")


# ==============================================================================
# Evaluation
# ==============================================================================
def evaluate_model(model, val_dataset, class_names):
    """
    Run full evaluation on the validation set:
    - Loss and accuracy
    - Per-class classification report
    - Confusion matrix heatmap saved to confusion_matrix.png

    Args:
        model       (keras.Model):       Trained model.
        val_dataset (tf.data.Dataset):   Validation dataset.
        class_names (list):              Class name strings.
    """
    print("\n" + "=" * 70)
    print("  Model Evaluation")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Collect All Predictions
    # -------------------------------------------------------------------------
    all_true   = []
    all_pred   = []

    for images, labels in val_dataset:
        preds = model(images, training=False).numpy()
        all_true.extend(labels.numpy())
        all_pred.extend(np.argmax(preds, axis=1))

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)

    # -------------------------------------------------------------------------
    # Classification Report
    # -------------------------------------------------------------------------
    print("\n  Classification Report\n")
    print(classification_report(all_true, all_pred, target_names=class_names))

    # -------------------------------------------------------------------------
    # Confusion Matrix
    # -------------------------------------------------------------------------
    cm = confusion_matrix(all_true, all_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("SkinCareAI — Confusion Matrix", fontsize=13, fontweight="bold")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  [OK] Confusion matrix saved -> confusion_matrix.png")
    print("=" * 70 + "\n")


# ==============================================================================
# Main Training Pipeline
# ==============================================================================
def main():
    args = parse_args()

    # -------------------------------------------------------------------------
    # GPU Configuration
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  SkinCareAI - Dermatological Screening Training Pipeline")
    print("=" * 70)

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"\n  [GPU] {len(gpus)} GPU(s) detected:")
        for gpu in gpus:
            print(f"    {gpu}")
        tf.config.experimental.set_memory_growth(gpus[0], True)
    else:
        print("\n  [CPU] No GPU detected - training on CPU (slower).")

    print(f"\n  Data Directory   : {args.data_dir}")
    print(f"  Epochs (head)    : {args.epochs}")
    print(f"  Fine-Tune Epochs : {args.fine_tune_epochs}")
    print(f"  Batch Size       : {args.batch_size}")
    print(f"  Learning Rate    : {args.lr}")
    print(f"  Output Path      : {args.output}")

    # -------------------------------------------------------------------------
    # Dataset Info
    # -------------------------------------------------------------------------
    get_dataset_info(args.data_dir)

    # -------------------------------------------------------------------------
    # Build Datasets
    # -------------------------------------------------------------------------
    print("  [1/4] Building datasets...\n")

    train_dataset, class_names, num_classes = build_dataset(
        args.data_dir, subset="train", batch_size=args.batch_size
    )
    val_dataset, _, _ = build_dataset(
        args.data_dir, subset="val", batch_size=args.batch_size
    )

    # -------------------------------------------------------------------------
    # Build Model
    # -------------------------------------------------------------------------
    print("  [2/4] Building model...\n")

    model = build_model(
        num_classes=num_classes,
        dropout_rate=args.dropout,
        learning_rate=args.lr,
    )
    print_model_info(model)

    # -------------------------------------------------------------------------
    # Ensure Output Directory Exists
    # -------------------------------------------------------------------------
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # -------------------------------------------------------------------------
    # Phase 1 - Head Training (frozen backbone)
    # -------------------------------------------------------------------------
    history = None

    if not args.fine_tune_only:
        print("\n" + "=" * 70)
        print("  [3/4] Phase 1: Training Classification Head (Backbone Frozen)")
        print("=" * 70 + "\n")

        callbacks = build_callbacks(args.output)

        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=args.epochs,
            class_weight=CLASS_WEIGHTS,
            callbacks=callbacks,
            verbose=1,
        )

        best_val_acc = max(history.history["val_accuracy"])
        print(f"\n  [OK] Phase 1 Complete - Best Val Accuracy: {best_val_acc * 100:.2f}%")
    else:
        print("\n  [OK] --fine_tune_only: Loading saved model from", args.output)
        model = keras.models.load_model(args.output)
        print("  [OK] Model loaded successfully.")

    # -------------------------------------------------------------------------
    # Phase 2 - Fine-Tuning (unfreeze top ResNet50 layers)
    # -------------------------------------------------------------------------
    fine_tune_history = None

    if not args.no_fine_tune:
        print("\n" + "=" * 70)
        print("  [3/4] Phase 2: Fine-Tuning Top Layers (Partial Backbone Unfrozen)")
        print("=" * 70 + "\n")

        model = unfreeze_model(
            model,
            unfreeze_from_layer=143,
            learning_rate=args.fine_tune_lr,
        )

        fine_tune_callbacks = build_callbacks(args.output)

        fine_tune_history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=args.fine_tune_epochs,
            class_weight=CLASS_WEIGHTS,
            callbacks=fine_tune_callbacks,
            verbose=1,
        )

        best_ft_acc = max(fine_tune_history.history["val_accuracy"])
        print(f"\n  [OK] Phase 2 Complete - Best Val Accuracy: {best_ft_acc * 100:.2f}%")

    # -------------------------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------------------------
    print("\n  [4/4] Running Evaluation...\n")

    # Load best saved model for final evaluation
    best_model = keras.models.load_model(args.output)
    evaluate_model(best_model, val_dataset, class_names)

    # -------------------------------------------------------------------------
    # Save Training History Plot
    # -------------------------------------------------------------------------
    if history is not None:
        plot_training_history(history, fine_tune_history)

    # -------------------------------------------------------------------------
    # Final Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  Training Complete!")
    print("=" * 70)
    print(f"\n  Best Model Saved  : {args.output}")
    print(f"  Training Log      : training_log.csv")
    print(f"  History Plot      : training_history.png")
    print(f"  Confusion Matrix  : confusion_matrix.png")
    print("\n  Run the app:\n    streamlit run app.py\n")
    print("=" * 70 + "\n")


# ==============================================================================
# Entry Point
# ==============================================================================
if __name__ == "__main__":
    main()
