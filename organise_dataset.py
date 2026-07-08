# ==============================================================================
# SkinCareAI — Dataset Organiser
# Extracts archive.zip and organises HAM10000 images into:
#   dataset/train/{class}/
#   dataset/val/{class}/
# Uses an 80/20 stratified split per class.
# ==============================================================================

import os
import sys
import csv
import shutil
import zipfile
import random

# ==============================================================================
# Configuration
# ==============================================================================
ZIP_PATH    = r"C:\Users\bordo\Downloads\archive.zip"
EXTRACT_DIR = r"C:\Users\bordo\Downloads\ham10000_raw"
DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
METADATA_CSV = "HAM10000_metadata.csv"
TRAIN_RATIO  = 0.80
RANDOM_SEED  = 42

CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

# ==============================================================================
# Step 1 — Extract ZIP
# ==============================================================================
def extract_zip(zip_path, extract_dir):
    if os.path.exists(extract_dir):
        print(f"  [✓] Raw folder already exists: {extract_dir} — skipping extraction.")
        return

    print(f"\n  Extracting {zip_path}")
    print(f"  -> {extract_dir}")
    print("  (this may take a few minutes for ~5 GB...)\n")

    os.makedirs(extract_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        total = len(zf.infolist())
        for i, member in enumerate(zf.infolist(), 1):
            zf.extract(member, extract_dir)
            if i % 1000 == 0 or i == total:
                pct = i / total * 100
                print(f"    {i:>6}/{total}  ({pct:.1f}%)", end="\r")

    print(f"\n  [✓] Extraction complete.")


# ==============================================================================
# Step 2 — Read Metadata CSV
# ==============================================================================
def read_metadata(extract_dir, csv_name):
    csv_path = os.path.join(extract_dir, csv_name)

    if not os.path.exists(csv_path):
        print(f"  [✗] Metadata CSV not found: {csv_path}")
        sys.exit(1)

    records = []  # list of (image_id, dx_class)
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = row["image_id"].strip()
            dx       = row["dx"].strip().lower()
            if dx in CLASS_NAMES:
                records.append((image_id, dx))

    print(f"  [✓] Metadata loaded: {len(records):,} labelled images across {len(CLASS_NAMES)} classes.")
    return records


# ==============================================================================
# Step 3 — Find Image Files
# ==============================================================================
def build_image_map(extract_dir):
    """
    Walk the extracted directory and build a dict: image_id → full file path.
    Covers HAM10000_images_part_1/ and HAM10000_images_part_2/.
    """
    image_map = {}
    for root, _, files in os.walk(extract_dir):
        for fname in files:
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                image_id = os.path.splitext(fname)[0]
                image_map[image_id] = os.path.join(root, fname)

    print(f"  [✓] Image files found: {len(image_map):,}")
    return image_map


# ==============================================================================
# Step 4 — Stratified Split & Copy
# ==============================================================================
def organise_dataset(records, image_map, dataset_dir, train_ratio, seed):
    random.seed(seed)

    # -------------------------------------------------------------------------
    # Group by class
    # -------------------------------------------------------------------------
    class_buckets = {cls: [] for cls in CLASS_NAMES}
    for image_id, dx in records:
        if image_id in image_map:
            class_buckets[dx].append(image_id)
        # silently skip if image file is missing from zip

    # -------------------------------------------------------------------------
    # Create output directories
    # -------------------------------------------------------------------------
    for split in ["train", "val"]:
        for cls in CLASS_NAMES:
            os.makedirs(os.path.join(dataset_dir, split, cls), exist_ok=True)

    # -------------------------------------------------------------------------
    # Split and copy
    # -------------------------------------------------------------------------
    print(f"\n  Organising images (80/20 split)...\n")
    print(f"  {'Class':<8}  {'Total':>6}  {'Train':>6}  {'Val':>5}")
    print(f"  {'-'*8}  {'-'*6}  {'-'*6}  {'-'*5}")

    total_train = 0
    total_val   = 0

    for cls in CLASS_NAMES:
        ids = class_buckets[cls]
        random.shuffle(ids)

        split_idx  = int(len(ids) * train_ratio)
        train_ids  = ids[:split_idx]
        val_ids    = ids[split_idx:]

        for image_id in train_ids:
            src  = image_map[image_id]
            dst  = os.path.join(dataset_dir, "train", cls, os.path.basename(src))
            shutil.copy2(src, dst)

        for image_id in val_ids:
            src  = image_map[image_id]
            dst  = os.path.join(dataset_dir, "val", cls, os.path.basename(src))
            shutil.copy2(src, dst)

        total_train += len(train_ids)
        total_val   += len(val_ids)

        print(f"  {cls:<8}  {len(ids):>6,}  {len(train_ids):>6,}  {len(val_ids):>5,}")

    print(f"  {'TOTAL':<8}  {total_train+total_val:>6,}  {total_train:>6,}  {total_val:>5,}")


# ==============================================================================
# Main
# ==============================================================================
def main():
    print("\n" + "=" * 60)
    print("  SkinCareAI — HAM10000 Dataset Organiser")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # Check zip exists
    # -------------------------------------------------------------------------
    if not os.path.exists(ZIP_PATH):
        print(f"\n  [✗] archive.zip not found at: {ZIP_PATH}")
        sys.exit(1)

    print(f"\n  archive.zip : {os.path.getsize(ZIP_PATH) / 1e9:.2f} GB")
    print(f"  Extract to  : {EXTRACT_DIR}")
    print(f"  Dataset dir : {DATASET_DIR}")

    # -------------------------------------------------------------------------
    # Run pipeline
    # -------------------------------------------------------------------------
    extract_zip(ZIP_PATH, EXTRACT_DIR)

    records   = read_metadata(EXTRACT_DIR, METADATA_CSV)
    image_map = build_image_map(EXTRACT_DIR)

    organise_dataset(records, image_map, DATASET_DIR, TRAIN_RATIO, RANDOM_SEED)

    # -------------------------------------------------------------------------
    # Done
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  [✓] Dataset ready!")
    print("=" * 60)
    print(f"\n  Output: {DATASET_DIR}")
    print("\n  Train the model:")
    print("    .\\venv\\Scripts\\python.exe src\\train.py")
    print("\n  Launch the app:")
    print("    .\\venv\\Scripts\\streamlit.exe run app.py")
    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
