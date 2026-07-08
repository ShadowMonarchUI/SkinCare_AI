# 🔬 SkinCareAI — Dermatological Screening Assistant

A Deep Learning web application that classifies **7 types of skin lesions** from dermoscopy images using **Transfer Learning with ResNet50**, built on the [HAM10000 dataset](https://www.kaggle.com/datasets/kmader/skin-lesion-analysis-toward-melanoma-detection).

---

## 📋 Features

- Deep Learning based Skin Lesion Classification (7 classes)
- Transfer Learning using ResNet50 (ImageNet Weights)
- 2-Phase Training: Head Training + Fine-Tuning
- Data Augmentation for improved generalisation
- Class Imbalance Handling via Class Weights
- Model Evaluation with Classification Report & Confusion Matrix
- Interactive Streamlit Web Application
- Upload skin lesion image and receive instant triage assessment
- Colour-coded Triage System (🔴 Urgent / 🟡 Monitor / 🟢 Low Risk)
- Confidence score and per-class probability breakdown

---

## 🖼️ Demo

### Streamlit Interface

Upload a skin lesion image → Analyse → Get AI-assisted triage result

---

## 📂 Dataset

Dataset: **HAM10000 (Human Against Machine with 10000 training images)**

Classes:

| ID | Code  | Full Name                                     | Triage     |
|----|-------|-----------------------------------------------|------------|
| 0  | akiec | Actinic Keratoses / Intraepithelial Carcinoma | 🔴 URGENT  |
| 1  | bcc   | Basal Cell Carcinoma                          | 🔴 URGENT  |
| 2  | bkl   | Benign Keratosis-like Lesions                 | 🟡 MONITOR |
| 3  | df    | Dermatofibroma                                | 🟡 MONITOR |
| 4  | mel   | Melanoma                                      | 🔴 URGENT  |
| 5  | nv    | Melanocytic Nevi (Mole)                       | 🟢 LOW RISK|
| 6  | vasc  | Vascular Lesions                              | 🟢 LOW RISK|

Dataset Structure:

```
dataset/
    train/
        akiec/   bcc/   bkl/   df/   mel/   nv/   vasc/
    val/
        akiec/   bcc/   bkl/   df/   mel/   nv/   vasc/
```

---

## 🧠 Model Architecture

Transfer Learning with **ResNet50**

```
Input Image (224×224×3)
        │
ResNet50 (ImageNet Weights, frozen in Phase 1)
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
```

---

## ⚙️ Technologies Used

- Python
- TensorFlow
- Keras
- Streamlit
- NumPy
- Matplotlib
- Scikit-learn
- Seaborn
- Pillow

---

## 🚀 Setup & Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Dataset

Download HAM10000 from Kaggle and organise into the directory structure above.

### 3. Train the Model

```bash
python src/train.py
```

With custom options:

```bash
python src/train.py --data_dir dataset --epochs 30 --batch_size 32 --lr 0.0001
```

| Argument           | Default                  | Description                              |
|--------------------|--------------------------|------------------------------------------|
| `--data_dir`       | `dataset`                | Root data directory                      |
| `--epochs`         | `30`                     | Head training epochs                     |
| `--fine_tune_epochs`| `10`                    | Fine-tuning epochs                       |
| `--batch_size`     | `32`                     | Batch size                               |
| `--lr`             | `1e-4`                   | Initial learning rate                    |
| `--fine_tune_lr`   | `1e-5`                   | Fine-tuning learning rate                |
| `--dropout`        | `0.3`                    | Dropout rate                             |
| `--output`         | `models/skin_model.keras`| Model output path                        |
| `--no_fine_tune`   | flag                     | Skip fine-tuning phase                   |

### 4. Run the Streamlit App

```bash
streamlit run app.py
```

---

## 📊 Model Performance

| Metric        | Value        |
|---------------|:------------:|
| Architecture  | ResNet50     |
| Input Size    | 224 × 224    |
| Classes       | 7            |
| Training Data | HAM10000     |

---

## ⚕️ Clinical Disclaimer

SkinCareAI is an AI-assisted screening tool for **educational and research purposes only**. It does not constitute medical advice and must not be used as a substitute for professional clinical diagnosis. Always consult a qualified dermatologist for medical evaluation.

---

Developed using ResNet50 Transfer Learning on the HAM10000 dataset | Streamlit + TensorFlow
