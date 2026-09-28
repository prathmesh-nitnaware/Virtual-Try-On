# 🧪 V-TryOn Machine Learning & Model Training

<div align="center">

![PyTorch](https://img.shields.io/badge/PyTorch-2.2-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![DeepFashion](https://img.shields.io/badge/Dataset-DeepFashion-blue?style=for-the-badge)
![CP-VTON+](https://img.shields.io/badge/Architecture-CP--VTON%2B-green?style=for-the-badge)
![SCHP](https://img.shields.io/badge/Parsing-SCHP%20ATR-purple?style=for-the-badge)

**Deep learning pipelines, model training notebooks, and evaluation workflows for virtual try-on and fashion classification.**

</div>

---

## 🧠 Integrated Models & Datasets

### 1. DeepFashion ResNet-50 Classifier
- **Notebook:** `train_deepfashion.ipynb`
- **Architecture:** ResNet-50 with custom classification head for 50 fine-grained apparel categories.
- **Mapping:** `class_to_idx.json` defines category index mappings for top-5 ranking.

### 2. VITON Neural Try-On U-Net
- **Directory:** `VITON_Training/`
- **Files:** `viton_training.ipynb`, `viton_training.py`
- **Architecture:** Multi-scale U-Net trained on paired garment and model images with L1 and perceptual loss.

### 3. CP-VTON+ Neural Draping Pipeline (Phase 4A)
- **SCHP (Self-Correction Human Parsing):** 18-class semantic body segmentation.
- **HRNet-W32:** 18-keypoint body pose estimation.
- **GMM (Geometric Matching Module):** Thin-Plate Spline (TPS) cloth transformation network.
- **TOM (Try-On Module):** Composition and neural rendering synthesis network.

---

## 📊 Training & Evaluation

To inspect or reproduce model training:
- Open `train_deepfashion.ipynb` in Jupyter Notebook or Google Colab for classification training.
- Open `VITON_Training/viton_training.ipynb` for U-Net try-on experimentation.
