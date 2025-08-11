# 🩺 Skin Cancer Detection Using Hybrid CNN-XGBoost

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.6%2B-brightgreen?logo=xgboost&logoColor=white)](https://xgboost.ai/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/jayavanth18/Skin-Cancer-Detection?style=social)](https://github.com/jayavanth18/Skin-Cancer-Detection)

**A hybrid deep learning pipeline combining EfficientNetB4 feature extraction with XGBoost classification for skin lesion analysis using the HAM10000 dataset.**

</div>

---

## 🌟 Overview

This project implements a **two-stage skin cancer classification system**:

1. **Feature Extraction** → EfficientNetB4 (ImageNet pretrained, frozen)  
   - Outputs **1,792-dimensional** feature vectors from dermatoscopic images.

2. **Classification** → XGBoost  
   - Multi-class classification for **7 skin lesion types**.
   - Handles **imbalanced medical datasets** more robustly than pure CNN models.

**Performance Achieved:**  
- **Test Accuracy:** 75.84%  
- **Weighted F1-score:** 0.73  

⚠ **Disclaimer:** This project is for **research & educational purposes only**. Not suitable for clinical use.

---

## 📊 Results

| Metric          | Score  |
|-----------------|--------|
| **Test Accuracy** | **75.84%** |
| Macro F1        | 0.42   |
| Weighted F1     | 0.73   |

**Per-Class Breakdown**
```

nv      0.83/0.95/0.88  ✅ Best
bkl     0.54/0.51/0.52
mel     0.46/0.39/0.42
bcc     0.64/0.31/0.42
vasc    1.00/0.39/0.56
akiec   0.50/0.09/0.16
df      0.00/0.00/0.00  ❌ Worst

```
*(precision / recall / F1)*

---

## 🧠 Why Hybrid CNN + XGBoost?

| CNN Alone ❌ | Hybrid ✅ |
|-------------|----------|
| Overfits small datasets | Reuses pretrained EfficientNet features |
| Poor minority-class accuracy | XGBoost handles imbalance better |
| Long training times | Faster (no full CNN backprop) |
| Lower interpretability | XGBoost offers feature importance |

---

## 🔧 Technical Pipeline

**Data**
- **Dataset:** [HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) — 10,015 dermatoscopic images  
- **Split:** 8,012 train / 2,003 test  
- **Size:** 380×380×3  
- **Augmentations:** Rotation, flips, zoom, shifts

**Architecture**
```

Input Image → EfficientNetB4 (frozen) → GAP → 1,792-dim features → XGBoost → Prediction

````

**XGBoost Params**
```python
n_estimators = 300
learning_rate = 0.05
max_depth = 6
subsample = 0.8
colsample_bytree = 0.8
````

---

## 📈 Insights

✅ **Strengths**

* Strong Nevus detection (83% precision, 95% recall)
* Transfer learning captures visual lesion patterns
* Training completes in minutes

⚠ **Challenges**

* Rare classes (df, akiec) perform poorly
* Clinical deployment requires >90% sensitivity for malignant classes

---

## 🚀 Future Work

* **Data:** Oversampling (SMOTE), targeted augmentation, more external datasets
* **Model:** Ensemble multiple CNN backbones, fine-tune EfficientNet layers
* **Evaluation:** Cross-validation, focus on melanoma sensitivity/specificity

---

## 🛠 Installation & Usage

### 1️⃣ Clone the Repo

```bash
git clone https://github.com/jayavanth18/Skin-Cancer-Detection.git
cd Skin-Cancer-Detection
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Train & Evaluate

```python
# 1. Download HAM10000 via kagglehub
# 2. Extract features with EfficientNetB4
# 3. Train XGBoost classifier
# 4. Evaluate model on test set
```

### 4️⃣ Predict Single Image

```python
def predict_lesion(image_path):
    features = efficientnet_extractor.predict(image)
    pred = xgboost_model.predict(features)
    return lesion_type_mapping[pred]
```

---

## 📦 Requirements

```
tensorflow>=2.10.0
efficientnet>=1.1.1
xgboost>=1.6.0
scikit-learn>=1.0.0
kagglehub>=0.1.0
```

---

## 👨‍💻 Author

<div align="center">

**[A. Jayavanth](https://github.com/jayavanth18)**

[![GitHub](https://img.shields.io/badge/GitHub-jayavanth18-black?logo=github\&logoColor=white)](https://github.com/jayavanth18)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-jayavanth-blue?logo=linkedin\&logoColor=white)](https://www.linkedin.com/in/jayavanth/)

</div>

---

<div align="center">

**⭐ If you found this project useful, please star the repo! ⭐**

</div>
```

---

This one will render **all badges correctly** on GitHub.
Do you want me to now **add a social preview image banner** so that when you share the repo link it shows a clean thumbnail? That’ll make it even more attractive when shared.
