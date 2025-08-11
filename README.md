# 🔬 Medical Image Classification: CNN + XGBoost Hybrid



[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColorimg.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow&logoColor=io/badge/XGBoost-1.6%2B-brightgreene](https://img.](https://img.shields.io/badge the traditional CNN mold: A two-stage pipeline that combines deep feature extraction with gradient boosting for dermatology image classification.**



***

## 🎯 The Big Picture

Think of this as **feature extraction meets traditional ML**. Instead of training a massive end-to-end neural network (which often fails on medical data), we split the problem:

1. **Extract rich features** using a pre-trained EfficientNetB4
2. **Classify intelligently** using XGBoost on those features

**Result:** 75.84% accuracy on HAM10000 dataset with faster training and better interpretability.

***

## 📈 Performance Snapshot

| **Metric** | **Our Score** |
|------------|---------------|
| Test Accuracy | **75.84%** |
| Macro F1 | 0.42 |
| Weighted F1 | 0.73 |

**What's Working vs. What's Not:**

| Class Type | Performance | Reality Check |
|------------|-------------|---------------|
| **Nevus** (common) | 🟢 88% F1 | Solid detection |
| **Melanoma** (critical) | 🟡 42% F1 | Needs improvement |
| **Rare lesions** | 🔴  && cd skin-cancer-hybrid && pip install -r requirements.txt
```

**Core prediction function:**
```python
def classify_lesion(image_path):
    # Extract features using frozen EfficientNetB4
    features = feature_extractor.predict(preprocess_image(image_path))
    
    # Classify with trained XGBoost
    prediction = xgb_model.predict(features)
    confidence = xgb_model.predict_proba(features).max()
    
    return lesion_types[prediction], confidence
```

***

## 📋 Dependencies

```txt
tensorflow>=2.10.0    # CNN backbone
xgboost>=1.6.0       # Classifier
efficientnet>=1.1.1  # Pre-trained model
scikit-learn>=1.0.0  # Metrics & preprocessing
kagglehub>=0.1.0     # Dataset access
```

***

## ⚠️ Medical Disclaimer

**This is research code, period.**  
- ❌ Not FDA approved
- ❌ Not for clinical decisions  
- ✅ Educational use only
- ✅ Computer vision portfolio showcase

***

## 🙋♂️ Get in Touch

**Author:** [Your Name]  
**Connect:** [LinkedIn](https://linkedin.com/in/yourprofile) | [Email](mailto:your.email@domain.com)

***

**🌟 Found this approach interesting? Star the repo and let's discuss hybrid ML strategies!**
