# Concrete Wall Defect Classification using EfficientNet Features & ML

## Overview

ML-based classification of concrete wall defects into three classes:

* Crack
* Honeycomb
* Spalling

The project uses **EfficientNetB0** as a feature extractor and evaluates classical ML classifiers on the extracted features.

## Dataset

| Class     |    Images |
| --------- | --------: |
| Crack     |     7,305 |
| Honeycomb |     1,872 |
| Spalling  |       650 |
| **Total** | **9,827** |

Dataset split using stratified sampling:

* Training: **70%**
* Validation: **15%**
* Testing: **15%**
* Random seed: **42**

## Approach

```text
Input Images
      ↓
Image Preprocessing
      ↓
EfficientNetB0 Feature Extraction
      ↓
Feature Vectors
      ↓
ML Classification
      ↓
Defect Class Prediction
```

Models evaluated:

* Logistic Regression
* Random Forest
* Decision Tree

## Validation Results

| Model                   |   Accuracy |   Macro F1 |
| ----------------------- | ---------: | ---------: |
| **Logistic Regression** | **99.03%** |   **~98%** |
| Random Forest           | **97.00%** |   **~93%** |
| Decision Tree           | **86.93%** | **77.99%** |

### Best Model

**Logistic Regression** currently provides the best validation performance.

## Decision Tree Analysis

The initial Decision Tree showed significant overfitting:

* Training Accuracy: **100%**
* Validation Accuracy: **87.23%**
* Training Macro F1: **100%**
* Validation Macro F1: **77.39%**

A controlled tree was then evaluated using depth and minimum-sample constraints.

Best configuration:

```text
max_depth = 15
min_samples_split = 10
min_samples_leaf = 5
class_weight = balanced
```

Result:

* Validation Accuracy: **86.93%**
* Validation Macro F1: **77.99%**

The Decision Tree performed strongly on **Crack**, but had comparatively lower performance on **Spalling**.

## Project Structure

```text
├── ML_Wall_Defect_Classification.ipynb
├── app.py
├── master_dataset.csv
├── wall_defect_dataset_split.csv
├── .gitignore
└── README.md
```

Raw datasets and generated feature files are excluded from the repository.

## Tech Stack

Python • TensorFlow • EfficientNetB0 • Scikit-learn • Pandas • NumPy • Matplotlib • Streamlit

## Status

Feature extraction and ML model comparison completed.
**Final test-set evaluation and application integration are the next steps.**
