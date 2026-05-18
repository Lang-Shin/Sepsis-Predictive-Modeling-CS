# Sepsis Prediction using Decision Tree Classifier

## Authors
Created by: **Charles William Bajura** && **Carlo Castro**

## Overview
This project is a Machine Learning-based Sepsis Prediction System built using the **Decision Tree Classifier** algorithm.

The model predicts whether a patient is likely to have **Sepsis** based on medical-related input features from the dataset.

---

## Machine Learning Algorithm Used
- Decision Tree Classifier

The model also uses:
- GridSearchCV for hyperparameter tuning
- Stratified K-Fold Cross Validation
- Confusion Matrix and Classification Report for evaluation

---

## Dataset Information
Dataset file:
```bash
data/edu_perp_data.csv
```

The dataset contains:
- Patient-related medical features
- A target label indicating:
  - `0 = No Sepsis`
  - `1 = Sepsis`

> Disclaimer:
> This dataset is used strictly for educational and research practice purposes only.

---

## Features of the Project
- Data preprocessing using Pandas
- Train/Test Split
- Hyperparameter Optimization using GridSearchCV
- Cross Validation Evaluation
- Recall-focused training for detecting sepsis cases
- Decision Tree Visualization
- Confusion Matrix Heatmap
- Performance Metrics Visualization

---

## Project Structure

```bash
project-folder/
│
├── data/
│   └── edu_perp_data.csv
│
├── model.py
│
└── README.md
```

---

## Model Workflow

### 1. Load Dataset
The dataset is loaded using Pandas.

```python
df = pd.read_csv('data/edu_perp_data.csv')
```

---

### 2. Split Features and Labels

```python
features = df.iloc[:, :-1]
labels = df.iloc[:, -1]
```

---

### 3. Train/Test Split
- 90% Training Data
- 10% Testing Data

```python
train_test_split(...)
```

---

### 4. Hyperparameter Tuning
The model uses **GridSearchCV** to automatically find the best parameters.

Parameters tuned:
- `max_depth`
- `min_samples_split`
- `min_samples_leaf`
- `class_weight`

The scoring metric used is:

```python
scoring='recall'
```

Recall is prioritized because detecting actual sepsis cases is very important.

---

### 5. Cross Validation
The project uses:

```python
StratifiedKFold(n_splits=5)
```

This helps evaluate the model more reliably.

Metrics evaluated:
- Recall
- Accuracy

---

### 6. Prediction and Evaluation

The model generates:
- Confusion Matrix
- Classification Report

Metrics include:
- Precision
- Recall
- F1-Score
- Accuracy

---

### 7. Visualization
The project visualizes:
- Decision Tree Structure
- Confusion Matrix Heatmap
- Performance Metrics Bar Chart

---

## Sample Output

### Cross Validation Output

```bash
--- Cross-Validation Results (5-Fold) ---
Recall per fold : [0.70 0.73 0.69 0.74 0.71]
Mean Recall     : 0.71 ± 0.02

Accuracy per fold : [0.71 0.73 0.70 0.74 0.72]
Mean Accuracy     : 0.72 ± 0.01
```

---

## Educational Purpose Notice

This project is intended only for:
- Machine Learning practice
- Educational demonstrations
- Academic projects
- Research learning

---

