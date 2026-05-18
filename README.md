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
