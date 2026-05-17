import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from sklearn import tree
import seaborn as sns


df = pd.read_csv('data/edu_perp_data.csv')

features = df.iloc[:, :-1]   # features of the data
labels = df.iloc[:, -1]      # labels of the data

X_train, X_test, y_train, y_test = train_test_split(features, labels,
                                                    test_size=0.1, 
                                                    shuffle=True,
                                                    stratify=labels,
                                                    random_state=42)


# GridSearchCV 
param_grid = {
    'max_depth': [3, 4, 5, 6, 7, 8],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'class_weight': ['balanced', None]   # 'balanced' helps with class imbalance
}

grid_search = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='recall',   # prioritize catching actual sepsis cases
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
print("Best Hyperparameters:", grid_search.best_params_)

# Use the best model found by GridSearchCV
d_tree = grid_search.best_estimator_


# Cross-Validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

recall_scores = cross_val_score(d_tree, features, labels, cv=cv, scoring='recall')
accuracy_scores = cross_val_score(d_tree, features, labels, cv=cv, scoring='accuracy')

print("\n--- Cross-Validation Results (5-Fold) ---")
print(f"Recall    per fold : {np.round(recall_scores, 2)}")
print(f"Mean Recall        : {recall_scores.mean():.2f} ± {recall_scores.std():.2f}")
print(f"Accuracy  per fold : {np.round(accuracy_scores, 2)}")
print(f"Mean Accuracy      : {accuracy_scores.mean():.2f} ± {accuracy_scores.std():.2f}")

y_pred = d_tree.predict(X_test)

print("\n--- Test Set Results ---")
print("\n", confusion_matrix(y_test, y_pred))
print("\n\n", classification_report(y_test, y_pred))


# Visualization
plt.figure(figsize=(12, 7))
tree.plot_tree( d_tree, 
                max_depth=3,
                feature_names=features.columns,
                class_names=['No Sepsis', 'Sepsis'],
                filled=True, 
                fontsize=8)
plt.title("Decision Tree Structure for Sepsis Prediction (Truncated to Depth 3)")
plt.show()


fig, axes = plt.subplots(1, 2, figsize=(12, 5))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Sepsis', 'Sepsis'],
            yticklabels=['No Sepsis', 'Sepsis'],
            ax=axes[0])
axes[0].set_title('Confusion Matrix')
axes[0].set_ylabel('Actual')
axes[0].set_xlabel('Predicted')

# 4. Use 1D indexing for the second plot
desc = ['Precision', 'Recall', 'F1-Score', 'Accuracy']
scale = [0.68, 0.71, 0.69, 0.72]
axes[1].bar(desc, scale, color="#2e7742", edgecolor="#000000") # Added dummy data so .plot() works
axes[1].set_title('Second Plot')
axes[1].grid(True, alpha=0.7)
axes[1].set_axisbelow(True)

plt.tight_layout() # Keeps things from overlapping
plt.show()