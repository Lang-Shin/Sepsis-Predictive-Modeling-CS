import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from sklearn import tree




df = pd.read_csv('data/edu_perp_data.csv')

features = df.iloc[:, :-1]   # features of the data
labels = df.iloc[:, -1]      # labels of the data

X_train, X_test, y_train, y_test = train_test_split(features, labels,
                                                    test_size=0.2, shuffle=True,
                                                    random_state=42)


# d_tree = DecisionTreeClassifier(max_depth=5, random_state=42)
