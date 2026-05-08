import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree


df = pd.read_csv('data/edu_perp_data.csv')

print(df)
print("\n\n", (df['infection_hist'] == 1).sum())
print((df['mental_confusion'] == 1).sum())
print((df['recent_surgery'] == 1).sum())
print((df['sepsis'] == 1).sum())