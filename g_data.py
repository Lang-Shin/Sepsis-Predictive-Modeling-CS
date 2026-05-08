import numpy as np
import pandas as pd


np.random.seed(42)  
n = 1000            # data threshold


# Generate Independent Features
age = np.random.randint(18, 90, n)
infection_hist = np.random.choice(
    [0, 1],
    size = n,
    p = [0.70, 0.30]
)
recent_surgery = np.random.choice(
    [0, 1],
    size = n,
    p = [0.85, 0.15]
)
mental_confusion = np.random.choice(
    [0, 1],
    size = n,
    p = [0.90, 0.10]
)


