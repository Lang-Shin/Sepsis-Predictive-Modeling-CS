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


# Generate Vital Signs
temp = np.random.normal(37.0, 0.8, n)   # loc, scale, size
temp = np.clip(temp, 35.0, 41.0)        # var, min, max

heart_rate = np.random.normal(82, 18, n)
heart_rate = np.clip(heart_rate, 50, 160)

respiratory_rate = np.random.normal(17, 5, n)
repiratory_rate = np.clip(respiratory_rate, 10, 40)

systolic_BP = np.random.normal(118, 18, n)
systolic_BP = np.clip(systolic_BP, 70, 180)

oxygen_sat = np.random.normal(97, 2.5, n)
oxygen_sat = np.clip(oxygen_sat, 75, 100)