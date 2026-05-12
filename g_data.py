import numpy as np
import pandas as pd


np.random.seed(42)  
n = 1500           # data threshold


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


# Generate Lab Metrics (char lab results)
wbc_count = np.random.normal(8.5, 4.0, n)
wbc_count = np.clip(wbc_count, 3000, 25000)

lactate = np.random.normal(1.2, 0.8, n)
lactate = np.clip(lactate, 0.5, 8.0)


# Clinical risk logic
risk_score = (
    (temp > 38.0).astype(int) * 2 +

    (heart_rate > 100).astype(int) * 1.5 +

    (respiratory_rate > 22).astype(int) * 2 +

    (systolic_BP < 100).astype(int) * 2 +

    (oxygen_sat < 94).astype(int) * 1.5 +

    (wbc_count > 12000).astype(int) * 2 +

    (lactate > 2.0).astype(int) * 3 +

    (mental_confusion * 2) +

    (recent_surgery * 1) +

    (infection_hist * 1.5)
)

# Using sigmoid function
prob = 1 / (1 + np.exp(-(risk_score - 4)))

sepsis = np.random.binomial(1, prob)

df = pd.DataFrame({
    'age' : age,
    'temperature' : np.round(temp, 1),
    'heart_rate' : np.round(heart_rate, 0),
    'respiratory_rate' : np.round(respiratory_rate, 0),
    'systolic_bp' : np.round(systolic_BP, 0),
    'oxygen_sat' : np.round(oxygen_sat, 1),
    'wbc' : np.round(wbc_count, 0),
    'lactate' : np.round(lactate, 1),
    'infection_hist' : infection_hist,
    'mental_confusion' : mental_confusion,
    'recent_surgery' : recent_surgery,
    'sepsis' : sepsis
})


# Save dataframe into csv
df.to_csv('data/edu_perp_data.csv', index=False)

print("Data created successfully")
print("Stored in data folder")