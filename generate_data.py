import pandas as pd
import numpy as np

np.random.seed(42)
n_records = 3000

data = {
    'Age': np.random.randint(22, 60, n_records),
    'MonthlyIncome': np.random.randint(2500, 15000, n_records),
    'JobSatisfaction': np.random.randint(1, 5, n_records),
    'WorkLifeBalance': np.random.randint(1, 5, n_records),
    'YearsAtCompany': np.random.randint(0, 20, n_records),
    'OverTime': np.random.choice(['Yes', 'No'], n_records, p=[0.3, 0.7])
}

df = pd.DataFrame(data)
risk_score = (
    (df['OverTime'] == 'Yes').astype(int) * 2.5 +
    (df['MonthlyIncome'] < 5000).astype(int) * 2.0 +
    (df['JobSatisfaction'] == 1).astype(int) * 1.5 +
    (df['WorkLifeBalance'] == 1).astype(int) * 1.5 -
    (df['YearsAtCompany'] > 10).astype(int) * 0.5
)
df['Attrition'] = np.where(risk_score + np.random.normal(0, 1, n_records) > 2.5, 1, 0)

# Save directly to the current directory
df.to_csv('employee_data.csv', index=False)
print("Dataset created successfully in the root folder!")