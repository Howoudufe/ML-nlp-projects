import pandas as pd
import numpy as np

df = pd.read_csv('fake_job_postings.csv')

required_education = ["Bachelor's Degree", "Master's Degree", "High School or equivalent"]

# Filter
filtered_df = df[(df['fraudulent'] == 0) & (df['required_education'].isin(required_education))]

# Random choose
sample = filtered_df.sample(n=50, random_state=42)

sample_true = sample[['title', 'description', 'required_education']]
sample_test = sample[['title', 'description']]

# Save as csv files
sample_true.to_csv('50_true.csv', index=False)
sample_test.to_csv('50_test.csv', index=False)

print("New subsets have been saved.")

