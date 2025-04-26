import pandas as pd
import os

# Define paths
FAKE_PATH = 'attached_assets/Fake.csv'
TRUE_PATH = 'attached_assets/True.csv'
SAMPLE_FAKE_PATH = 'dataset/Fake_small.csv'
SAMPLE_TRUE_PATH = 'dataset/True_small.csv'

# Ensure the dataset directory exists
os.makedirs('dataset', exist_ok=True)

# Create a smaller sample dataset for faster training
print("Creating very small sample dataset for quick training...")

# Load the original datasets
fake_df = pd.read_csv(FAKE_PATH)
true_df = pd.read_csv(TRUE_PATH)

print(f"Original fake news dataset: {len(fake_df)} rows")
print(f"Original true news dataset: {len(true_df)} rows")

# Create balanced samples (1000 from each)
sample_size = 1000
fake_sample = fake_df.sample(sample_size, random_state=42)
true_sample = true_df.sample(sample_size, random_state=42)

print(f"Sample fake news dataset: {len(fake_sample)} rows")
print(f"Sample true news dataset: {len(true_sample)} rows")

# Save the samples
fake_sample.to_csv(SAMPLE_FAKE_PATH, index=False)
true_sample.to_csv(SAMPLE_TRUE_PATH, index=False)

print(f"Small sample datasets saved to {SAMPLE_FAKE_PATH} and {SAMPLE_TRUE_PATH}")