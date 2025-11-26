"""
Run preprocessing steps to generate cleaned_processed.cleveland.csv
This script executes the key data cleaning steps from preprocessing_nb.ipynb
"""

import pandas as pd
import numpy as np

print("="*70)
print("RUNNING DATA PREPROCESSING")
print("="*70)

# Step 1: Load raw data
print("\n1. Loading raw data from data/processed.cleveland.data...")
cols = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
    'restecg', 'thalach', 'exang', 'oldpeak', 'slope',
    'ca', 'thal', 'target'
]

df = pd.read_csv("data/processed.cleveland.data", names=cols, na_values='?')
print(f"   ✓ Loaded {df.shape[0]} rows, {df.shape[1]} columns")

# Step 2: Check for missing values
print("\n2. Checking for missing values...")
missing_before = df.isnull().sum()
print(f"   Missing values found:")
print(missing_before[missing_before > 0])

# Step 3: Drop rows with missing values
print("\n3. Dropping rows with missing values...")
df = df.dropna()
print(f"   ✓ Dataset now has {df.shape[0]} rows")

# Step 4: Convert all columns to numeric
print("\n4. Converting all columns to numeric types...")
df = df.apply(pd.to_numeric)
print(f"   ✓ All columns converted")

# Step 5: Simplify target variable (binary classification)
print("\n5. Converting target to binary classification...")
print(f"   Original target distribution:")
print(df['target'].value_counts().sort_index())
df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)
print(f"   Binary target distribution:")
print(df['target'].value_counts())

# Step 6: Remove duplicates
print("\n6. Removing duplicate records...")
rows_before = df.shape[0]
df = df.drop_duplicates()
rows_after = df.shape[0]
duplicates_removed = rows_before - rows_after
print(f"   ✓ Removed {duplicates_removed} duplicate(s)")
print(f"   ✓ Final dataset: {rows_after} rows")

# Step 7: Final sanity checks
print("\n7. Final sanity checks...")
print(f"   Missing values: {df.isnull().sum().sum()}")
print(f"   Data types: All numeric ✓")
print(f"   Shape: {df.shape}")

# Step 8: Save cleaned data
print("\n8. Saving cleaned data to cleaned_processed.cleveland.csv...")
df.to_csv("cleaned_processed.cleveland.csv", index=False)
print(f"   ✓ File saved successfully!")

# Display summary
print("\n" + "="*70)
print("PREPROCESSING COMPLETE!")
print("="*70)
print(f"\nCleaned dataset summary:")
print(f"  • Total records: {df.shape[0]}")
print(f"  • Features: {df.shape[1] - 1}")
print(f"  • Target variable: binary (0 = No Disease, 1 = Disease)")
print(f"  • No missing values")
print(f"  • No duplicates")
print(f"\nClass distribution:")
print(df['target'].value_counts())
print(f"\nFile saved: cleaned_processed.cleveland.csv")
print("\n✓ Ready for modeling!")
print("="*70)
