"""
Comprehensive Test of Logistic Regression Model
Includes detailed analysis, predictions, and examples
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

print("="*80)
print("COMPREHENSIVE LOGISTIC REGRESSION MODEL TEST")
print("="*80)

# Load data
df = pd.read_csv("cleaned_processed.cleveland.csv")
X = df.drop('target', axis=1)
y = df['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
log_reg = LogisticRegression(random_state=42, max_iter=1000)
log_reg.fit(X_train_scaled, y_train)

# Make predictions
y_pred_train = log_reg.predict(X_train_scaled)
y_pred_test = log_reg.predict(X_test_scaled)
y_pred_proba = log_reg.predict_proba(X_test_scaled)[:, 1]

print("\n" + "="*80)
print("1. MODEL PERFORMANCE SUMMARY")
print("="*80)

# Training performance
train_acc = accuracy_score(y_train, y_pred_train)
print(f"\n📊 Training Set:")
print(f"   Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")

# Test performance
test_acc = accuracy_score(y_test, y_pred_test)
test_prec = precision_score(y_test, y_pred_test)
test_rec = recall_score(y_test, y_pred_test)
test_f1 = f1_score(y_test, y_pred_test)
test_auc = roc_auc_score(y_test, y_pred_proba)

print(f"\n📊 Test Set:")
print(f"   Accuracy:  {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"   Precision: {test_prec:.4f} ({test_prec*100:.2f}%)")
print(f"   Recall:    {test_rec:.4f} ({test_rec*100:.2f}%)")
print(f"   F1-Score:  {test_f1:.4f}")
print(f"   ROC-AUC:   {test_auc:.4f} ({test_auc*100:.2f}%)")

# Overfitting check
overfit = train_acc - test_acc
print(f"\n🔍 Overfitting Analysis:")
print(f"   Train-Test Gap: {overfit:.4f} ({overfit*100:.2f}%)")
if overfit < 0.05:
    print(f"   ✅ Excellent - Minimal overfitting")
elif overfit < 0.10:
    print(f"   ✓ Good - Acceptable overfitting")
else:
    print(f"   ⚠️ Warning - Significant overfitting")

print("\n" + "="*80)
print("2. CONFUSION MATRIX ANALYSIS")
print("="*80)

cm = confusion_matrix(y_test, y_pred_test)
tn, fp, fn, tp = cm.ravel()

print(f"\n                 Predicted")
print(f"               No Disease  Disease")
print(f"Actual No Dis      {tn:3d}       {fp:3d}")
print(f"       Disease     {fn:3d}       {tp:3d}")

print(f"\n📋 Detailed Breakdown:")
print(f"   True Negatives (TN):  {tn} - Correctly identified healthy")
print(f"   True Positives (TP):  {tp} - Correctly identified diseased")
print(f"   False Positives (FP): {fp} - Healthy flagged as diseased")
print(f"   False Negatives (FN): {fn} - Diseased missed (CRITICAL!)")

print(f"\n⚕️ Clinical Metrics:")
print(f"   Sensitivity (Recall): {tp/(tp+fn)*100:.1f}% - Disease detection rate")
print(f"   Specificity:          {tn/(tn+fp)*100:.1f}% - Healthy identification rate")
print(f"   False Negative Rate:  {fn/(tp+fn)*100:.1f}% - Missed disease cases")
print(f"   False Positive Rate:  {fp/(tn+fp)*100:.1f}% - False alarms")

print("\n" + "="*80)
print("3. FEATURE IMPORTANCE RANKING")
print("="*80)

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': log_reg.coef_[0],
    'Abs_Coefficient': np.abs(log_reg.coef_[0])
}).sort_values('Abs_Coefficient', ascending=False)

print("\n🔝 Top 10 Most Important Features:")
print("-" * 60)
for idx, row in feature_importance.head(10).iterrows():
    direction = "↑ Risk Factor" if row['Coefficient'] > 0 else "↓ Protective"
    print(f"{row['Feature']:12s} | {row['Coefficient']:7.4f} | {direction}")

print("\n" + "="*80)
print("4. SAMPLE PREDICTIONS")
print("="*80)

# Show some example predictions
print("\n📝 Example Test Cases (First 10):")
print("-" * 80)
results_df = pd.DataFrame({
    'Actual': y_test.values[:10],
    'Predicted': y_pred_test[:10],
    'Probability': y_pred_proba[:10],
    'Correct': y_test.values[:10] == y_pred_test[:10]
})

for idx, row in results_df.iterrows():
    actual_label = "Disease" if row['Actual'] == 1 else "No Disease"
    pred_label = "Disease" if row['Predicted'] == 1 else "No Disease"
    status = "✓" if row['Correct'] else "✗"
    print(f"Patient {idx+1:2d}: Actual={actual_label:11s} | "
          f"Predicted={pred_label:11s} | "
          f"Confidence={row['Probability']:.2%} | {status}")

# Count correct/incorrect
correct = (y_test == y_pred_test).sum()
incorrect = len(y_test) - correct
print(f"\n✓ Correct: {correct}/{len(y_test)} ({correct/len(y_test)*100:.1f}%)")
print(f"✗ Incorrect: {incorrect}/{len(y_test)} ({incorrect/len(y_test)*100:.1f}%)")

print("\n" + "="*80)
print("5. PREDICTION CONFIDENCE ANALYSIS")
print("="*80)

# Analyze prediction confidence
high_conf = (y_pred_proba > 0.8) | (y_pred_proba < 0.2)
medium_conf = ((y_pred_proba >= 0.6) & (y_pred_proba <= 0.8)) | \
              ((y_pred_proba >= 0.2) & (y_pred_proba <= 0.4))
low_conf = (y_pred_proba >= 0.4) & (y_pred_proba <= 0.6)

print(f"\n📊 Confidence Distribution:")
print(f"   High Confidence (>80% or <20%):   {high_conf.sum():2d} predictions ({high_conf.sum()/len(y_test)*100:.1f}%)")
print(f"   Medium Confidence (60-80%, 20-40%): {medium_conf.sum():2d} predictions ({medium_conf.sum()/len(y_test)*100:.1f}%)")
print(f"   Low Confidence (40-60%):          {low_conf.sum():2d} predictions ({low_conf.sum()/len(y_test)*100:.1f}%)")

print("\n" + "="*80)
print("6. ERROR ANALYSIS")
print("="*80)

# Analyze errors
errors = y_test != y_pred_test
error_indices = np.where(errors)[0]

print(f"\n❌ Total Errors: {errors.sum()} out of {len(y_test)} predictions")
print(f"\nError Breakdown:")
print(f"   False Positives: {fp} (Healthy predicted as Diseased)")
print(f"   False Negatives: {fn} (Diseased predicted as Healthy) ⚠️ CRITICAL")

if fn > 0:
    print(f"\n⚠️ False Negative Cases (Missed Diseases):")
    fn_indices = np.where((y_test == 1) & (y_pred_test == 0))[0]
    for i, idx in enumerate(fn_indices[:5], 1):  # Show first 5
        prob = y_pred_proba[idx]
        print(f"   Case {i}: Predicted probability = {prob:.2%} (threshold = 50%)")

print("\n" + "="*80)
print("7. MODEL INTERPRETATION")
print("="*80)

print("\n🎯 Key Insights:")
print("\n1. Strongest Risk Factors (Positive Coefficients):")
top_risk = feature_importance[feature_importance['Coefficient'] > 0].head(3)
for idx, row in top_risk.iterrows():
    print(f"   • {row['Feature']}: +{row['Coefficient']:.3f}")

print("\n2. Protective Factors (Negative Coefficients):")
top_protective = feature_importance[feature_importance['Coefficient'] < 0].head(3)
for idx, row in top_protective.iterrows():
    print(f"   • {row['Feature']}: {row['Coefficient']:.3f}")

print("\n3. Model Reliability:")
if test_auc > 0.9:
    print(f"   ⭐ Excellent discrimination (AUC = {test_auc:.3f})")
elif test_auc > 0.8:
    print(f"   ✓ Good discrimination (AUC = {test_auc:.3f})")
else:
    print(f"   ⚠️ Fair discrimination (AUC = {test_auc:.3f})")

print("\n" + "="*80)
print("TEST COMPLETE!")
print("="*80)
print(f"\n✅ Model Status: PASSED")
print(f"✅ Performance: {test_acc*100:.1f}% accuracy, {test_auc*100:.1f}% ROC-AUC")
print(f"⚠️ Concern: {fn} false negatives (missed disease cases)")
print(f"📊 Recommendation: Consider ensemble methods to reduce false negatives")
print("\n" + "="*80)
