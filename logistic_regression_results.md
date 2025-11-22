# Logistic Regression Model Results

## Model Performance Summary

### Dataset Split
- **Training Set**: 237 samples (124 no disease, 113 disease)
- **Testing Set**: 60 samples (36 no disease, 24 disease)

### Performance Metrics

#### Test Set Results
- **Accuracy**: 88.33%
- **Precision**: 84.00%
- **Recall**: 87.50%
- **F1-Score**: 85.71%
- **ROC-AUC Score**: 93.29%

#### Training Set Results
- **Accuracy**: 86.50%

### Confusion Matrix (Test Set)
```
                Predicted
              No Disease  Disease
Actual
No Disease        32         4
Disease            3        21
```

**Interpretation:**
- **True Negatives (TN)**: 32 - Correctly predicted no disease
- **False Positives (FP)**: 4 - Incorrectly predicted disease
- **False Negatives (FN)**: 3 - Incorrectly predicted no disease
- **True Positives (TP)**: 21 - Correctly predicted disease

### Classification Report
```
              precision    recall  f1-score   support

  No Disease       0.91      0.89      0.90        36
     Disease       0.84      0.88      0.86        24

    accuracy                           0.88        60
   macro avg       0.88      0.88      0.88        60
weighted avg       0.88      0.88      0.88        60
```

## Feature Importance (Top 10)

Features ranked by absolute coefficient value:

1. **ca** (1.152) - Number of major vessels colored by fluoroscopy
   - *Positive correlation*: More vessels → Higher disease risk

2. **sex** (0.722) - Gender (1 = male, 0 = female)
   - *Positive correlation*: Males have higher disease risk

3. **trestbps** (0.514) - Resting blood pressure
   - *Positive correlation*: Higher BP → Higher disease risk

4. **oldpeak** (0.436) - ST depression induced by exercise
   - *Positive correlation*: Higher ST depression → Higher disease risk

5. **thalach** (-0.431) - Maximum heart rate achieved
   - *Negative correlation*: Higher max heart rate → Lower disease risk

6. **cp_3.0** (-0.401) - Chest pain type 3
   - *Negative correlation*: This chest pain type → Lower disease risk

7. **cp_4.0** (0.373) - Chest pain type 4 (asymptomatic)
   - *Positive correlation*: Asymptomatic chest pain → Higher disease risk

8. **exang** (0.369) - Exercise-induced angina
   - *Positive correlation*: Exercise angina → Higher disease risk

9. **thal_7.0** (0.351) - Thalassemia type 7 (reversible defect)
   - *Positive correlation*: Reversible defect → Higher disease risk

10. **fbs** (-0.324) - Fasting blood sugar > 120 mg/dl
    - *Negative correlation*: Surprisingly, high fasting blood sugar shows negative correlation

## Key Insights

### Model Strengths
1. **High ROC-AUC (93.29%)**: Excellent ability to distinguish between disease and no disease
2. **Balanced Performance**: Similar accuracy on training (86.5%) and test (88.33%) sets - no overfitting
3. **Good Recall (87.5%)**: Successfully identifies most disease cases (only 3 false negatives)
4. **High Precision for No Disease (91%)**: Very reliable when predicting no disease

### Clinical Relevance
- **Most Important Predictors**:
  - Number of major vessels with blockage (ca)
  - Gender (males at higher risk)
  - Blood pressure levels
  - ST depression during exercise
  - Maximum heart rate capacity

- **Protective Factors**:
  - Higher maximum heart rate achieved
  - Certain chest pain types (type 3)

### Model Limitations
- **False Positives**: 4 patients predicted to have disease but don't (11% of no-disease cases)
- **False Negatives**: 3 patients with disease not detected (12.5% of disease cases)
- In a clinical setting, false negatives are more concerning as they represent missed diagnoses

## Recommendations

1. **Clinical Use**: This model shows strong predictive performance and could assist in preliminary screening
2. **Feature Focus**: Pay special attention to vessel blockage count, gender, and exercise-related metrics
3. **Further Improvement**: Consider ensemble methods or more complex models to reduce false negatives
4. **Validation**: Test on external datasets to ensure generalizability

## Files Generated
- `logistic_regression_results.png` - Confusion matrix and ROC curve visualizations
- `feature_importance.png` - Feature coefficient visualization
- `logistic_regression_model.py` - Complete model code
