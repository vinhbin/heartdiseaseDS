# ❤️ Heart Disease Prediction Project ❤️

## Team Members

**Vinh Le, Niyako Abajabel, Sharmake Botan, Giada Giacobbe, Abi Sheldon** 

## Overview

This project analyzes the Cleveland Heart Disease dataset from the UCI Machine Learning Repository.

We clean the data, perform exploratory data analysis (EDA), and build a simple machine learning model to predict heart disease presence in which 0 indicates no disease and 1, disease present.

## Folder Structure

```text
heartdisease/
│
├── data/
│   └── processed.cleveland.data  # Raw dataset from UCI
│
├── project.ipynb                 # Main Jupyter notebook
│
└── README.md                     
```


## Data Cleaning Summary

- Replace missing values (? → NaN → drop or impute)
- Convert all columns to numeric
- Simplify target (0 or 1)
- Remove duplicates
- Encode categorical variables (cp, thal, slope)
- Confirm no missing values and correct datatypes

After cleaning: 297 rows × 14 columns, ready for analysis.

## Exploratory Data Analysis

- Feature distributions (age, cholesterol, etc.)
- Correlation heatmap
- Feature comparison between target = 0 and target = 1 
- Outlier visualization (boxplots, scatterplots)

## Modeling

Build a Logistic Regression model to predict heart disease presence. Optional models: Decision Tree, Random Forest.

## Project Summary

- Cleaned and structured dataset
- Binary classification target
- EDA and modeling

**Goal**: Predict heart disease presence using medical attributes.

## Credits

**Dataset**: UCI Machine Learning Repository - Heart Disease (Cleveland)

**Language**: Python 3

**Libraries**: Pandas, NumPy, Seaborn, Matplotlib, Scikit-learn, JupyterLab