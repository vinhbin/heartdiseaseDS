❤️ Heart Disease Prediction Project
📘 Overview

This project analyzes the Cleveland Heart Disease dataset from the UCI Machine Learning Repository
.
We clean the data, perform exploratory data analysis (EDA), and build a simple machine learning model to predict heart disease presence — where
0 = no disease and 1 = disease present.

🧠 Team Members

Vinh Le

Niyako Abajabel 

Sharmake Botan 

Giada Giacobbe 

Abi Sheldon 

🗂️ Folder Structure

heartdisease/
│
├── data/
│ └── processed.cleveland.data # Raw dataset from UCI
│
├── project.ipynb # Main Jupyter notebook
│
└── README.md # This file

⚙️ Setup Instructions
1️⃣ Install Python

Make sure Python 3.8 or above is installed.
Check version:
python --version
If not installed, download from python.org/downloads
.

2️⃣ Create a Virtual Environment

Inside your project folder:
python -m venv venv

Activate it:

Windows: venv\Scripts\activate

macOS/Linux: source venv/bin/activate

3️⃣ Install Dependencies

Install required libraries:
pip install pandas numpy matplotlib seaborn scikit-learn jupyterlab

4️⃣ Launch JupyterLab

Run:
jupyter lab
Then open project.ipynb from the Jupyter interface.
If it doesn’t open automatically:
python -m jupyter lab

5️⃣ Dataset Setup

Download the dataset from UCI:
Heart Disease (Cleveland)

Save as:
heartdisease/data/processed.cleveland.data

If teammates clone the repo, they just need to confirm the file exists in /data.

🧹 Data Cleaning Summary

Replace missing values (? → NaN → drop or impute)

Convert all columns to numeric

Simplify target (0 or 1)

Remove duplicates

Encode categorical variables (cp, thal, slope)

Confirm no missing values and correct datatypes

After cleaning: 297 rows × 14 columns, ready for analysis.

📊 Exploratory Data Analysis

Feature distributions (age, cholesterol, etc.)

Correlation heatmap

Feature comparison between target=0 and target=1

Outlier visualization (boxplots, scatterplots)

🧠 Modeling

We’ll build a Logistic Regression model to predict heart disease presence.
Optional models: Decision Tree, Random Forest.

🧩 Troubleshooting

If you see:
ModuleNotFoundError: No module named 'pandas'
→ Run:
pip install pandas numpy matplotlib seaborn scikit-learn

If Jupyter doesn’t open:
python -m jupyter lab

To exit your virtual environment:
deactivate

🏁 Quick Setup Recap
git clone <repo-link>
cd heartdisease
python -m venv venv
source venv/bin/activate    # macOS/Linux
venv\Scripts\activate       # Windows
pip install -r requirements.txt
jupyter lab

✅ Project Summary

Cleaned and structured dataset

Binary classification target

Ready for EDA and modeling

Goal: Predict heart disease presence using medical attributes.

✨ Credits

Dataset: UCI Machine Learning Repository — Heart Disease (Cleveland)
Language: Python 3
Libraries: Pandas, NumPy, Seaborn, Matplotlib, Scikit-learn, JupyterLab
Developed by: 5 of Hearts Team