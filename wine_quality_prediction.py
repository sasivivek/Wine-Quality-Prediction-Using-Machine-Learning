"""
Wine Quality Prediction Using Machine Learning
================================================
This project predicts the quality of wine based on various physicochemical 
features using multiple Machine Learning models (Logistic Regression, 
XGBClassifier, and SVC).

Dataset: Wine Quality Dataset (winequalityN.csv)
Reference: https://www.geeksforgeeks.org/machine-learning/wine-quality-prediction-machine-learning/
"""

# ============================================================
# 1. Import Libraries
# ============================================================
import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn import metrics
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 2. Load the Dataset
# ============================================================
df = pd.read_csv('winequalityN.csv')
print("=" * 60)
print("FIRST 5 ROWS OF THE DATASET")
print("=" * 60)
print(df.head())

# ============================================================
# 3. Explore the Dataset
# ============================================================
print("\n" + "=" * 60)
print("DATASET INFO")
print("=" * 60)
df.info()

print("\n" + "=" * 60)
print("DESCRIPTIVE STATISTICS")
print("=" * 60)
print(df.describe().T)

# ============================================================
# 4. Exploratory Data Analysis (EDA)
# ============================================================

# --- Check for null values ---
print("\n" + "=" * 60)
print("NULL VALUES IN EACH COLUMN")
print("=" * 60)
print(df.isnull().sum())

# --- Impute missing values with column mean ---
for col in df.columns:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].mean())

print("\nTotal null values after imputation:", df.isnull().sum().sum())

# --- Histogram of all features ---
df.hist(bins=20, figsize=(10, 10))
plt.suptitle("Distribution of Features", fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig("histograms.png", dpi=150, bbox_inches='tight')
plt.show()

# --- Bar plot: Quality vs Alcohol ---
plt.figure(figsize=(8, 5))
plt.bar(df['quality'], df['alcohol'])
plt.xlabel('Quality')
plt.ylabel('Alcohol')
plt.title('Quality vs Alcohol')
plt.savefig("quality_vs_alcohol.png", dpi=150, bbox_inches='tight')
plt.show()

# --- Replace wine type with numeric before correlation ---
df.replace({'white': 1, 'red': 0}, inplace=True)

# --- Correlation Heatmap ---
plt.figure(figsize=(12, 12))
sb.heatmap(df.corr(numeric_only=True) > 0.7, annot=True, cbar=False)
plt.title("Correlation Heatmap (Threshold > 0.7)")
plt.savefig("correlation_heatmap.png", dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# 5. Feature Engineering
# ============================================================

# Drop highly correlated feature
df = df.drop('total sulfur dioxide', axis=1)

# Create binary target: 1 if quality > 5, else 0
df['best quality'] = [1 if x > 5 else 0 for x in df.quality]

# Wine type already converted to numeric above (white=1, red=0)

# ============================================================
# 6. Train-Test Split
# ============================================================
features = df.drop(['quality', 'best quality'], axis=1)
target = df['best quality']

xtrain, xtest, ytrain, ytest = train_test_split(
    features, target, test_size=0.2, random_state=40
)

# Impute missing values after splitting
imputer = SimpleImputer(strategy='mean')
xtrain = imputer.fit_transform(xtrain)
xtest = imputer.transform(xtest)

print("\n" + "=" * 60)
print("TRAIN-TEST SPLIT SHAPES")
print("=" * 60)
print(f"Training set: {xtrain.shape}")
print(f"Test set:     {xtest.shape}")

# ============================================================
# 7. Normalize Features
# ============================================================
norm = MinMaxScaler()
xtrain = norm.fit_transform(xtrain)
xtest = norm.transform(xtest)

# ============================================================
# 8. Model Training & Evaluation
# ============================================================
print("\n" + "=" * 60)
print("MODEL TRAINING & EVALUATION")
print("=" * 60)

models = [LogisticRegression(), XGBClassifier(), SVC(kernel='rbf')]

for i in range(3):
    models[i].fit(xtrain, ytrain)
    print(f'\n{models[i]} :')
    print('Training Accuracy : ',
          metrics.roc_auc_score(ytrain, models[i].predict(xtrain)))
    print('Validation Accuracy : ',
          metrics.roc_auc_score(ytest, models[i].predict(xtest)))

# ============================================================
# 9. Confusion Matrix (Best Model - XGBClassifier)
# ============================================================
print("\n" + "=" * 60)
print("CONFUSION MATRIX (XGBClassifier)")
print("=" * 60)

cm = confusion_matrix(ytest, models[1].predict(xtest))
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=models[1].classes_
)
disp.plot()
plt.title("Confusion Matrix - XGBClassifier")
plt.savefig("confusion_matrix.png", dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# 10. Classification Report
# ============================================================
print("\n" + "=" * 60)
print("CLASSIFICATION REPORT (XGBClassifier)")
print("=" * 60)
print(metrics.classification_report(ytest, models[1].predict(xtest)))

print("\n[DONE] Wine Quality Prediction complete!")
print("[PLOTS SAVED] histograms.png, quality_vs_alcohol.png,")
print("              correlation_heatmap.png, confusion_matrix.png")
