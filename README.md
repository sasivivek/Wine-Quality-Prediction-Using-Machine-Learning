# 🍷 Wine Quality Prediction Using Machine Learning

Predict the quality of wine based on physicochemical features using multiple Machine Learning classification models.

## 📋 Overview

This project uses the **Wine Quality Dataset** to predict whether a wine is of "best quality" (quality > 5) or not. Three ML models are trained and compared:

| Model | Description |
|-------|-------------|
| **Logistic Regression** | Linear classification model |
| **XGBClassifier** | Extreme Gradient Boosting |
| **SVC (RBF kernel)** | Support Vector Classifier |

## 📁 Project Structure

```
Wine Quality Prediction Using Machine Learning/
├── wine_quality_prediction.py   # Main script
├── winequalityN.csv             # Dataset
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 🛠️ Setup & Installation

1. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate        # Linux/Mac
   venv\Scripts\activate           # Windows
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the project**:
   ```bash
   python wine_quality_prediction.py
   ```

## 📊 What the Script Does

1. **Loads & Explores** the wine quality dataset (6,497 samples)
2. **Handles Missing Values** by imputing with column means
3. **Visualizes Data** with histograms, bar charts, and correlation heatmaps
4. **Engineers Features**: drops highly correlated columns, creates binary target
5. **Trains 3 Models**: Logistic Regression, XGBoost, SVC
6. **Evaluates** using ROC-AUC score, confusion matrix, and classification report
7. **Saves Plots** as PNG files

## 📈 Expected Output

```
LogisticRegression() :
Training Accuracy :  ~0.697
Validation Accuracy :  ~0.685

XGBClassifier() :
Training Accuracy :  ~0.976
Validation Accuracy :  ~0.804

SVC() :
Training Accuracy :  ~0.720
Validation Accuracy :  ~0.707
```

## 🔗 References

- [GeeksforGeeks - Wine Quality Prediction](https://www.geeksforgeeks.org/machine-learning/wine-quality-prediction-machine-learning/)
- [Wine Quality Dataset](https://media.geeksforgeeks.org/wp-content/uploads/20240910131455/winequalityN.csv)
