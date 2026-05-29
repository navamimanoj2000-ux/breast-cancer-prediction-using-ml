# =========================================================
# Predictive Modeling Using Machine Learning
# Full Working ML Project
# Dataset: Breast Cancer Prediction
# Algorithms:
#   1. Logistic Regression
#   2. Decision Tree
#   3. Random Forest


# IMPORT LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

# =========================================================
# LOAD DATASET
# =========================================================

data = load_breast_cancer()

# Convert dataset into DataFrame
df = pd.DataFrame(data.data, columns=data.feature_names)

# Add target column
df['target'] = data.target

print("Dataset Preview:\n")
print(df.head())

print("\nDataset Shape:", df.shape)

# =========================================================
# SPLIT FEATURES AND TARGET
# =========================================================

X = df.drop('target', axis=1)
y = df['target']

# =========================================================
# TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================================================
# FEATURE SCALING
# =========================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================================================
# INITIALIZE MODELS
# =========================================================

log_model = LogisticRegression()

tree_model = DecisionTreeClassifier(
    criterion='gini',
    max_depth=5,
    random_state=42
)

forest_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# =========================================================
# TRAIN MODELS
# =========================================================

log_model.fit(X_train, y_train)
tree_model.fit(X_train, y_train)
forest_model.fit(X_train, y_train)

# =========================================================
# PREDICTIONS
# =========================================================

log_pred = log_model.predict(X_test)
tree_pred = tree_model.predict(X_test)
forest_pred = forest_model.predict(X_test)

# =========================================================
# ACCURACY SCORES
# =========================================================

print("\n==============================")
print("MODEL ACCURACY")
print("==============================")

print("Logistic Regression Accuracy:",
      accuracy_score(y_test, log_pred))

print("Decision Tree Accuracy:",
      accuracy_score(y_test, tree_pred))

print("Random Forest Accuracy:",
      accuracy_score(y_test, forest_pred))

# =========================================================
# CLASSIFICATION REPORT
# =========================================================

print("\n==============================")
print("CLASSIFICATION REPORT")
print("==============================")

print("\nRandom Forest Report:\n")
print(classification_report(y_test, forest_pred))

# =========================================================
# CONFUSION MATRIX
# =========================================================

cm = confusion_matrix(y_test, forest_pred)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()

# =========================================================
# ROC CURVE
# =========================================================

# Predict probabilities
forest_probs = forest_model.predict_proba(X_test)[:,1]

# ROC values
fpr, tpr, thresholds = roc_curve(y_test, forest_probs)

roc_auc = auc(fpr, tpr)

# Plot ROC Curve
plt.figure(figsize=(7,6))

plt.plot(
    fpr,
    tpr,
    label='Random Forest (AUC = %0.2f)' % roc_auc
)

plt.plot([0,1], [0,1], linestyle='--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend(loc="lower right")

plt.show()

# =========================================================
# FEATURE IMPORTANCE


importance = forest_model.feature_importances_

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importance
})

feature_importance = feature_importance.sort_values(
    by='Importance',
    ascending=False
)

print("\nTop Important Features:\n")
print(feature_importance.head(10))

# Plot feature importance
plt.figure(figsize=(10,6))

sns.barplot(
    x='Importance',
    y='Feature',
    data=feature_importance.head(10)
)

plt.title("Top 10 Important Features")

plt.show()

# =========================================================
# SAMPLE PREDICTION


sample = X_test[0].reshape(1, -1)

prediction = forest_model.predict(sample)

if prediction[0] == 1:
    print("\nPrediction Result: Benign")
else:
    print("\nPrediction Result: Malignant")

