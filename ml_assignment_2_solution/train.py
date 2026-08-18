"""
Train all required classification models for ML Assignment 2.

Dataset:
UCI Breast Cancer Wisconsin (Diagnostic) dataset, accessed through
scikit-learn's bundled copy of the UCI dataset.

Split:
80% train / 20% test, stratified, random_state=42.

The script:
1. Loads the dataset.
2. Trains five models listed in the assignment.
3. Saves models into model/.
4. Saves held-out test data into test_data.csv.
5. Saves evaluation metrics into metrics.csv.
"""

from pathlib import Path
import joblib
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef
)

BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE / "model"
MODEL_DIR.mkdir(exist_ok=True)

ds = load_breast_cancer(as_frame=True)
X = ds.data.copy()
y = ds.target.copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=5000, random_state=42))
    ]),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "kNN": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=7))
    ]),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=-1
    ),
}

files = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "kNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Random Forest": "random_forest.joblib",
}

results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_DIR / files[name])

    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    results.append({
        "ML Model Name": name,
        "Accuracy": accuracy_score(y_test, pred),
        "AUC": roc_auc_score(y_test, proba),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1": f1_score(y_test, pred, zero_division=0),
        "MCC": matthews_corrcoef(y_test, pred),
    })

pd.DataFrame(results).to_csv(BASE / "metrics.csv", index=False)

test_df = X_test.copy()
test_df["target"] = y_test.values
test_df.to_csv(BASE / "test_data.csv", index=False)

print(pd.DataFrame(results).round(4).to_string(index=False))
print("\nSaved models, metrics.csv and test_data.csv.")
