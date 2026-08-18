import io
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE / "model"

MODEL_FILES = {
    "Logistic Regression": MODEL_DIR / "logistic_regression.joblib",
    "Decision Tree": MODEL_DIR / "decision_tree.joblib",
    "kNN": MODEL_DIR / "knn.joblib",
    "Naive Bayes": MODEL_DIR / "naive_bayes.joblib",
    "Random Forest": MODEL_DIR / "random_forest.joblib",
}

FEATURES = [
    "mean radius", "mean texture", "mean perimeter", "mean area",
    "mean smoothness", "mean compactness", "mean concavity",
    "mean concave points", "mean symmetry", "mean fractal dimension",
    "radius error", "texture error", "perimeter error", "area error",
    "smoothness error", "compactness error", "concavity error",
    "concave points error", "symmetry error", "fractal dimension error",
    "worst radius", "worst texture", "worst perimeter", "worst area",
    "worst smoothness", "worst compactness", "worst concavity",
    "worst concave points", "worst symmetry", "worst fractal dimension"
]

st.set_page_config(
    page_title="ML Assignment 2 - Classification",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Breast Cancer Classification Dashboard")
st.caption("BITS M.Tech AIML/DSE — Machine Learning Assignment 2")

st.sidebar.header("Controls")
selected_model = st.sidebar.selectbox(
    "Select classification model",
    list(MODEL_FILES.keys())
)

uploaded = st.sidebar.file_uploader(
    "Upload test CSV",
    type=["csv"],
    help="Upload test data containing the 30 feature columns and a 'target' column."
)

@st.cache_resource
def load_model(path):
    return joblib.load(path)

@st.cache_data
def load_default_test():
    return pd.read_csv(BASE / "test_data.csv")

try:
    model = load_model(MODEL_FILES[selected_model])
except Exception as exc:
    st.error(f"Could not load model: {exc}")
    st.stop()

if uploaded is not None:
    try:
        df = pd.read_csv(io.BytesIO(uploaded.getvalue()))
        source_name = "Uploaded CSV"
    except Exception as exc:
        st.error(f"Unable to read CSV: {exc}")
        st.stop()
else:
    df = load_default_test()
    source_name = "Bundled test_data.csv"

missing = [c for c in FEATURES if c not in df.columns]
if missing:
    st.error(
        "The CSV is missing required feature columns. "
        f"Missing: {', '.join(missing)}"
    )
    st.stop()

if "target" not in df.columns:
    st.error("The CSV must contain a 'target' column so evaluation metrics can be calculated.")
    st.stop()

X = df[FEATURES]
y = df["target"].astype(int)

pred = model.predict(X)
proba = model.predict_proba(X)[:, 1]

metrics = {
    "Accuracy": accuracy_score(y, pred),
    "AUC": roc_auc_score(y, proba),
    "Precision": precision_score(y, pred, zero_division=0),
    "Recall": recall_score(y, pred, zero_division=0),
    "F1 Score": f1_score(y, pred, zero_division=0),
    "MCC": matthews_corrcoef(y, pred),
}

st.info(f"Data source: {source_name} | Rows evaluated: {len(df)}")

cols = st.columns(6)
for col, (label, value) in zip(cols, metrics.items()):
    col.metric(label, f"{value:.4f}")

st.subheader("Confusion Matrix")
cm = confusion_matrix(y, pred)
fig, ax = plt.subplots()
im = ax.imshow(cm)
ax.set_xlabel("Predicted label")
ax.set_ylabel("True label")
ax.set_title(f"{selected_model} — Confusion Matrix")
for (i, j), value in np.ndenumerate(cm):
    ax.text(j, i, str(value), ha="center", va="center")
fig.colorbar(im, ax=ax)
st.pyplot(fig)
plt.close(fig)

st.subheader("Classification Report")
report = classification_report(
    y, pred,
    target_names=["Malignant (0)", "Benign (1)"],
    output_dict=True,
    zero_division=0
)
st.dataframe(pd.DataFrame(report).transpose().round(4), use_container_width=True)

st.subheader("Prediction Preview")
preview = X.copy()
preview["Actual Target"] = y.values
preview["Predicted Target"] = pred
preview["Predicted Probability (Class 1)"] = proba
st.dataframe(preview.head(20), use_container_width=True)

st.download_button(
    "Download prediction results",
    preview.to_csv(index=False).encode("utf-8"),
    file_name="predictions.csv",
    mime="text/csv"
)

st.divider()
st.caption(
    "Academic note: this is a reference implementation. Customize the UI, "
    "wording, hyperparameters and observations before submission."
)
