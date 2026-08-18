# Machine Learning Assignment 2 — Classification Models

## 1. Problem Statement

Build and evaluate multiple classification models on the same public classification dataset and expose the trained models through an interactive Streamlit application.

The application supports test-data upload, model selection, evaluation metrics, confusion matrix and classification report.

## 2. Dataset Description

**Dataset:** Breast Cancer Wisconsin (Diagnostic)

**Source:** UCI Machine Learning Repository. The implementation uses the copy bundled with `scikit-learn` through `sklearn.datasets.load_breast_cancer`.

- Instances: **569**
- Features: **30 numeric features**
- Target: **2 classes**
- Target encoding: `0 = malignant`, `1 = benign`
- Train/test split: **80% / 20%**
- Split method: stratified
- `random_state = 42`

The dataset satisfies the assignment requirement of at least 12 features and 500 instances.

## 3. GitHub Repository Link

**Replace this placeholder with your actual repository URL:**

`https://github.com/<YOUR-USERNAME>/<YOUR-REPOSITORY>`

## 4. Models Used and Evaluation

The assignment PDF explicitly lists five models: Logistic Regression, Decision Tree, kNN, Naive Bayes and Random Forest. Although one line refers to "6 ML models", only these five models are specified and the supplied comparison-table template also contains five rows. This solution therefore implements the five explicitly named models.

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree | 0.9211 | 0.9163 | 0.9565 | 0.9167 | 0.9362 | 0.8341 |
| kNN | 0.9737 | 0.9884 | 0.9600 | 1.0000 | 0.9796 | 0.9442 |
| Naive Bayes | 0.9386 | 0.9878 | 0.9452 | 0.9583 | 0.9517 | 0.8676 |
| Random Forest (Ensemble) | 0.9561 | 0.9944 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Achieved the strongest overall result in this fixed 80/20 split, with very high accuracy, AUC, F1 and MCC. Standardization is applied before logistic regression because the features have different numeric scales. |
| Decision Tree | Performance is lower than the other models on this split. The maximum depth of 5 limits tree complexity and helps reduce overfitting, but it also restricts the model's ability to capture some patterns. |
| kNN | Performed very strongly and obtained perfect recall on this particular test split. Feature standardization is important because kNN is distance-based. |
| Naive Bayes | Produced good results and a high AUC, but its conditional-independence assumption is restrictive for correlated medical features. |
| Random Forest (Ensemble) | Strong and stable performance with high AUC, precision, recall and F1. It improves over a single decision tree through ensemble averaging, although it did not exceed logistic regression on this particular split. |
| **Overall Winner** | **Logistic Regression**, based on the highest accuracy, F1 and MCC in this reproducible test split. |

> Important: the numerical results are specific to `random_state=42`, the chosen hyperparameters and the supplied test split. If you change these, rerun `train.py` and update the README table.

## 5. Streamlit Application

The application provides:

1. Test CSV upload.
2. Model-selection dropdown.
3. Accuracy, AUC, Precision, Recall, F1 and MCC.
4. Confusion matrix.
5. Classification report.
6. Prediction preview and CSV download.

### Run locally

```bash
pip install -r requirements.txt
python train.py
streamlit run app.py
```

## 6. Repository Structure

```text
project-folder/
├── app.py
├── train.py
├── requirements.txt
├── README.md
├── metrics.csv
├── test_data.csv
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    └── random_forest.joblib
```

## 7. Streamlit Community Cloud Deployment

1. Push this complete folder to GitHub.
2. Open Streamlit Community Cloud.
3. Sign in with GitHub.
4. Select **New app**.
5. Select your repository and `main` branch.
6. Select `app.py`.
7. Deploy.
8. Open the generated public URL and verify that the frontend loads.

**Live Streamlit App Link:** Replace this placeholder after deployment:

`https://<YOUR-STREAMLIT-APP-URL>`

## 8. BITS Virtual Lab Screenshot

Run the project in BITS Virtual Lab and capture one screenshot showing the assignment execution, as required by the assignment instructions.

## 9. Submission PDF

The final PDF should contain, in order:

1. GitHub repository link.
2. Live Streamlit app link.
3. BITS Virtual Lab screenshot.
4. This README content.

Do not submit placeholder links.

## 10. Academic Integrity

The assignment explicitly states that AI tools are allowed for learning support but not for direct copy-paste submissions. Customize this reference implementation, understand every component, and use your own repository history, UI design, explanations and observations before submission.
