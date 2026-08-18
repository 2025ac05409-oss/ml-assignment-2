# Machine Learning Assignment 2 — Classification Models

## 1. Problem Statement

The objective of this assignment is to implement and evaluate multiple machine learning classification models using the same classification dataset. The trained models are made available through an interactive Streamlit application.

The application allows the user to:

* Upload test data in CSV format.
* Select a classification model.
* Generate predictions.
* View classification performance metrics.
* View the confusion matrix.
* View the classification report.
* Preview predictions and download the prediction results.

---

## 2. Dataset Description

### Dataset: Breast Cancer Wisconsin (Diagnostic)

The Breast Cancer Wisconsin (Diagnostic) dataset available through `scikit-learn` is used for this assignment.

The dataset contains:

* **569 instances**
* **30 numerical features**
* **2 target classes**
* Target `0` = Malignant
* Target `1` = Benign

The data is divided into training and testing sets using an **80:20 stratified split** with:

```text
random_state = 42
```

Stratification is used so that the class distribution is maintained between the training and testing datasets.

The dataset satisfies the assignment requirement of having at least 500 instances and more than 12 features.

---

## 3. GitHub Repository

**GitHub Repository:** https://github.com/2025ac05409-oss/ml-assignment-2/tree/main/ml_assignment_2_solution

> Replace the above placeholder with the actual GitHub repository URL before submitting.

---

## 4. Machine Learning Models

Five classification algorithms were implemented and evaluated using the same test dataset:

1. Logistic Regression
2. Decision Tree
3. k-Nearest Neighbours (kNN)
4. Naive Bayes
5. Random Forest

The models were trained using the same training/test split so that their performance could be compared fairly.

---

## 5. Model Performance Comparison

The following table shows the evaluation results obtained on the test dataset.

| ML Model            | Accuracy |    AUC | Precision | Recall | F1 Score |    MCC |
| ------------------- | -------: | -----: | --------: | -----: | -------: | -----: |
| Logistic Regression |   0.9825 | 0.9954 |    0.9861 | 0.9861 |   0.9861 | 0.9623 |
| Decision Tree       |   0.9211 | 0.9163 |    0.9565 | 0.9167 |   0.9362 | 0.8341 |
| kNN                 |   0.9737 | 0.9884 |    0.9600 | 1.0000 |   0.9796 | 0.9442 |
| Naive Bayes         |   0.9386 | 0.9878 |    0.9452 | 0.9583 |   0.9517 | 0.8676 |
| Random Forest       |   0.9561 | 0.9944 |    0.9589 | 0.9722 |   0.9655 | 0.9054 |

### Metric Definitions

**Accuracy** measures the proportion of correctly classified observations.

**AUC (Area Under the ROC Curve)** measures the model's ability to distinguish between the two classes.

**Precision** measures how many observations predicted as positive actually belong to the positive class.

**Recall** measures how many actual positive observations were correctly identified.

**F1 Score** is the harmonic mean of precision and recall.

**MCC (Matthews Correlation Coefficient)** provides a balanced measure of classification quality, particularly useful for binary classification.

---

## 6. Model-wise Observations

### Logistic Regression

Logistic Regression produced the best overall performance on the selected test split. It achieved an accuracy of **98.25%**, an AUC of **0.9954**, and an MCC of **0.9623**.

Feature standardization was used because the numerical features have different scales. The model provided a strong balance between precision and recall.

### Decision Tree

The Decision Tree achieved an accuracy of **92.11%**. Its performance was lower than the other models on this test split.

Limiting the tree depth helps control model complexity and reduce the possibility of overfitting, although it can also restrict the model's ability to capture more complex relationships.

### k-Nearest Neighbours

The kNN model achieved an accuracy of **97.37%** and a recall of **1.0000** on the test dataset.

Since kNN is distance-based, feature standardization is important for obtaining meaningful distances between observations.

### Naive Bayes

Naive Bayes achieved an accuracy of **93.86%** and an AUC of **0.9878**.

The model performed reasonably well, although its assumption that the input features are conditionally independent can be restrictive for a dataset containing correlated medical measurements.

### Random Forest

Random Forest achieved an accuracy of **95.61%** and an AUC of **0.9944**.

As an ensemble method consisting of multiple decision trees, Random Forest provides more stable predictions than a single decision tree and performed strongly across the evaluation metrics.

---

## 7. Overall Comparison

Based on the results obtained from the selected 80:20 test split, **Logistic Regression produced the strongest overall performance**.

It achieved:

* Highest Accuracy: **0.9825**
* Highest AUC: **0.9954**
* Highest F1 Score: **0.9861**
* Highest MCC: **0.9623**

kNN also performed very well, particularly in terms of recall, while Random Forest provided strong overall performance.

The Decision Tree produced the lowest overall performance among the five models for this particular test split.

These results are specific to the selected dataset split, preprocessing steps and model parameters.

---

## 8. Streamlit Application

An interactive Streamlit application was developed to provide access to the trained classification models.

### Application Features

The application provides:

1. **CSV test-data upload**
2. **Classification model selection**
3. **Accuracy**
4. **AUC**
5. **Precision**
6. **Recall**
7. **F1 Score**
8. **MCC**
9. **Confusion matrix**
10. **Classification report**
11. **Prediction preview**
12. **Prediction CSV download**

The application initially uses the bundled `test_data.csv` when no external test file is uploaded.

The model can be changed using the model-selection dropdown available in the application.

---

## 9. Running the Project Locally

Clone or download the repository and open the project directory.

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Train the models:

```bash
python train.py
```

Start the Streamlit application:

```bash
streamlit run app.py
```

The application can then be opened using the local Streamlit URL displayed in the terminal.

---

## 10. Project Structure

```text
ml_assignment_2_solution/
│
├── app.py
├── train.py
├── requirements.txt
├── README.md
├── test_data.csv
├── metrics.csv
├── DATASET_SOURCE.txt
│
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    └── random_forest.joblib
```

### File Description

| File                 | Purpose                               |
| -------------------- | ------------------------------------- |
| `app.py`             | Streamlit application                 |
| `train.py`           | Model training and evaluation         |
| `requirements.txt`   | Required Python libraries             |
| `test_data.csv`      | Test dataset used by the application  |
| `metrics.csv`        | Stored model evaluation metrics       |
| `DATASET_SOURCE.txt` | Dataset source information            |
| `model/`             | Saved trained machine learning models |
| `README.md`          | Project documentation                 |

---

## 11. Streamlit Community Cloud Deployment

The application can be deployed using Streamlit Community Cloud.

Deployment steps:

1. Push the complete project to GitHub.
2. Open Streamlit Community Cloud.
3. Sign in using the GitHub account.
4. Create a new application.
5. Select the GitHub repository.
6. Select the `main` branch.
7. Set the application file to:

```text
ml_assignment_2_solution/app.py
```

8. Deploy the application.
9. Open the generated public URL.
10. Test the model-selection, CSV upload and evaluation features.

### Live Streamlit Application

https://jntch9vqtxkazrplh2gufn.streamlit.app/

> Replace this placeholder with the actual deployed application URL before submission.

---


## 12. Conclusion

Five classification algorithms were implemented and evaluated on the Breast Cancer Wisconsin (Diagnostic) dataset.

Among the evaluated models, Logistic Regression achieved the strongest overall performance on the selected test split, with an accuracy of **98.25%** and an AUC of **0.9954**.

The trained models were integrated into a Streamlit application that provides an interactive interface for model selection, test-data upload, prediction and evaluation.

The project therefore combines the complete machine learning workflow of data preparation, model training, evaluation, model persistence and interactive deployment.
