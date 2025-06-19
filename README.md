# Fraud Detection in Financial Transactions

## Table of Contents
- [Fraud Detection in Financial Transactions](#fraud-detection-in-financial-transactions)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Project Structure and Components](#project-structure-and-components)
  - [Data](#data)
  - [Methodology](#methodology)
    - [Data Understanding \& Loading](#data-understanding--loading)
    - [Data Preprocessing](#data-preprocessing)
    - [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
    - [Model Development](#model-development)
    - [Model Evaluation](#model-evaluation)
  - [Web Application Interface](#web-application-interface)
  - [Results](#results)
  - [How to Run](#how-to-run)
  - [Conclusion \& Future Work](#conclusion--future-work)

## Overview

This project develops a machine learning model to detect fraudulent financial transactions. The primary goal is to accurately classify transactions as either legitimate or fraudulent, helping to mitigate financial losses.

This solution employs a **Support Vector Machine (SVM)** algorithm, a powerful classifier well-suited for this task. The project covers the complete data science lifecycle, from initial data analysis and preprocessing to model training, evaluation, and finally, deployment as an interactive web application built with **Streamlit**. This allows for real-time fraud prediction based on user input.

## Project Structure and Components

The project consists of a Jupyter Notebook for model development (`svm_model_training.ipynb`) and a Python script for the web application (`app.py`).

*   **Model Training Notebook (`svm_model_training.ipynb`):**
    *   **`Data` Class:** Handles loading the `data.csv` dataset and initial data exploration.
    *   **`DataPreprocessing` Class:** Selects relevant features and performs label encoding on categorical data (e.g., converting 'Online', 'ATM' to numbers).
    *   **EDA Classes (`Graph`, `UnivariateAnalysis`, `BivariateAnalysis`):** Used for creating visualizations to understand data distributions and relationships between features and fraud.
    *   **`Model` Class:** Manages the core machine learning pipeline, including splitting data, applying `StandardScaler` for feature scaling, training a **Support Vector Classifier (SVC)**, and evaluating its performance.
    *   **`PickleData` Class:** Saves the trained model (`svm_model.pkl`) and scaler (`scaler.pkl`) for use in the application.

*   **Streamlit Web Application (`app.py`):**
    *   Provides a user-friendly interface for real-time predictions.
    *   Loads the saved `svm_model.pkl` and `scaler.pkl`.
    *   Takes transaction details from the user via a form.
    *   Processes the input using the saved scaler and predicts the fraud risk using the loaded model.
    *   Displays the result ("High Risk" or "Low Risk") in a clear and visually appealing manner.

## Data

The model is trained on a financial transaction dataset (`data.csv`) using the following key features:

*   **Numerical Features:** `TransactionAmount`, `CustomerAge`, `AccountBalance`, `LoginAttempts`.
*   **Categorical Feature:** `Channel` (encoded into `ChannelEncoded`).
*   **Target Variable:** `is_fraud` (a binary variable where `1` indicates a fraudulent transaction and `0` indicates a legitimate one).

## Methodology

### Data Understanding & Loading

The process begins by loading the `data.csv` dataset into a Pandas DataFrame. Initial exploration is performed to understand the data's structure, check for null values, and generate descriptive statistics.

### Data Preprocessing

The `DataPreprocessing` class executes the following key steps:
*   **Feature Selection:** A subset of relevant features is chosen for model training.
*   **Label Encoding:** The categorical `Channel` column is converted into numerical representations (`{'ATM': 0, 'Online': 1, 'Branch': 2}`), making it suitable for the SVM algorithm.

### Exploratory Data Analysis (EDA)

Visualizations are created to gain insights from the data:
*   **Univariate Analysis:** Distributions of individual features like `TransactionAmount` and `CustomerAge` are plotted to understand their spread.
*   **Bivariate Analysis:** The relationship between features and the target variable (`is_fraud`) is analyzed to identify patterns associated with fraudulent activities.

### Model Development

The core machine learning pipeline is executed by the `Model` class:
*   **Data Splitting:** The dataset is split into training (80%) and testing (20%) sets. Stratified sampling is used to maintain the class distribution in both sets.
*   **Feature Scaling:** `StandardScaler` is applied to the training and testing data. This standardizes features to have a mean of 0 and a standard deviation of 1, which is crucial for SVM performance.
*   **Model Training:** A **Support Vector Classifier (SVC)** with a linear kernel is trained on the scaled training data.

### Model Evaluation

The trained model's performance is assessed on the unseen test data using standard metrics:
*   **Confusion Matrix:** To see the counts of correct and incorrect predictions.
*   **Classification Report:** To review precision, recall, and F1-score for each class.
*   **Accuracy Score:** To get the overall percentage of correct classifications.

## Web Application Interface

The project is deployed as an interactive web app using **Streamlit**, featuring two main sections:

*   **Prediction Page:** An intuitive form where users can input transaction details. The app processes this input in real-time and displays the model's prediction with clear visual cues (red for "High Risk" 🚨 and green for "Low Risk" ✅).
*   **About Page:** Provides detailed information about the project's purpose, the technologies used, and the developer.

## Results

*Model Performance Summary:*

The SVM model's performance on the test set is as follows:
          precision    recall  f1-score   support

       0       0.83      0.94      0.88       376
       1       0.70      0.43      0.53       127

accuracy                           0.81       503

*   **Overall Accuracy:** Approximately **81%**. The model correctly classifies the majority of transactions.
*   **Fraud Class (1) Performance:**
    *   **Precision (0.70):** 70% of transactions flagged as fraud are actually fraudulent.
    *   **Recall (0.43):** The model identifies 43% of all actual fraudulent transactions.

## How to Run

Follow these steps to set up and run the project:

1.  **Install Prerequisites:**
    ```bash
    pip install pandas scikit-learn matplotlib streamlit streamlit-option-menu
    ```

2.  **Run the Streamlit Application:**
    *   Ensure `app.py`, `svm_model.pkl`, and `scaler.pkl` are in the same directory.
    *   Open your terminal and navigate to the project folder.
    *   Execute the following command:
    ```bash
    streamlit run app.py
    ```
    *   The application will open in your web browser.

## Conclusion & Future Work

This project successfully builds an end-to-end system for fraud detection, from model training to a functional web application. The SVM model provides a solid baseline for identifying fraudulent transactions.

**Future Work:**
*   **Improve Model Recall:** Implement techniques like SMOTE or use class weights to better handle the imbalanced nature of the dataset and improve the detection of fraudulent cases.
*   **Advanced Models:** Experiment with tree-based models like XGBoost or LightGBM, which often yield higher performance on tabular data.
*   **Feature Engineering:** Develop new features from existing data (e.g., transaction frequency, time-of-day analysis) to enhance predictive power.
*   **Cloud Deployment:** Deploy the Streamlit app to a cloud service to make it publicly accessible.