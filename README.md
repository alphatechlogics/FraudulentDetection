# Credit Card Fraud Detection Model 💳🚨

This project aims to build a machine learning model to detect fraudulent credit card transactions using an ensemble approach. The model was trained on a dataset containing anonymized credit card transactions, and it utilizes multiple classification models to improve the performance.

## Table of Contents 📑

- [Project Overview](#project-overview-)
- [Dataset Details](#dataset-details-)
- [Data Preprocessing](#data-preprocessing-)
- [Model Building](#model-building-)
- [Model Performance](#model-performance-)
- [Usage](#usage-)
- [Installation](#installation-)
- [Contributors](#contributors-)
- [License](#license-)

## Project Overview 🎯

Credit card fraud detection is a crucial task for financial institutions to protect customers and reduce losses due to fraudulent activities. This project involves training a machine learning model to predict whether a given transaction is legitimate or fraudulent.

### Goal of the Project:

- **Objective**: To predict whether a given credit card transaction is fraudulent or legitimate.
- **Approach**: We utilized a combination of multiple classification models, forming an ensemble model to improve accuracy and performance.

## Dataset Details 📊

The dataset used for this project is the **Credit Card Fraud Detection** dataset from Kaggle. It contains anonymized credit card transactions made by European cardholders in September 2013. The dataset has a significant class imbalance, with fraudulent transactions accounting for only 0.17% of all transactions.

### Dataset Link:

[Credit Card Fraud Detection Dataset on Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud?resource=download)

### Dataset Features:

The dataset contains 31 features:

- **V1 to V28**: These are the PCA-transformed features (Principal Components), which are used as the input variables. These transformed features are a result of dimensionality reduction and do not have direct, human-readable names.
- **Time**: The number of seconds elapsed between each transaction and the first transaction in the dataset.
- **Amount**: The monetary value of the transaction.
- **Class**: The target variable where `1` indicates fraudulent transactions and `0` indicates legitimate transactions.

### Dataset Structure:

- **Rows**: 284,807
- **Columns**: 31 (including `Time`, `Amount`, and `Class`)

## Data Preprocessing 🔄

To ensure the dataset is ready for machine learning models, we performed several preprocessing steps:

1. **Handling Imbalanced Data**: The dataset had a significant class imbalance (fraudulent transactions were only 0.17% of all transactions). To handle this, we applied **Random Under Sampling** to balance the class distribution.
2. **Correlation Filtering**: We filtered out features with low correlation to the target variable (`Class`) to reduce noise in the data. Features with a correlation coefficient less than 0.13 with `Class` were dropped.
3. **Splitting the Data**: The dataset was split into a training set (80%) and a test set (20%) for model evaluation.

## Model Building 🛠️

### Ensemble Approach:

To improve the model's robustness and accuracy, we used an **ensemble method** that combines predictions from multiple classifiers:

- **Logistic Regression**
- **Decision Tree Classifier**
- **Random Forest Classifier**
- **Gradient Boosting Classifier**
- **AdaBoost Classifier**
- **XGBoost Classifier**
- **Label Spreading**
- **Label Propagation**

We combined the predictions from these classifiers into a single **Voting Classifier**, where the final prediction is determined by the majority vote across all models. This ensemble method enhances the performance by leveraging the strengths of different models.

### Hyperparameter Tuning:

We performed **Grid Search Cross-Validation** to find the best hyperparameters for the models, which helped improve the model's generalization performance.

### Model Evaluation:

- **Accuracy**: We evaluated the model based on accuracy, precision, recall, and F1-score.
- **Confusion Matrix**: We used the confusion matrix to check how well the model distinguishes between legitimate and fraudulent transactions.

## Model Performance 📈

The final ensemble model achieved the following performance on the test set:

- **Accuracy**: 93.40%
- **F1 Score**: 92.88%
- **Precision**: 94.70%
- **Recall**: 91.84%
- **ROC AUC**: 93.39%

These results indicate that the model is well-suited for detecting fraudulent transactions, and it balances precision and recall well given the class imbalance.

## Usage 🖥️

To use the fraud detection model, you can input the transaction details into the Streamlit-based user interface. The app takes in features like the principal components (V1, V2, ..., V28), `Time`, and `Amount`, and predicts whether the transaction is legitimate or fraudulent.

1. **Run the Streamlit App**:
   streamlit run app.py

2. **Input Features**: The user will be prompted to enter the values for the following features:

- Principal Components (V1 to V7)
- Time (Seconds since the first transaction)
- Amount (Transaction amount)

3. **Prediction**: After entering the features, the model will predict whether the transaction is fraudulent or legitimate.

## Installation 🛠️

To run the app locally, follow these steps:

1. **Clone the repository**:

   ```bash
   git clone https://github.com/alphatechlogics/FraudulentDetection.git
   cd credit-card-fraud-detection
   ```

2. **Install the required dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app**:

   ```bash
   streamlit run app.py
   ```

### SUMMARY

Credit Card Fraud Detection Model Summary 🚨💳
This project aims to detect fraudulent credit card transactions using a machine learning model built with an ensemble approach. The model is trained using a dataset of anonymized credit card transactions. It combines predictions from multiple classification models to accurately identify whether a transaction is legitimate or fraudulent.

Key Features:
Principal Components (PCA): The model uses 7 PCA-transformed features (V1, V2, V3, V4, V5, V6, V7) as inputs, which are derived from dimensionality reduction techniques.
Transaction Details: The model also uses 2 original features, Time (time elapsed since the first transaction) and Amount (the transaction value).
Use Case:
Input: The user inputs transaction details, including the 7 PCA features, Time, and Amount.
Output: The model predicts whether the transaction is fraudulent (1) or legitimate (0).
By leveraging an ensemble of models, the project aims to provide high accuracy, precision, and recall, ensuring reliable detection of fraudulent transactions.

For ease of use, a Streamlit app allows users to input the necessary features and obtain real-time predictions for credit card transactions.
