# RTECS-screening-model

This machine learning model predicts whether a research article is relevant to the RTECS project based on its title and abstract. It helps reduce manual screening effort by automatically filtering out highly irrelevant articles before the manual review stage. By acting as an intermediate step in the screening pipeline, the model improves the overall relevancy rate during manual screening. 

## Project Files Description

### 1. Sample data.xlsx
Dataset used to train the machine learning model.
Contains approximately 14,000 research articles with Title, Abstract, and relevancy labels collected from past RTECS screening data.

### 2. train.py

Python script used to train the machine learning model.

Key steps performed:

  - Text preprocessing and cleaning

  - TF-IDF vectorization of title and abstract separately

  - Feature combination

  - Training a Linear SVM classifier

  - Probability calibration

  - Saving the trained model and vectorizers as .joblib files. 

### 3. Model Files

These files store the trained model and feature vectorizers used for prediction.

  tfidf_title.joblib – TF-IDF vectorizer trained on article titles

  tfidf_abstract.joblib – TF-IDF vectorizer trained on article abstracts

  calibrated_svm_model.joblib – Probability-calibrated Linear SVM classification model used for relevancy prediction

### 4. Test.xlsx

Input file used for running predictions with the trained model.
Contains article data (Title and Abstract) that need to be screened.

### 5. predict.py

Python script used to predict relevancy labels for articles in Test.xlsx.

Workflow:

Load trained model and vectorizers

Preprocess text

Convert text into TF-IDF features

Predict relevancy label and probability score

Output results to a new file.

### 6. app.py and app_single.py

Streamlit applications used to demonstrate the trained model through a simple user interface.
