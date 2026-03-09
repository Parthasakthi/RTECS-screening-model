# predict_excel.py
import pandas as pd
import joblib
import re
from scipy.sparse import hstack

# -----------------------------
# Configuration (DO NOT CHANGE)
# -----------------------------
THRESHOLD = 0.25

INPUT_FILE = "TTest1.xlsx"
OUTPUT_FILE = "output1.xlsx"

# -----------------------------
# Text cleaning (MUST match training)
# -----------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# -----------------------------
# Load model & vectorizers
# -----------------------------
print("🔹 Loading model and vectorizers...")
title_vectorizer = joblib.load("tfidf_title.joblib")
abstract_vectorizer = joblib.load("tfidf_abstract.joblib")
model = joblib.load("calibrated_svm_model.joblib")

# -----------------------------
# Load input Excel
# -----------------------------
print("🔹 Reading input Excel...")
df = pd.read_excel(INPUT_FILE)

required_cols = {"PMID", "Title", "Abstract"}
if not required_cols.issubset(df.columns):
    raise ValueError(
        f"Input Excel must contain columns: {required_cols}"
    )

# -----------------------------
# Preprocess text
# -----------------------------
df["_title_clean"] = df["Title"].fillna("").apply(clean_text)
df["_abstract_clean"] = df["Abstract"].fillna("").apply(clean_text)

# -----------------------------
# Vectorize
# -----------------------------
print("🔹 Vectorizing text...")
X_title = title_vectorizer.transform(df["_title_clean"])
X_abs = abstract_vectorizer.transform(df["_abstract_clean"])

X = hstack([X_title, X_abs])

# -----------------------------
# Predict
# -----------------------------
print("🔹 Predicting labels...")
probs = model.predict_proba(X)[:, 1]
labels = (probs >= THRESHOLD).astype(int)

# -----------------------------
# Attach output
# -----------------------------
df["Label"] = labels

# Drop internal columns
df.drop(columns=["_title_clean", "_abstract_clean"], inplace=True)

# -----------------------------
# Save output Excel
# -----------------------------
print("🔹 Writing output Excel...")
df.to_excel(OUTPUT_FILE, index=False)

print("\n✅ Prediction completed successfully")
print(f"📄 Output file : {OUTPUT_FILE}")
print(f"🚨 Threshold  : {THRESHOLD}")
print("🧠 Model     : Calibrated LinearSVC")