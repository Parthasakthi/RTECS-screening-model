# train.py
import pandas as pd
import re
import joblib
import numpy as np
from scipy.sparse import hstack

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix

# -----------------------------
# 1) Text cleaning function
# -----------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# -----------------------------
# 2) Load and preprocess data
# -----------------------------
df = pd.read_excel("Sample data.xlsx")

df['title_clean'] = df['Title'].fillna('').apply(clean_text)
df['abstract_clean'] = df['Abstract'].fillna('').apply(clean_text)

print("✅ Data loaded successfully.")
print("Rows:", len(df))
print("Label distribution:\n", df['Label'].value_counts())

# -----------------------------
# 3) Train-test split
# -----------------------------
X_title_train, X_title_test, \
X_abs_train, X_abs_test, \
y_train, y_test = train_test_split(
    df['title_clean'],
    df['abstract_clean'],
    df['Label'],
    test_size=0.2,
    random_state=42,
    stratify=df['Label']
)

# -----------------------------
# 4) TF-IDF Vectorization (separate)
# -----------------------------
title_vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=10000,
    ngram_range=(1, 2),
    sublinear_tf=True
)

abstract_vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=40000,
    ngram_range=(1, 3),
    sublinear_tf=True
)

X_title_train_vec = title_vectorizer.fit_transform(X_title_train)
X_title_test_vec = title_vectorizer.transform(X_title_test)

X_abs_train_vec = abstract_vectorizer.fit_transform(X_abs_train)
X_abs_test_vec = abstract_vectorizer.transform(X_abs_test)

# -----------------------------
# 5) Combine features
# -----------------------------
X_train_vec = hstack([X_title_train_vec, X_abs_train_vec])
X_test_vec = hstack([X_title_test_vec, X_abs_test_vec])

print("✅ TF-IDF vectorization complete.")
print("Total feature space size:", X_train_vec.shape[1])

# -----------------------------
# 6) Train Calibrated Linear SVM
# -----------------------------
base_svm = LinearSVC(
    class_weight={0: 1, 1: 3},   # penalize FN
    max_iter=5000
)

clf = CalibratedClassifierCV(
    estimator=base_svm,
    method='sigmoid',
    cv=5
)

clf.fit(X_train_vec, y_train)
print("✅ Calibrated SVM training complete.")

# -----------------------------
# 7) Threshold-based prediction
# -----------------------------
THRESHOLD = 0.25   # 🔽 lower = fewer FN

y_scores = clf.predict_proba(X_test_vec)[:, 1]
y_pred = (y_scores >= THRESHOLD).astype(int)

# -----------------------------
# 8) Evaluation
# -----------------------------
print(f"\n🔹 Classification Report (threshold = {THRESHOLD}):")
print(classification_report(y_test, y_pred, digits=3))

cm = confusion_matrix(y_test, y_pred)
print("\n🔹 Confusion Matrix:")
print(cm)

fn = cm[1, 0]
fp = cm[0, 1]
print(f"\n❗ False Negatives (1 → 0): {fn}")
print(f"⚠ False Positives (0 → 1): {fp}")

# -----------------------------
# 9) Save model & vectorizers
# -----------------------------
joblib.dump(title_vectorizer, "tfidf_title.joblib")
joblib.dump(abstract_vectorizer, "tfidf_abstract.joblib")
joblib.dump(clf, "calibrated_svm_model.joblib")

print("\n💾 Saved calibrated SVM model and vectorizers.")
