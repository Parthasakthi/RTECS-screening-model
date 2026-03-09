import streamlit as st
import joblib
import re
import numpy as np
from scipy.sparse import hstack  
st.set_page_config(page_title="RTECS Relevance Predictor", layout="centered")

# -----------------------------
# Configuration (must match training)
# -----------------------------
THRESHOLD = 0.25

# -----------------------------
# Load model & vectorizers
# -----------------------------
@st.cache_resource
def load_models():
    title_vectorizer = joblib.load("tfidf_title.joblib")
    abstract_vectorizer = joblib.load("tfidf_abstract.joblib")
    model = joblib.load("calibrated_svm_model.joblib")
    return title_vectorizer, abstract_vectorizer, model

title_vectorizer, abstract_vectorizer, model = load_models()

# -----------------------------
# Text cleaning (must match training)
# -----------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# -----------------------------
# Streamlit UI
# ----------------------------

st.title("🔬 RTECS Relevance Prediction")
st.markdown("Enter a **Title** and **Abstract** to predict if the article is **Relevant**.")

pmid = st.text_input("PMID / Sample ID (optional)")
title = st.text_input("Title")
abstract = st.text_area("Abstract", height=200)

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔍 Predict"):
    if not title and not abstract:
        st.warning("Please enter at least a Title or Abstract.")
    else:
        with st.spinner("Analyzing..."):

            # Clean input
            title_clean = clean_text(title)
            abstract_clean = clean_text(abstract)

            # Vectorize separately
            X_title = title_vectorizer.transform([title_clean])
            X_abs = abstract_vectorizer.transform([abstract_clean])

            # Combine features
            X = hstack([X_title, X_abs])

            # Predict probability
            prob = model.predict_proba(X)[0][1]
            label = 1 if prob >= THRESHOLD else 0

        # -----------------------------
        # Display results
        # -----------------------------
        st.subheader("🧠 Prediction Result")

        if pmid:
            st.write("**PMID / ID:**", pmid)

        if label == 1:
            st.success("✅ **Relevant**")
        else:
            st.error("❌ **Not Relevant**")

        st.write("**Relevance Probability:**", f"{prob:.3f}")
        st.write("**Threshold used:**", THRESHOLD)

        # Decision explanation
        
        with st.expander("How this decision was made"):
            st.write("""
- Title and Abstract are cleaned  
- TF-IDF vectors are generated separately  
- Features are combined  
- Calibrated Linear SVM predicts probability  
- If probability ≥ 0.25 → Relevant  
            """)
