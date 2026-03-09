import streamlit as st
import pandas as pd
import joblib
import re
from scipy.sparse import hstack

st.set_page_config(page_title="RTECS Relevance Predictor", layout="centered")

# -----------------------------
# Configuration (must match predict_excel.py)
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
# -----------------------------

st.title("📊 RTECS Relevance Prediction – Excel Upload")

st.markdown("""
Upload an Excel file with the columns:  
**PMID, Title, Abstract**

The system will predict whether each record is **Relevant (1)** or **Not Relevant (0)**  
using the **Calibrated Linear SVM (Threshold = 0.25)**.
""")

uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx"])

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)

        st.subheader("📄 Uploaded Data Preview")
        st.dataframe(df.head())

        required_cols = {"PMID", "Title", "Abstract"}
        if not required_cols.issubset(df.columns):
            st.error(f"Excel must contain columns: {required_cols}")
        else:
            if st.button("🔍 Run Prediction"):
                with st.spinner("Processing and predicting..."):

                    # -----------------------------
                    # Preprocess
                    # -----------------------------
                    df["_title_clean"] = df["Title"].fillna("").apply(clean_text)
                    df["_abstract_clean"] = df["Abstract"].fillna("").apply(clean_text)

                    # -----------------------------
                    # Vectorize
                    # -----------------------------
                    X_title = title_vectorizer.transform(df["_title_clean"])
                    X_abs = abstract_vectorizer.transform(df["_abstract_clean"])
                    X = hstack([X_title, X_abs])

                    # -----------------------------
                    # Predict
                    # -----------------------------
                    probs = model.predict_proba(X)[:, 1]
                    labels = (probs >= THRESHOLD).astype(int)

                    # -----------------------------
                    # Attach results
                    # -----------------------------
                    df["Probability"] = probs
                    df["Label"] = labels

                    df.drop(columns=["_title_clean", "_abstract_clean"], inplace=True)

                st.success("✅ Prediction completed")

                st.subheader("📊 Prediction Results (Preview)")
                st.dataframe(df.head())

                # -----------------------------
                # Download output
                # -----------------------------
                output_excel = "prediction_output.xlsx"
                df.to_excel(output_excel, index=False)

                with open(output_excel, "rb") as f:
                    st.download_button(
                        label="⬇ Download Output Excel",
                        data=f,
                        file_name="RTECS_Predictions.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

                st.info(f"Threshold used: {THRESHOLD}")
                st.caption("Model: Calibrated LinearSVC")

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
