# 🌍 ESG Risk Analyzer 🧠

A powerful AI-driven application that evaluates Environmental, Social, and Governance (ESG) risks in news data using classification, regression, sentiment analysis, explainability, and retrieval-based question answering.

---

## 📌 Project Overview

The **ESG Risk Analyzer** is an end-to-end application built as a Capstone Project for the **Data Science & AI Certification** program. It utilizes state-of-the-art Natural Language Processing (NLP) techniques to analyze ESG-related news and provide:

- ESG **Risk Level Classification**
- ESG **Risk Score Prediction**
- **Sentiment Analysis** using FinBERT
- **RAG (Retrieval-Augmented Generation)** based Question Answering
- **SHAP Explainability** for model interpretation
- **Report Generation** for key insights

---

## 📊 Features

- ✅ ESG Risk Level: Classify headlines as **High**, **Medium**, or **Low** risk.
- ✅ ESG Risk Score: Predict a continuous ESG risk score from 0 to 1.
- ✅ FinBERT Sentiment: Analyze sentiment of ESG news (Positive/Negative/Neutral).
- ✅ RAG-based QA: Ask questions and get context-rich answers from the dataset.
- ✅ SHAP Explainability: Understand model predictions visually.
- ✅ PDF Report Export: Download the analysis as a structured report.

---

## 📁 Dataset

- ESG News Dataset containing:  
  `date`, `headline`, `text`  
- Company names extracted using spaCy-based Named Entity Recognition (NER)

---

## 🏗️ Tech Stack

- **Frontend:** Streamlit  
- **Backend:** Python, Transformers (HuggingFace), SentenceTransformers  
- **ML Libraries:** Scikit-learn, PyTorch, SHAP  
- **Models:**  
  - Classification: RandomForest, Logistic Regression  
  - Regression: Ridge, XGBoost  
  - NLP: FinBERT, `sentence-transformers/all-MiniLM-L6-v2`

---

## 📥 How to Run Locally

### 🛠️ Install Dependencies

```bash
conda create -n esg310 python=3.10
conda activate esg310

pip install -r requirements.txt
```

### 🚀 Run the App 

```bash
streamlit run ESG_app.py
```
## ❓ **Sample RAG Questions**

You can try asking the following questions in the app to test the Retrieval-Augmented Generation (RAG) feature:

"What are the key ESG risks mentioned for Shell?"

"Which company is involved in human rights violations?"

"Are there governance-related risks for Tesla?"









