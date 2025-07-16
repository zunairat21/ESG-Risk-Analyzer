{\rtf1\ansi\ansicpg1252\cocoartf2639
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;\f1\fnil\fcharset0 AppleColorEmoji;\f2\froman\fcharset0 Times-Roman;
\f3\fnil\fcharset134 STSongti-SC-Regular;\f4\ftech\fcharset77 Symbol;\f5\fmodern\fcharset0 Courier;
\f6\froman\fcharset0 Times-Bold;}
{\colortbl;\red255\green255\blue255;\red251\green2\blue7;\red0\green0\blue0;\red251\green2\blue255;
\red251\green2\blue7;}
{\*\expandedcolortbl;;\cssrgb\c100000\c14913\c0;\cssrgb\c0\c0\c0;\cssrgb\c100000\c25279\c100000;
\cssrgb\c100000\c14913\c0;}
{\*\listtable{\list\listtemplateid1\listhybrid{\listlevel\levelnfc23\levelnfcn23\leveljc0\leveljcn0\levelfollow0\levelstartat1\levelspace360\levelindent0{\*\levelmarker \{disc\}}{\leveltext\leveltemplateid1\'01\uc0\u8226 ;}{\levelnumbers;}\fi-360\li720\lin720 }{\listname ;}\listid1}
{\list\listtemplateid2\listhybrid{\listlevel\levelnfc23\levelnfcn23\leveljc0\leveljcn0\levelfollow0\levelstartat1\levelspace360\levelindent0{\*\levelmarker \{disc\}}{\leveltext\leveltemplateid101\'01\uc0\u8226 ;}{\levelnumbers;}\fi-360\li720\lin720 }{\listname ;}\listid2}
{\list\listtemplateid3\listhybrid{\listlevel\levelnfc23\levelnfcn23\leveljc0\leveljcn0\levelfollow0\levelstartat1\levelspace360\levelindent0{\*\levelmarker \{disc\}}{\leveltext\leveltemplateid201\'01\uc0\u8226 ;}{\levelnumbers;}\fi-360\li720\lin720 }{\listname ;}\listid3}
{\list\listtemplateid4\listhybrid{\listlevel\levelnfc23\levelnfcn23\leveljc0\leveljcn0\levelfollow0\levelstartat1\levelspace360\levelindent0{\*\levelmarker \{disc\}}{\leveltext\leveltemplateid301\'01\uc0\u8226 ;}{\levelnumbers;}\fi-360\li720\lin720 }{\listname ;}\listid4}}
{\*\listoverridetable{\listoverride\listid1\listoverridecount0\ls1}{\listoverride\listid2\listoverridecount0\ls2}{\listoverride\listid3\listoverridecount0\ls3}{\listoverride\listid4\listoverridecount0\ls4}}
\paperw11900\paperh16840\margl1440\margr1440\vieww15260\viewh11940\viewkind0
\pard\tx566\tx1133\tx1700\tx2267\tx2834\tx3401\tx3968\tx4535\tx5102\tx5669\tx6236\tx6803\pardirnatural\partightenfactor0

\f0\fs24 \cf0 # 
\f1 \uc0\u55356 \u57137 
\f0  ESG Risk Analyzer 
\f1 \uc0\u55358 \u56800 \u55357 \u56522 
\f0 \
\
**Capstone Project | Data Science & AI Certification**  \
**Author**: Zunaira Tariq | Aspiring Data Scientist & AI/ML Engineer\
\
---\
\
## 
\f1 \uc0\u55357 \u56524 
\f0  \cf2 Project Overview\cf0 \
\
The **ESG Risk Analyzer** is a Streamlit-based application designed to assess a company's Environmental, Social, and Governance (ESG) risks. It integrates machine learning models, sentiment analysis, and Retrieval-Augmented Generation (RAG) to provide:\
\
- ESG Risk Level Classification (High, Medium, Low, Very High)\
- ESG Risk Score Regression (0\'96100 scale)\
- News Headline Sentiment Analysis using fine-tuned FinBERT\
- ESG-related Q&A using a RAG-based NLP system\
- Model Explainability using SHAP\
- Downloadable ESG reports and visual insights\
\
---\
## 
\f1 \uc0\u55358 \u56800 
\f0  \cf2 Key Features\cf0 \
\
| Feature | Description |\
|--------|-------------|\
| **ESG Risk Classification** | Classifies companies into ESG risk levels using a LightGBM classifier |\
| **ESG Score Prediction** | Predicts ESG risk score using a Random Forest Regressor |\
| **Sentiment Analysis** | Fine-tuned FinBERT sentiment model to assess ESG relevance and tone |\
| **RAG-based Q&A** | Uses sentence embeddings + cosine similarity to retrieve ESG news insights |\
| **Explainable AI (SHAP)** | Visual SHAP summary plots for both classification and regression models |\
| **Report Generation** | Downloadable textual and visual ESG reports in `.txt` and `.png` format |\
\
---\
\
## 
\f1 \uc0\u55357 \u56770 \u65039 
\f0  \cf2 Project Structure\
\pard\pardeftab720\partightenfactor0

\f1 \cf0 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec3 \uc0\u55357 \u56513 
\f2  ESG_Risk_Analyzer/\

\f3 \'a9\'a6
\f2 \

\f3 \'a9\'c0\'a9\'a4\'a9\'a4
\f2  ESG_app.py # Streamlit application script\

\f3 \'a9\'c0\'a9\'a4\'a9\'a4
\f2  company_esg_financial_dataset.csv # Dataset used for training models\

\f3 \'a9\'c0\'a9\'a4\'a9\'a4
\f2  esg_balanced_sentiment.csv # Labeled and balanced sentiment dataset\

\f3 \'a9\'c0\'a9\'a4\'a9\'a4
\f2  ESG_Sentiment_Label.ipynb # Sentiment training notebook\

\f3 \'a9\'c0\'a9\'a4\'a9\'a4
\f2  ESG_Model_Training.ipynb # Classification + regression model training\

\f3 \'a9\'c0\'a9\'a4\'a9\'a4
\f2  finetuned_finbert_esg/ # Directory with fine-tuned FinBERT model\

\f3 \'a9\'c0\'a9\'a4\'a9\'a4
\f2  assets/\

\f3 \'a9\'a6
\f2  
\f3 \'a9\'c0\'a9\'a4\'a9\'a4
\f2  shap_classification_summary.png\

\f3 \'a9\'a6
\f2  
\f3 \'a9\'b8\'a9\'a4\'a9\'a4
\f2  shap_regression_summary.png\

\f3 \'a9\'c0\'a9\'a4\'a9\'a4
\f2  *.joblib # Saved models and encoders\

\f3 \'a9\'a6
\f2  
\f3 \'a9\'b8\'a9\'a4\'a9\'a4
\f2  rf_regression_pipeline.joblib\

\f3 \'a9\'a6
\f2  
\f3 \'a9\'b8\'a9\'a4\'a9\'a4
\f2  lgb_model.joblib\

\f3 \'a9\'a6
\f2  
\f3 \'a9\'b8\'a9\'a4\'a9\'a4
\f2  preprocessor.joblib\

\f3 \'a9\'a6
\f2  
\f3 \'a9\'b8\'a9\'a4\'a9\'a4
\f2  label_encoder.joblib\

\f3 \'a9\'a6
\f2  
\f3 \'a9\'b8\'a9\'a4\'a9\'a4
\f2  esg_embeddings.pkl # News embeddings for RAG\

\f3 \'a9\'a6
\f2  
\f3 \'a9\'b8\'a9\'a4\'a9\'a4
\f2  esg_metadata.pkl # Metadata used for RAG document retrieval\

\f3 \'a9\'c0\'a9\'a4\'a9\'a4
\f2  README.md # Project documentation (this file)\
\
---\
\
## 
\f1 \uc0\u9881 \u65039 
\f2 \cf2  How It Works\
\cf0 \
### 1. **\cf4 Input Company Details\cf0 **\
Fill in company-specific financials and ESG scores in the sidebar.\
\
### 2. **\cf4 Make Predictions\cf0 **\
- **Risk Level**: Uses LightGBM Classifier\
- **Risk Score**: Uses Random Forest Regressor\
\
### 3. **\cf4 Analyze ESG News\cf0 **\
- Input news headline 
\f4 \uc0\u8594 
\f2  get sentiment (Positive, Neutral, Negative, Not ESG)\
- Ask a question 
\f4 \uc0\u8594 
\f2  RAG retrieves top 3 relevant ESG news items\
\
### 4. **\cf4 Download Reports\cf0 **\
- Get a `.txt` ESG summary and `.png` ESG bar chart.\
\
---\
\
## 
\f1 \uc0\u55357 \u56522 
\f2  \cf2 Machine Learning Models\
\cf0 \
| Task | Model | Notes |\
|------|-------|-------|\
| Risk Classification | `LGBMClassifier` | Balanced with SMOTE |\
| Risk Score Regression | `RandomForestRegressor` | Pipeline with preprocessing |\
| Sentiment Analysis | `FinBERT` (fine-tuned) | 4-class model: positive, negative, neutral, not_esg |\
| RAG Q&A | `SentenceTransformer (MiniLM-L6-v2)` | Used with cosine similarity |\
\
---\
\
## 
\f1 \uc0\u55358 \u56810 
\f2  \cf2 Training Overview\
\cf0 \
- Used `scikit-learn`, `imbalanced-learn`, `transformers`, `datasets`, `sentence-transformers`, `SHAP`\
- Manual labeling of 500+ ESG news headlines\
- Applied class balancing via upsampling\
- Fine-tuned `FinBERT` on balanced dataset\
- Generated SHAP plots for model transparency\
\
## 
\f1 \uc0\u55357 \u56507 
\f2  How to Run Locally\
\
### 
\f1 \uc0\u55357 \u57056 \u65039 
\f2  Install Dependencies\
\
### 
\f1 \uc0\u55358 \u56830 
\f2  \cf2 Generate requirements.txt\cf0 \
\
To create `requirements.txt` after installing all libraries:\
```bash\
pip freeze > requirements.txt\
\
\pard\pardeftab720\sa240\partightenfactor0
\cf0 \strokec3 Create a 
\f5\fs26 requirements.txt
\f2\fs24  file using:\
\pard\pardeftab720\partightenfactor0
\cf0 \strokec3 pip freeze > requirements.txt\
\
\
\pard\pardeftab720\sl312\slmult1\partightenfactor0
\cf0 \outl0\strokewidth0 ##\cf5  
\f1 \uc0\u55357 \u56492 
\f2  Sample Questions for ESG News Insights (RAG)\
\
Ask your own question or try one of the examples below:\
\cf0 \
\pard\pardeftab720\partightenfactor0
\cf0 \outl0\strokewidth0 \strokec3 - 
\f1 \uc0\u55357 \u56589 
\f2  *"Has Tesla been involved in any environmental issues recently?"\
- 
\f1 \uc0\u55357 \u56589 
\f2  *"Which companies have faced ESG controversies this month?"*\
- 
\f1 \uc0\u55357 \u56589 
\f2  *"What governance-related news is associated with Apple?"*\
\
\
\
\pard\pardeftab720\sl360\slmult1\partightenfactor0
\cf2 ## 
\f1 \uc0\u55358 \u56813 
\f2  Ideas for Extension\
\pard\pardeftab720\sa298\partightenfactor0

\f1\fs36 \cf0 \strokec3 \uc0\u55358 \u56813 
\f6\b  Ideas for Extension\
\pard\pardeftab720\sa240\partightenfactor0

\f2\b0\fs24 \cf0 While the current version of the ESG Risk Analyzer is fully functional and delivers value across multiple ESG-related tasks, the following ideas could further improve its performance, user experience, and impact:\
\pard\pardeftab720\sa280\partightenfactor0

\f1\fs28 \cf0 \uc0\u55357 \u56633 
\f6\b  1. RAG Model Enhancements\
\pard\tx220\tx720\pardeftab720\li720\fi-720\sa240\partightenfactor0
\ls1\ilvl0
\f2\b0\fs24 \cf0 \kerning1\expnd0\expndtw0 \outl0\strokewidth0 {\listtext	\uc0\u8226 	}\expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec3 Fine-tune the embedding model on ESG-specific text for better contextual retrieval.\
\ls1\ilvl0\kerning1\expnd0\expndtw0 \outl0\strokewidth0 {\listtext	\uc0\u8226 	}\expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec3 Expand the document database to include real-time ESG feeds and filings.\
\pard\pardeftab720\sa280\partightenfactor0

\f1\fs28 \cf0 \uc0\u55357 \u56633 
\f6\b  2. Sentiment Model Improvements\
\pard\tx220\tx720\pardeftab720\li720\fi-720\sa240\partightenfactor0
\ls2\ilvl0
\f2\b0\fs24 \cf0 \kerning1\expnd0\expndtw0 \outl0\strokewidth0 {\listtext	\uc0\u8226 	}\expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec3 Increase training data size and diversity to boost accuracy.\
\ls2\ilvl0\kerning1\expnd0\expndtw0 \outl0\strokewidth0 {\listtext	\uc0\u8226 	}\expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec3 Explore newer transformer models like RoBERTa or DeBERTa for ESG sentiment tasks.\
\pard\pardeftab720\sa280\partightenfactor0

\f1\fs28 \cf0 \uc0\u55357 \u56633 
\f6\b  3. Interactive User Experience\
\pard\tx220\tx720\pardeftab720\li720\fi-720\sa240\partightenfactor0
\ls3\ilvl0
\f2\b0\fs24 \cf0 \kerning1\expnd0\expndtw0 \outl0\strokewidth0 {\listtext	\uc0\u8226 	}\expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec3 Add feedback collection to continuously improve predictions.\
\ls3\ilvl0\kerning1\expnd0\expndtw0 \outl0\strokewidth0 {\listtext	\uc0\u8226 	}\expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec3 Support multi-lingual inputs for global ESG reporting.\
\pard\pardeftab720\sa280\partightenfactor0

\f1\fs28 \cf0 \uc0\u55357 \u56633 
\f6\b  4. Deployment\
\pard\tx220\tx720\pardeftab720\li720\fi-720\sa240\partightenfactor0
\ls4\ilvl0
\f2\b0\fs24 \cf0 \kerning1\expnd0\expndtw0 \outl0\strokewidth0 {\listtext	\uc0\u8226 	}\expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec3 Host the app on platforms like 
\f6\b Streamlit Cloud
\f2\b0 , 
\f6\b Hugging Face Spaces
\f2\b0 , or 
\f6\b Heroku
\f2\b0 .\
\ls4\ilvl0\kerning1\expnd0\expndtw0 \outl0\strokewidth0 {\listtext	\uc0\u8226 	}\expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec3 Add authentication or team dashboards for enterprise use.\
}