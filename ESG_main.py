# ESG Risk Analyzer - Main Script 

# Phase 01: Import Libraries
import os
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, classification_report
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from joblib import dump


# Phase 02: Load & Preprocess Data
data = pd.read_csv("company_esg_financial_dataset.csv")
data.drop_duplicates(inplace=True)

imputer = SimpleImputer(strategy='median')
data['GrowthRate'] = imputer.fit_transform(data[['GrowthRate']])

data.drop(columns=['CompanyID', 'CompanyName', 'Year', 'CarbonEmissions', 'WaterUsage', 'EnergyConsumption'], inplace=True)

# Phase 03: Regression - Predict ESG_Overall (Numeric Risk Score)
X_reg = data.drop(columns=['ESG_Overall'])
y_reg = data['ESG_Overall']
num_cols = X_reg.select_dtypes(include=['float64', 'int64']).columns
cat_cols = X_reg.select_dtypes(include=['object']).columns

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
])

reg_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(random_state=42))
])

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
reg_pipeline.fit(X_train_reg, y_train_reg)

# Regression Evaluation
y_pred_reg = reg_pipeline.predict(X_test_reg)
print(f"\n📊 RMSE (Regression): {np.sqrt(mean_squared_error(y_test_reg, y_pred_reg)):.2f}")

# Regression SHAP Explainability
reg_model = reg_pipeline.named_steps['model']
X_sample_reg_pre = reg_pipeline.named_steps['preprocessor'].transform(X_test_reg[:100])
if hasattr(X_sample_reg_pre, 'toarray'):
    X_sample_reg_pre = X_sample_reg_pre.toarray()
feature_names_reg = [name.split("__")[-1] for name in preprocessor.get_feature_names_out()]
explainer_reg = shap.Explainer(reg_model, X_sample_reg_pre)
shap_values_reg = explainer_reg(X_sample_reg_pre)

# Save SHAP regression plot
os.makedirs("assets", exist_ok=True)
shap.summary_plot(shap_values_reg, X_sample_reg_pre, feature_names=feature_names_reg, show=False)
plt.tight_layout()
plt.savefig("assets/shap_regression_summary.png", bbox_inches='tight')
plt.clf()

# Save regression model
dump(reg_pipeline, "rf_regression_pipeline.joblib")

# Phase 04: Create Risk Level (Classification Target)
def bin_esg_risk(score):
    if score > 80:
        return 'Low_Risk'
    elif score > 50:
        return 'Medium_Risk'
    elif score > 20:
        return 'High_Risk'
    else:
        return 'Very_High_Risk'

data['ESG_Risk_Level'] = data['ESG_Overall'].apply(bin_esg_risk)

# Phase 05: Classification - Predict ESG Risk Level
X_cls = X_reg.copy()
y_cls = data['ESG_Risk_Level']
label_encoder = LabelEncoder()
y_cls_encoded = label_encoder.fit_transform(y_cls)

# Print class mappings
print("\n✅ Class label encoding:")
class_mappings = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
for k, v in class_mappings.items():
    print(f"{k} → {v}")

X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X_cls, y_cls_encoded, test_size=0.2, random_state=42)
X_train_cls_processed = preprocessor.fit_transform(X_train_cls)

# Balance with SMOTE
smote = SMOTE(random_state=42)
X_train_cls_resampled, y_train_cls_resampled = smote.fit_resample(X_train_cls_processed, y_train_cls)

# Train Classifier
clf_model = LGBMClassifier(n_jobs=1, random_state=42)
clf_model.fit(X_train_cls_resampled, y_train_cls_resampled)

# Classification Evaluation
X_test_cls_processed = preprocessor.transform(X_test_cls)
y_pred_cls = clf_model.predict(X_test_cls_processed)
print("\n📈 Classification Report:\n")
print(classification_report(y_test_cls, y_pred_cls, target_names=label_encoder.classes_))

# SHAP Explainability - Classification
X_sample_cls = X_cls.sample(100, random_state=42)
X_sample_cls_pre = preprocessor.transform(X_sample_cls)
if hasattr(X_sample_cls_pre, 'toarray'):
    X_sample_cls_pre = X_sample_cls_pre.toarray()
feature_names_cls = [name.split("__")[-1] for name in preprocessor.get_feature_names_out()]
explainer_cls = shap.Explainer(clf_model, X_sample_cls_pre)
shap_values_cls = explainer_cls(X_sample_cls_pre)

# Save SHAP classification plot
shap.summary_plot(shap_values_cls, X_sample_cls_pre, feature_names=feature_names_cls, show=False)
plt.tight_layout()
plt.savefig("assets/shap_classification_summary.png", bbox_inches='tight')
plt.clf()

# Save classification artifacts
dump(clf_model, "lgb_model.joblib")
dump(label_encoder, "label_encoder.joblib")
dump(preprocessor, "preprocessor.joblib")

print("\n✅ SHAP Plots Generated & Model Files Saved.")

print("✅ All models trained and saved.")
 







