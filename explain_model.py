import shap
import pickle
import pandas as pd

if __name__ != "__main__":
    exit

# =============================================================================
# SHAP
# =============================================================================

with open("data/gbt_model.pkl", "rb") as model_file:
    gbt = pickle.load(model_file)
    
X_train = pd.read_parquet("data/X_train.parquet")
X_test = pd.read_parquet("data/X_test.parquet")
    
explainer = shap.Explainer(gbt, X_train)

# get SHAP values for the test data
shap_values = explainer(X_test)

# SHAP plots
shap.summary_plot(shap_values, X_test)
shap.summary_plot(shap_values, X_test, plot_type="bar")