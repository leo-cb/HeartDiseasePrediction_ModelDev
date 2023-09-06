import shap
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import argparse

if __name__ != "__main__":
    exit

# create argument parser
parser = argparse.ArgumentParser(description="Outputs and plots SHAP summary and bar plots computed in the test set, \
                                 using the saved model.")

# --show-plots arg
parser.add_argument(
    "--show-plots",
    action="store_true", 
    help="Display plots (default: False)",
)

# parse args and create show_plots bool var
args = parser.parse_args()
show_plots = args.show_plots

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

# save SHAP plots
shap.summary_plot(shap_values, X_test, show=False) # show needs to be false for figure to be saved
plt.savefig("images/shap_summary.png")  # save the plot as an image
plt.close()  # close the current plot

shap.summary_plot(shap_values, X_test, plot_type="bar", show=False) # show needs to be false for figure to be saved
plt.savefig("images/shap_bar.png")  # save the bar plot as an image
plt.close()  # close the current plot

print("Saved SHAP plots to /images/.")

# show SHAP plots
if show_plots:
    shap.summary_plot(shap_values, X_test, show=True)
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=True)