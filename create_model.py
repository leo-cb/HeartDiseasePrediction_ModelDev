import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import pickle
import pyarrow as pa
import pyarrow.parquet as pq

PATH_INPUT = "input"

if __name__ != "__main__":
    exit
    
df = pd.read_csv(PATH_INPUT + "/heart_2.csv")

# =============================================================================
# MODEL WITH GBT (Databricks model could not be exported)
# =============================================================================

# feature columns and target column
feature_columns = ['cp', 'ca', 'thal', 'oldpeak', 'thalach', 'exang', 'age', 'slope', 'sex']
target_column = 'target'

# split the data into features and target
X = df[feature_columns]
y = df[target_column]

# split the data into training, validation, and test sets (70/20/10)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=123)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.33, random_state=123)

# initialize a GradientBoostingClassifier
gbt = HistGradientBoostingClassifier(random_state=123)

# train the model on the training data
gbt.fit(X_train, y_train)

# make predictions on the validation data
val_predictions = gbt.predict(X_val)

# calculate the AUC score on validation data
val_auc = roc_auc_score(y_val, val_predictions)
print("Validation AUC:", val_auc)

# make predictions on the test data
test_predictions = gbt.predict(X_test)

# calculate the AUC score on test data
test_auc = roc_auc_score(y_test, test_predictions)
print("Test AUC:", test_auc)

# =============================================================================
# EXPORT ARTIFACTS
# =============================================================================

# export model
with open("data/gbt_model.pkl", "wb") as model_file:
    pickle.dump(gbt, model_file)
    
# export datasets
X_train.to_parquet("data/X_train.parquet")
X_val.to_parquet("data/X_val.parquet")
X_test.to_parquet("data/X_test.parquet")

y_train_df = y_train.to_frame(name="target").reset_index(drop=True)
y_val_df = y_val.to_frame(name="target").reset_index(drop=True)
y_test_df = y_test.to_frame(name="target").reset_index(drop=True)

y_train_df.to_parquet("data/y_train.parquet")
y_val_df.to_parquet("data/y_val.parquet")
y_test_df.to_parquet("data/y_test.parquet")