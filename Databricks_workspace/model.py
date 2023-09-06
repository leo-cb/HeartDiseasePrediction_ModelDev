from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import col
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import mlflow
import mlflow.pyfunc
import mlflow.spark
from pyspark.ml.classification import LogisticRegression
from mlflow.models.signature import infer_signature
from mlflow.utils.environment import _mlflow_conda_env
import cloudpickle
import pyspark
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import h2o
from h2o.automl import H2OAutoML
from pyspark.ml.classification import GBTClassifier

def get_assembled_data(features):
    """Assemble data"""
    assembler = VectorAssembler(inputCols=features, outputCol="features")
    return assembler.transform(data)

# initialize H2O
h2o.init()

# read dataset
data = spark.read.format("delta").load("dbfs:/user/hive/warehouse/heart_model")

data.show()

# features from highest feature importance to lowest
features_cols = ["cp", "ca", "thal", "oldpeak", "thalach", "exang", "age", "slope", "sex", "trestbps", "chol", "restecg", "fbs_cat"]

# features list of lists, from 1 feature to all
features = [features_cols[:i+1] for i in range(len(features_cols))]

# datasets
datasets = [get_assembled_data(f) for f in features]

token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
dbutils.fs.put("file:///root/.databrickscfg","[DEFAULT]\nhost=https://community.cloud.databricks.com\ntoken = "+token,overwrite=True)

# log model with MLflow
experiment_id = "1363294494268155"

### LOGISTIC REGRESSION

# loop through datasets
for i,dataset in enumerate(datasets):

    # split data
    train_data, validation_data, test_data = dataset.randomSplit([0.7, 0.2, 0.1], seed=42)

    ## logistic regression
    # start MLflow run
    with mlflow.start_run(run_name=f"Logistic Regression ({features[i]})"):

        # define and train the model
        lr = LogisticRegression(featuresCol="features", labelCol="target")
        lr_model = lr.fit(train_data)

        # log parameters
        mlflow.log_param("regParam", lr.getRegParam())
        mlflow.log_param("elasticNetParam", lr.getElasticNetParam())
        
        # evaluate the model and log metrics
        predictions = lr_model.transform(validation_data)
        
        # AUC
        evaluator = BinaryClassificationEvaluator(labelCol="target")
        auc = evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})
        mlflow.log_metric("auc", auc)
        
        # other metrics
        evaluator_multi = MulticlassClassificationEvaluator(labelCol="target", predictionCol="prediction")
        accuracy = evaluator_multi.evaluate(predictions, {evaluator_multi.metricName: "accuracy"})
        f1 = evaluator_multi.evaluate(predictions, {evaluator_multi.metricName: "f1"})
        precision = evaluator_multi.evaluate(predictions, {evaluator_multi.metricName: "weightedPrecision"})
        recall = evaluator_multi.evaluate(predictions, {evaluator_multi.metricName: "weightedRecall"})
        
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        
        # log the model with signature and environment
        signature = infer_signature(train_data.toPandas()[features[i]], lr_model.transform(train_data).toPandas()["prediction"])
        conda_env = _mlflow_conda_env(
            additional_pip_deps=["pyspark=={}".format(pyspark.__version__), "cloudpickle=={}".format(cloudpickle.__version__)],
        )
        mlflow.spark.log_model(lr_model, "lr_model", signature=signature, conda_env=conda_env)  

        # export the model
        model.save(f"dbfs:/user/hive/warehouse/LR_({features[i]})")
        
### AUTOML

# split data
train_data, validation_data, test_data = datasets[-1].randomSplit([0.7, 0.2, 0.1], seed=42) # use the dataset with all the features

# convert training and validation data to H2OFrame
train_data_h2o = h2o.H2OFrame(train_data.toPandas())
validation_data_h2o = h2o.H2OFrame(validation_data.toPandas())
train_data_h2o["target"] = train_data_h2o["target"].asfactor()
validation_data_h2o["target"] = validation_data_h2o["target"].asfactor()

# start MLflow run
with mlflow.start_run(run_name="AutoML"):

    # use H2O AutoML to train a model
    automl_model = H2OAutoML(
        max_runtime_secs=30*60, # 30 minutes
        stopping_metric="AUC", 
        sort_metric="AUC"
    )
    
    automl_model.train(y="target", training_frame=train_data_h2o, leaderboard_frame=validation_data_h2o)
    
    # get the best model
    best_model = automl_model.leader
    
    # predict on validation set
    predictions = best_model.predict(validation_data_h2o)
    performance = best_model.model_performance(validation_data_h2o)
    
    # convert predictions to Spark DataFrame if needed
    predictions_spark = predictions.as_data_frame(use_pandas=True)
    
    # calculate metrics
    auc = performance.auc()
    accuracy = performance.accuracy()[0][1]
    f1 = performance.F1()[0][1]
    precision = performance.precision()[0][1]
    recall = performance.recall()[0][1]
    
    # log metrics with MLflow
    mlflow.log_metric("auc", auc)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1", f1)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    
    # log the best model with MLflow
    mlflow.log_artifact(best_model.download_mojo(), "model")

    print(f"auc,accuracy,f1,precision,recall={auc}, {accuracy}, {f1}, {precision}, {recall}")

### GBT Model

# loop through datasets
for i, dataset in enumerate(datasets):

    # split data
    train_data, validation_data, test_data = dataset.randomSplit([0.7, 0.2, 0.1], seed=42)

    ## GBT Model (Gradient Boosted Trees in PySpark)
    # start MLflow run
    with mlflow.start_run(run_name=f"GBT ({features[i]})"):

        # define and train the model
        xgb = GBTClassifier(featuresCol="features", labelCol="target")
        xgb_model = xgb.fit(train_data)

        # log parameters (example: max depth)
        mlflow.log_param("maxDepth", xgb.getMaxDepth())

        # evaluate the model and log metrics
        predictions = xgb_model.transform(validation_data)

        # AUC
        evaluator = BinaryClassificationEvaluator(labelCol="target")
        auc = evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})
        mlflow.log_metric("auc", auc)

        # other metrics
        evaluator_multi = MulticlassClassificationEvaluator(labelCol="target", predictionCol="prediction")
        accuracy = evaluator_multi.evaluate(predictions, {evaluator_multi.metricName: "accuracy"})
        f1 = evaluator_multi.evaluate(predictions, {evaluator_multi.metricName: "f1"})
        precision = evaluator_multi.evaluate(predictions, {evaluator_multi.metricName: "weightedPrecision"})
        recall = evaluator_multi.evaluate(predictions, {evaluator_multi.metricName: "weightedRecall"})

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)

        # log the model with signature
        signature = infer_signature(train_data.toPandas()[features[i]], xgb_model.transform(train_data).toPandas()["prediction"])
        mlflow.spark.log_model(xgb_model, "gbt_model", signature=signature)

        # export the model
        model.save(f"dbfs:/user/hive/warehouse/GBT_({features[i]})")