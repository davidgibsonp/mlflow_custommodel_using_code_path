# Databricks notebook source
# MAGIC %md
# MAGIC # Train Custom Model in a Single File
# MAGIC 
# MAGIC When logging a custom model where the source code is in the same file we can successfully load the logged model **and** deploy as an API **and** load from a new directory.

# COMMAND ----------

import mlflow
print(mlflow.__version__)

# COMMAND ----------

# MAGIC %pip install --upgrade mlflow

# COMMAND ----------

from sklearn.neighbors import NearestNeighbors
import numpy as np
import mlflow

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load model for inference

# COMMAND ----------

# fake data
samples = [[0, 0, 2], [1, 0, 0], [0, 0, 1], [0, 2, 1], [1, 0, 1]]
x_test = [[0, 0, 1.3]]

# train model
neigh = NearestNeighbors(n_neighbors=2, radius=0.4)
neigh.fit(samples)

# test predict
neigh.kneighbors(x_test, n_neighbors=1)

# COMMAND ----------


class SklearnModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model

    def predict(self, x, n_neighbors=4):
        return self.model.kneighbors(x, n_neighbors)


model = SklearnModelWrapper(neigh)

model.predict(x_test, n_neighbors=1)

# COMMAND ----------

mlflow.sklearn.log_model(
    sk_model=model,
    artifact_path="stack_databricks_bugs_code_path_single_file",
    registered_model_name="stack_databricks_bugs_code_path_single_file",
    input_example=x_test,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy model as API via UI
# MAGIC 
# MAGIC When deployed as an API everything works as expected.
# MAGIC 
# MAGIC ## Load model for inference
# MAGIC 
# MAGIC Works as expected.

# COMMAND ----------

# works
model_uri = "models:/stack_databricks_bugs_code_path_single_file/2"
loaded_model = mlflow.sklearn.load_model(model_uri)

loaded_model

# COMMAND ----------

loaded_model.predict([[0, 0, 1.3]])
