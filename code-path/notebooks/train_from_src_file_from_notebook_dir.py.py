# Databricks notebook source
# MAGIC %md
# MAGIC # Train Custom Model in a SRC File From Notebook Directory 

# COMMAND ----------

import mlflow

print(mlflow.__version__)

# COMMAND ----------

# MAGIC %pip install --upgrade mlflow

# COMMAND ----------

from sklearn.neighbors import NearestNeighbors
import numpy as np
import mlflow
import inspect
import sys
import os

sys.path.append(os.path.abspath(".."))

from src.custom_module import SklearnModelWrapper

# COMMAND ----------

print(mlflow.__version__)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train and log model

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

model = SklearnModelWrapper(neigh)

model.predict(x_test, n_neighbors=1)

# COMMAND ----------

code_path = inspect.getfile(SklearnModelWrapper)
code_path

# COMMAND ----------

# log model and manual deploy as API in model registry
mlflow.sklearn.log_model(
    sk_model=model,
    artifact_path="model",
    registered_model_name="stack_databricks_bugs_code_path_src_file",
    code_paths=[code_path],
    input_example=x_test,
)

# COMMAND ----------

# re-load and test predict
model_uri = "models:/stack_databricks_bugs_code_path_src_file/2"
loaded_model = mlflow.pyfunc.load_model(model_uri)

print(loaded_model)
loaded_model.predict([[0, 0, 1.3]])
