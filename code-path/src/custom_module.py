import mlflow.pyfunc
class SklearnModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model

    def predict(self, x, n_neighbors=4):
        return self.model.kneighbors(x, n_neighbors)