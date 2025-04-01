from celery import Celery


from typing import Any
from trading_api.models.model import TrainingModel
from trading_api.models.training_parameters import TrainingParameters

app = Celery(
    "tasks",
    broker="pyamqp://guest@localhost//",
    result_backend="mongodb://localhost:27017",
)


@app.task
def train_on_symbol(training_parameters: dict[str:Any]):
    params = TrainingParameters.model_validate(training_parameters)
    model = TrainingModel(parameters=params)
    prediction = model.run()
    return prediction
