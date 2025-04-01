from fastapi import APIRouter
from trading_api.tasks.task import train_on_symbol
from trading_api.models.training_parameters import TrainingParameters
import os
from alpaca.data import StockBarsRequest, StockHistoricalDataClient
from datetime import datetime, UTC
from alpaca.data import TimeFrame
import json
import pandas as pd
from trading_api.database import model_entry_db, celery_db

from trading_api.env import API_KEY, SECRET_KEY

router = APIRouter()


@router.post("/train")
def generate_model_and_performance(params: TrainingParameters):
    task = train_on_symbol.delay(params.model_dump())
    return {"symbol": params.symbol, "task_id": task.task_id}


@router.get("/train/status/{task_id}")
def get_training_status(task_id: str):
    celery_task_collect = celery_db.get_collection("celery_taskmeta")
    query = celery_task_collect.find({"_id": task_id}, sort=[("_id", -1)]).limit(1)
    try:
        status = query[0]["status"]
        return {"status": status}
    except IndexError:
        return {"status": "NOT_READY"}


@router.get("/models/latest")
def get_latest_models():
    models = model_entry_db.get_collection("model_manager")
    m = models.find().limit(1)
    return {"models_ran": m[0]["models_list"]}


@router.get("/{symbol}/data")
def get_symbol_data(symbol: str):
    client = StockHistoricalDataClient(api_key=API_KEY, secret_key=SECRET_KEY)
    today = datetime.now(tz=UTC)
    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Day,
        start=datetime(2020, 1, 1),
        end=datetime(today.year, today.month, today.day),
    )
    data = json.loads(client.get_stock_bars(request).json())
    d = pd.DataFrame(data["data"][f"{symbol}"])
    n = d.drop("symbol", axis=1)
    return n.values.tolist()


@router.get("/{symbol}/predicted_data")
def get_predicted_symbol_data(symbol: str):
    models = model_entry_db.get_collection("model_data")
    dec = models.find({"symbol": symbol}, sort=[("_id", -1)]).limit(1)
    return {
        "symbol": dec[0]["symbol"],
        "raw_data": dec[0]["real_data"],
        "predictions": dec[0]["predicted"],
        "training_data_size": dec[0]["training_data_size"],
        "times": dec[0]["times"],
    }


@router.get("/{symbol}/parameters")
def get_training_parameters(symbol: str):
    models = model_entry_db.get_collection("model")
    dec = models.find({"symbol": symbol}, sort=[("_id", -1)]).limit(1)
    return {
        "symbol": dec[0]["symbol"],
        "LSTM_units": dec[0]["parameters"]["LSTM_units"],
        "dropout": dec[0]["parameters"]["dropout"],
        "loss_type": dec[0]["parameters"]["loss_type"],
        "optimizer": dec[0]["parameters"]["optimizer"],
        "epochs": dec[0]["parameters"]["epochs"],
        "batch_size": dec[0]["parameters"]["batch_size"],
        "training_set_size": dec[0]["parameters"]["training_set_size"],
    }
