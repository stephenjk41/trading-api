from pydantic import BaseModel, Field, PrivateAttr
import numpy as np
import pandas as pd
from typing import Any
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
import tempfile
from datetime import datetime, UTC
from alpaca.data.timeframe import TimeFrame
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from pathlib import Path
from trading_api.models.training_parameters import TrainingParameters
from trading_api.database import fs, model_entry_db
from bson import ObjectId
import itertools
import json
from trading_api.env import API_KEY, SECRET_KEY


class TrainingModel(BaseModel):
    parameters: TrainingParameters = Field(..., description="Model parameters")
    _raw_data: pd.DataFrame | None = PrivateAttr(None)
    _training_data: tuple[np.array, np.array] | None = PrivateAttr(None)
    _test_data: tuple[np.array, np.array] | None = PrivateAttr(None)
    _scaler: Any | None = PrivateAttr(None)
    _times: np.ndarray | None = PrivateAttr(None)
    model: Any | None = Field(None, description="The model")

    def run(self):
        self.check_if_ran()
        self.get_and_save_stock_data()
        self.preprocess()
        self.create_model()
        self.work_model()
        return True

    def get_and_save_stock_data(self):
        client = StockHistoricalDataClient(api_key=API_KEY, secret_key=SECRET_KEY)
        today = datetime.now(tz=UTC)
        request = StockBarsRequest(
            symbol_or_symbols=self.parameters.symbol,
            timeframe=TimeFrame.Day,
            start=datetime(2020, 1, 1),
            end=datetime(today.year, today.month, today.day),
        )
        req = client.get_stock_bars(request)
        self._raw_data = req.df
        data = json.loads(req.json())
        d = pd.DataFrame(data["data"][f"{self.parameters.symbol}"])
        self._times = [row[1] for row in d.values.tolist()]

    def check_if_ran(self):
        models = model_entry_db.get_collection("model_manager")
        m = models.find().limit(1)

        try:
            models_ran = m[0]["models_list"]
            if self.parameters.symbol in models_ran:
                return
            else:
                models_ran.append(self.parameters.symbol)
                models.find_one_and_update(
                    {"_id": ObjectId(m[0]["_id"])},
                    {"$set": {"models_list": models_ran}},
                )
        except IndexError:
            models_ran = 0
            # insert
            models.insert_one({"models_list": [self.parameters.symbol]})

    def preprocess(self):
        df = self._raw_data["open"].values.reshape(-1, 1)
        ds_train = np.array(df[: int(df.shape[0] * 0.8)])  # 80% of the dataset
        ds_test = np.array(
            df[int(df.shape[0] * 0.8) - 50 :]
        )  # get the rest of the data and 50 previous records

        scaler = MinMaxScaler(feature_range=(0, 1))
        ds_train = scaler.fit_transform(ds_train)
        ds_test = scaler.transform(
            ds_test
        )  # dont need to fit because we just fit scaler already knows the fit from training data

        # Create the dataset
        x_train, y_train = self._create_dataset(ds_train)
        x_test, y_test = self._create_dataset(ds_test)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        self._training_data = (x_train, y_train)
        self._test_data = (x_test, y_test)
        self._scaler = scaler

    @staticmethod
    def _create_dataset(ds: np.array) -> tuple[np.array, np.array]:
        x = []
        y = []
        for i in range(50, ds.shape[0]):
            x.append(ds[i - 50 : i, 0])
            y.append(ds[i, 0])
        x = np.array(x)
        y = np.array(y)
        return x, y

    def create_model(self):
        # set up sequential network
        model = Sequential()
        model.add(
            LSTM(
                units=self.parameters.LSTM_units,
                return_sequences=True,
                input_shape=(self._training_data[0].shape[1], 1),
            )
        )  # input shapes (time_steps (50 stock prices/days), features (opening days))
        model.add(Dropout(self.parameters.dropout))  # cutoff 20% of nuerons
        model.add(LSTM(units=self.parameters.LSTM_units, return_sequences=True))
        model.add(Dropout(self.parameters.dropout))
        model.add(LSTM(units=self.parameters.LSTM_units))
        model.add(Dropout(self.parameters.dropout))
        model.add(Dense(units=1))  # this will be the predicted price
        model.summary()  # show the summary of the network

        model.compile(
            loss=self.parameters.loss_type, optimizer=self.parameters.optimizer
        )
        model.fit(
            self._training_data[0],
            self._training_data[1],
            epochs=self.parameters.epochs,
            batch_size=self.parameters.batch_size,
        )  # epochs and batch size are hyper params, epochs is how many times the data is trained
        self.model = model

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / f"{self.parameters.symbol}.keras"
            model.save(path)
            file_id = fs.put(
                path.read_bytes(), filename=f"{self.parameters.symbol}.keras"
            )
            models = model_entry_db.get_collection("model")
            models.insert_one(
                {
                    "parameters": self.parameters.model_dump(),
                    "file": file_id,
                    "symbol": self.parameters.symbol,
                }
            )

    def work_model(self):
        predictions = self.model.predict(self._test_data[0])
        predictions = self._scaler.inverse_transform(predictions)
        raw_data = self._raw_data["open"].values.tolist()
        models = model_entry_db.get_collection("model_data")
        models.insert_one(
            {
                "symbol": self.parameters.symbol,
                "date": datetime.now(tz=UTC).isoformat(),
                "predicted": list(itertools.chain.from_iterable(predictions.tolist())),
                "real_data": raw_data,
                "training_data_size": len(self._training_data[1]),
                "times": self._times,
            }
        )

    def work_model_and_plot(self):
        predictions = self.model.predict(self._test_data[0])
        predictions = self._scaler.inverse_transform(predictions)
        fig, ax = plt.subplots(figsize=(8, 4))
        raw_data = self._raw_data["open"].values.reshape(-1, 1)
        plt.plot(raw_data, color="red", label="Original Price")
        ax.plot(
            range(
                len(self._training_data[1]) + 50,
                len(self._training_data[1]) + 50 + len(predictions),
            ),
            predictions,
            color="blue",
            label="Predicted Price",
        )
        plt.legend()
        plt.savefig(f"./{self.parameters.symbol}_predictions.png")
        plt.show()

    @classmethod
    def from_model_in_db(cls, model_id: str):
        # get from mongo
        col = model_entry_db.get_collection("models")
        doc = col.find_one({"_id": ObjectId(model_id)})
        # get parameters
        params = TrainingParameters.model_validate(doc["parameters"])
        # get model_file
        model_file_id = doc["file"]
        model_file = fs.get(model_file_id)
        with tempfile.TemporaryDirectory() as temp_dir:
            p = Path(temp_dir) / f"{params.symbol}.keras"
            p.write_bytes(model_file.read())
            model = load_model(p)

        training = cls(parameters=params, model=model)
        # get training data
        training.get_and_save_stock_data()
        training._raw_data.to_csv("./test.csv")
        training.preprocess()
        return training
