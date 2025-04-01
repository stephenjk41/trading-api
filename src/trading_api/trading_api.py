"""Main module."""

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest

from datetime import datetime, UTC
from alpaca.data.timeframe import TimeFrame
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from pathlib import Path
from trading_api.env import API_KEY, SECRET_KEY


def main():
    train_on_symbol("TSLA")


def train_on_symbol(symbol: str):
    get_and_save_stock_data(symbol)
    p = Path(f"./{symbol}.csv")
    df = pd.read_csv(p)
    train_set, test_set, scaler = preprocess(df)
    create_model(train_set, symbol)
    work_model(Path(f"./{symbol}_model.keras"), train_set, test_set, scaler, df)


def get_and_save_stock_data(symbol: str):
    client = StockHistoricalDataClient(api_key=API_KEY, secret_key=SECRET_KEY)
    today = datetime.now(tz=UTC)
    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Day,
        start=datetime(2020, 1, 1),
        end=datetime(today.year, today.month, today.day),
    )
    df = client.get_stock_bars(request).df
    df.to_csv(f"./{symbol}.csv")


def preprocess(df: pd.DataFrame):
    df = df["open"].values.reshape(-1, 1)
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
    x_train, y_train = create_dataset(ds_train)
    x_test, y_test = create_dataset(ds_test)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    return (x_train, y_train), (x_test, y_test), scaler


def create_model(train_data: tuple[np.array, np.array], symbol: str):
    # set up sequential network
    model = Sequential()
    model.add(
        LSTM(units=200, return_sequences=True, input_shape=(train_data[0].shape[1], 1))
    )  # input shapes (time_steps (50 stock prices/days), features (opening days))
    model.add(Dropout(0.2))  # cutoff 20% of nuerons
    model.add(LSTM(units=200, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=200))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))  # this will be the predicted price
    model.summary()  # show the summary of the network

    model.compile(loss="mean_squared_error", optimizer="adam")
    model.fit(
        train_data[0], train_data[1], epochs=100, batch_size=20
    )  # epochs and batch size are hyper params, epochs is how many times the data is trained
    model.save(Path(f"./{symbol}_model.keras"))


def work_model(p: Path, train_data, test_data: tuple[np.array, np.array], scaler, df):
    model = load_model(p)
    predictions = model.predict(test_data[0])
    predictions = scaler.inverse_transform(predictions)

    fig, ax = plt.subplots(figsize=(8, 4))
    plt.plot(df["open"], color="red", label="Original Price")
    ax.plot(
        range(len(train_data[0]) + 50, len(train_data[1]) + 50 + len(predictions)),
        predictions,
        color="blue",
        label="Predicted Price",
    )
    plt.legend()
    plt.show()


def create_dataset(ds: np.array) -> tuple[np.array, np.array]:
    x = []
    y = []
    for i in range(50, ds.shape[0]):
        x.append(ds[i - 50 : i, 0])
        y.append(ds[i, 0])
    x = np.array(x)
    y = np.array(y)
    return x, y


if __name__ == "__main__":
    main()
