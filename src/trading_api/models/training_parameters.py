from pydantic import BaseModel, Field


class TrainingParameters(BaseModel):
    symbol: str = Field(
        ..., description="The stock symbol to predict, must be US exchange"
    )
    LSTM_units: int = Field(96, description="Units")
    dropout: float = Field(0.2, description="Dropout percentage")
    loss_type: str = Field("mean_squared_error", description="Loss type")
    optimizer: str = Field("adam", description="optimizer type")
    epochs: int = Field(5, description="hyper parameter")  # for testing
    batch_size: int = Field(20, description="hyper parameter")
    training_set_size: int = Field(50, description="Training data size in days")
