from pymongo import MongoClient
import gridfs

client = MongoClient(host="localhost", port=27017)
model_file_db = client.trading_models_fs
model_entry_db = client.trading_models
celery_db = client.celery
model_data_db = client.trading_model_data
model_manager_db = client.model_manger
fs = gridfs.GridFS(model_file_db)
