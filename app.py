import sys
import os
import dagshub
import mlflow
import numpy as np
from mlflow.tracking import MlflowClient
from feast import FeatureStore
from pydantic import BaseModel
import time
from datetime import datetime, timezone

from src.exception.exception import CreditCardException
from src.logging.logger import logging
from src.pipeline.training_pipeline import TrainingPipeline
from src.utils.main_utils.utils import load_object

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Request, Depends
from uvicorn import run as app_run
from contextlib import asynccontextmanager
from fastapi.responses import Response, JSONResponse
from starlette.responses import RedirectResponse
import pandas as pd
import pymongo
import certifi

from typing import Union


from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

# Certificate for MongoDB connection
ca = certifi.where()

# Connect to MongoDB
MONGO_DB_URL = os.getenv("MONGO_DB_URL")
mongo_client = pymongo.MongoClient(MONGO_DB_URL, tlsCAFile=ca)
db = mongo_client["FRAUD"]
collection = db["creditcardData"]

# Global variables for model and preprocessor
model = None
preprocessor = None
store = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, preprocessor, store

    try:
        # Initialize Feature Store
        store = FeatureStore(repo_path="/mnt/d/real_time_streaming/my_feature_repo/feature_repo")

        # Set up MLflow
        mlflow.set_tracking_uri("https://dagshub.com/neerajjj6785/real-time-credit-card-transaction-fraud-detection-mlops.mlflow")

        # Set up MLflow credentials (Use environment variables instead of hardcoding)
        os.environ["MLFLOW_TRACKING_USERNAME"] = "neerajjj6785"
        os.environ["MLFLOW_TRACKING_PASSWORD"] = "d4073bd126374347538627d1f4f255bffaae1de0"

        # Load model and preprocessor once at startup
        model_name = "RandomForestClassifier"
        version_number = 5
        model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{version_number}")
        preprocessor = load_object("final_model/preprocessor.pkl")

        logging.info("Model and preprocessor loaded successfully")

        # Yield control back to the app for async startup tasks
        yield

    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        model = None
        yield  # Ensure app can still start even if the model loading fails

# app = FastAPI()
app = FastAPI(lifespan=lifespan, root_path="/api", openapi_url="/openapi.json", docs_url="/docs",  redoc_url="/redoc")
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the request body schemas
class TransactionRequest(BaseModel):
    cc_num: int
    merchant: str
    category: str
    amt: float
    first: str
    last: str
    gender: str
    street: str
    city: str
    state: str
    zip: int
    lat: float
    long: float
    city_pop: int
    job: str
    dob: str
    merch_lat: float
    merch_long: float

class SparkTransactionRequest(BaseModel):
    transaction_id: int
    
class PredictionRequest(BaseModel):
    transaction_id: int
    # event_timestamp: str = None

def ensure_all_fields(user_request_dict):
    """Ensure all required fields exist in the transaction data"""
    # Current time for timestamps
    current_time = datetime.now(timezone.utc)
    current_unix = int(time.time())
    
    # Add generated fields
    user_request_dict["trans_date_trans_time"] = current_time.strftime("%Y-%m-%d %H:%M:%S")
    user_request_dict["unix_time"] = current_unix
    user_request_dict["trans_num"] = str(int(current_unix * 1000))
    user_request_dict["is_fraud"] = 0  # Initial assumption, to be predicted
    
    # Add Unnamed_0 field that might be required for model consistency
    user_request_dict["Unnamed_0"] = int(current_unix % 100000)
    
    # Define all required fields
    required_fields = {
        "Unnamed_0", "trans_date_trans_time", "cc_num", "merchant", "category", "amt", "first", "last", "gender", "street", "city", "state", "zip", "lat", "long", "city_pop", "job", "dob", "trans_num", "unix_time", 
        "merch_lat", "merch_long", "is_fraud"
    }
    
    # Ensure all required fields exist with None as default for missing ones
    return {field: user_request_dict.get(field, None) for field in required_fields}

# # Initialize MLflow tracking with DagsHub
# async def setup():
#     global model, preprocessor

#     try:
#         # Set up MLflow
#         # dagshub.init(
#         #     repo_owner="neerajjj6785",
#         #     repo_name="real-time-credit-card-transaction-fraud-detection-mlops",
#         #     mlflow=True
#         # )

#         # Initialize Feature Store
#         store = FeatureStore(repo_path="/mnt/d/real_time_streaming/my_feature_repo/feature_repo")

#         mlflow.set_tracking_uri("https://dagshub.com/neerajjj6785/real-time-credit-card-transaction-fraud-detection-mlops.mlflow")
        
#         # Set up MLflow credentials
#         os.environ["MLFLOW_TRACKING_USERNAME"] = "neerajjj6785"
#         os.environ["MLFLOW_TRACKING_PASSWORD"] = "d4073bd126374347538627d1f4f255bffaae1de0"
        
#         # Load the model
#         model_name = "RandomForestClassifier"
#         version_number = 5
#         model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{version_number}")
        
#         # Load the preprocessor
#         preprocessor = load_object("final_model/preprocessor.pkl")
#         logging.info("Model and preprocessor loaded successfully")
#     except Exception as e:
#         logging.error(f"Error during startup: {e}")

# app.router.on_startup.append(setup)

@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")

@app.get("/train")
async def train_route():
    try:
        train_pipeline = TrainingPipeline()
        train_pipeline.run_pipeline()
        return Response("Training is successful")
    except Exception as e:
        raise CreditCardException(e, sys)
    
# Temporary storage for the last transaction_id received from Spark
latest_transaction_id = None  

@app.post("/transaction")
async def create_transaction(transaction: Union[TransactionRequest, SparkTransactionRequest]):
    global latest_transaction_id  # Allow modification of global variable

    try:
        insert_result = None  # Avoid uninitialized reference

        # If request is from Spark (contains transaction_id)
        if isinstance(transaction, SparkTransactionRequest):
            latest_transaction_id = transaction.dict().get("transaction_id")
            return {"status": "success", "message": f"Received transaction_id {latest_transaction_id}"}

        # If request is from Streamlit (TransactionRequest)
        if isinstance(transaction, TransactionRequest):
            # if not latest_transaction_id:
            #     return {
            #         "status": "error",
            #         "message": "No transaction_id available from Spark yet",
            #         "transaction_id": None
            #     }

            transaction_dict = transaction.dict()

            # # Assign transaction_id from Spark to the incoming transaction request
            # transaction_dict["transaction_id"] = latest_transaction_id

            # Ensure all fields are present
            complete_transaction = ensure_all_fields(transaction_dict)

            # Insert into MongoDB
            insert_result = collection.insert_one(complete_transaction)

            if not latest_transaction_id:
                return {
                    "status": "error",
                    "message": "No transaction_id available from Spark yet",
                    "transaction_id": None
                }
            
            return {
                "status": "success",
                "message": "Transaction recorded successfully",
                "transaction_id": latest_transaction_id,  # Now correctly assigned
                "mongodb_id": str(insert_result.inserted_id)
            }

    except Exception as e:
        return {"status": "error", "message": str(e)}

# @app.post("/transaction")
# async def create_transaction(transaction: Union[TransactionRequest, SparkTransactionRequest]):
#     try:
#         transaction_id = None
#         insert_result = None  # Initialize insert_result to avoid referencing before assignment

#         # Handle SparkTransactionRequest (first extract transaction_id)
#         if isinstance(transaction, SparkTransactionRequest):
#             transaction_id = transaction.dict().get("transaction_id")

#             # For SparkTransactionRequest, no response is sent, just return
#             return {}

#         # Handle TransactionRequest (Streamlit request)
#         if isinstance(transaction, TransactionRequest):
#             # Convert Pydantic model to dictionary
#             transaction_dict = transaction.dict()

#             # Ensure all fields are present
#             complete_transaction = ensure_all_fields(transaction_dict)

#             # Insert into MongoDB
#             insert_result = collection.insert_one(complete_transaction)

#             response = {
#                 "status": "success",
#                 "message": "Transaction recorded successfully",
#                 "transaction_id": transaction_id,  # Send the transaction_id extracted from SparkTransactionRequest
#                 "mongodb_id": str(insert_result.inserted_id)
#             }

#             return response
        
        # if isinstance(transaction, TransactionRequest):
        #     # Convert Pydantic model to dictionary
        #     transaction_dict = transaction.dict()

        #     # Ensure all fields are present
        #     complete_transaction = ensure_all_fields(transaction_dict)

        #     # Insert into MongoDB
        #     insert_result = collection.insert_one(complete_transaction)

        # if isinstance(transaction, SparkTransactionRequest):
        #     transaction_dict = transaction.dict()
        #     transaction_id = transaction_dict["transaction_id"]  # For SparkTransactionRequest, use the provided transaction_id

        # # Ensure transaction_id is assigned before proceeding
        # if not transaction_id:
        #     raise ValueError("Transaction ID is missing from the request.")
        
        # print("transaction_id: ", transaction_id)
        
        # # Return the transaction ID for future reference
        # response = {
        #     "status": "success",
        #     "message": "Transaction recorded successfully",
        #     "transaction_id": transaction_id,
        #     "mongodb_id": str(insert_result.inserted_id)
        # }

        # if insert_result:  # Only include MongoDB ID if the transaction was inserted into MongoDB
        #     response["mongodb_id"] = str(insert_result.inserted_id)

        # return response

    # except Exception as e:
    #     raise CreditCardException(e, sys)
    
@app.post("/predict")
async def predict(prediction_request: PredictionRequest):
    try:
        
        # Define the features to fetch
        features = [
            "creditcard_fraud:cc_num", 
            "creditcard_fraud:amt", 
            "creditcard_fraud:merchant", 
            "creditcard_fraud:category", 
            "creditcard_fraud:gender", 
            "creditcard_fraud:job", 
            "creditcard_fraud:trans_year", 
            "creditcard_fraud:trans_month", 
            "creditcard_fraud:city_pop"
        ]
        
        # Create entity dataframe for feature retrieval
        entity_dict = {"transaction_id": prediction_request.transaction_id}
        
        # Add event_timestamp if provided
        # if prediction_request.event_timestamp:
        #     entity_dict["event_timestamp"] = str(pd.to_datetime(prediction_request.event_timestamp))
            
        entity_rows = [entity_dict]
        
        # Fetch online features
        feature_data = store.get_online_features(
            features=features, 
            entity_rows=entity_rows
        ).to_dict()

        logging.info(f"Fetched feature data: {feature_data}")

        # Explicitly remove 'transaction_id' if it appears in feature_data
        if 'transaction_id' in feature_data:
            del feature_data['transaction_id']


        # Check for missing features
        # missing_features = [f for f in features if f not in feature_data]
        # if missing_features:
        #     logging.error(f"Missing features: {missing_features}")
        #     raise CreditCardException(e, sys)
        
        # Convert features to array for model input
        # feature_array = np.array([[feature_data[key][0] for key in feature_data]])

        # Convert features to pandas DataFrame for model input
        feature_df = pd.DataFrame(feature_data)
        
        # Preprocess features
        preprocessed_features = preprocessor.transform(feature_df)
       

        if model is None:
            raise CreditCardException("Model is not loaded", sys)

        # Get prediction
        if hasattr(model, "predict_proba"):
            fraud_probability = model.predict_proba(preprocessed_features)[:, 1][0]
        else:
            fraud_label = model.predict(preprocessed_features)[0]
        
        # Determine fraud label
        # fraud_label = 1 if fraud_probability >= 0.5 else 0
        
        # Update MongoDB record with prediction if needed
        # collection.update_one(
        #     {"trans_num": prediction_request.transaction_id},
        #     {"$set": {
        #         "fraud_probability": float(fraud_probability),
        #         "is_fraud": int(fraud_label)
        #     }}
        # )

        logging.info(f"Transaction ID: {prediction_request.transaction_id}, Fraud Label: {fraud_label}")

                
        return {
            "transaction_id": prediction_request.transaction_id,
            # "fraud_probability": float(fraud_probability),
            "fraud_label": int(fraud_label)
        }
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        raise CreditCardException(e, sys)

if __name__ == "__main__":
    app_run(app, host="0.0.0.0", port=8000)