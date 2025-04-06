import sys
import os
import mlflow
import logging
import json
import numpy as np
from mlflow.tracking import MlflowClient
from feast import FeatureStore
from pydantic import BaseModel
import time
from datetime import datetime, timezone
from typing import Union

from src.exception.exception import CreditCardException
# from src.logging.logger import logger
from src.logging.otel_logger import logger, tracer  # Import logger (initializes OpenTelemetry)

from src.pipeline.training_pipeline import TrainingPipeline

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Request, Depends
from uvicorn import run as app_run
from contextlib import asynccontextmanager
from fastapi.responses import Response, JSONResponse
from starlette.responses import RedirectResponse
import pandas as pd
import pymongo
import certifi

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

# Prometheus Metrics
from prometheus_client import Counter, Gauge
from prometheus_fastapi_instrumentator import Instrumentator

from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

# Certificate for MongoDB connection
ca = certifi.where()

# Prometheus Metrics Configuration
REQUEST_COUNT = Counter('app_requests_total', 'Total app requests')
PREDICTION_COUNT = Counter('prediction_requests_total', 'Total prediction requests')
FRAUD_DETECTION_GAUGE = Gauge('fraud_detection_gauge', 'Fraud detection model performance')

# Structured Logging Formatter
class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'message': record.getMessage(),
            'logger': record.name,
            'module': record.module,
            'line': record.lineno
        }
        return json.dumps(log_record)

# # Attach JsonFormatter to the logger
# handler = logging.StreamHandler()  # or use logging.FileHandler to write logs to a file
# handler.setFormatter(JsonFormatter())
# logging.addHandler(handler)
# Correct way to configure logger:
log = logging.getLogger()  # Get the root logger or use a specific name
handler = logging.StreamHandler()
handler.setFormatter(JsonFormatter())
log.addHandler(handler)  # Call addHandler on the logger instance

# Performance Logging Decorator
def log_performance(func):
    def wrapper(*args, **kwargs):
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span(func.__name__):
            try:
                # Increment request counter
                REQUEST_COUNT.inc()
                
                # Log method entry
                logger.info(json.dumps({
                    "event": f"{func.__name__}_started",
                    "args": str(args),
                    "kwargs": str(kwargs)
                }))
                
                # Execute the function
                result = func(*args, **kwargs)
                
                # Log successful execution
                logger.info(json.dumps({
                    "event": f"{func.__name__}_completed",
                    "status": "success"
                }))
                
                return result
            except Exception as e:
                # Log error with structured logging
                logger.error(json.dumps({
                    "event": f"{func.__name__}_failed",
                    "error": str(e),
                    "error_type": type(e).__name__
                }))
                raise
    return wrapper


# # OpenTelemetry Tracing Setup
# def setup_opentelemetry_tracing():
#     """
#     Configure OpenTelemetry tracing to export spans to the collector
#     """
#      # Check if a TracerProvider is already set up
#     if trace.get_tracer_provider().__class__.__name__ == "TracerProvider":
#         logger.info("TracerProvider already initialized, skipping setup")
#         return
    
#     # Create trace provider
#     trace.set_tracer_provider(TracerProvider())
    
#     # Setup OTLP exporter to send traces to OpenTelemetry Collector
#     otlp_exporter = OTLPSpanExporter(endpoint="http://otel-collector:4317")
#     span_processor = BatchSpanProcessor(otlp_exporter)
#     trace.get_tracer_provider().add_span_processor(span_processor)


# Connect to MongoDB
MONGO_DB_URL = os.getenv("MONGO_DB_URL", )
mongo_client = pymongo.MongoClient(MONGO_DB_URL, tlsCAFile=ca)
db = mongo_client["FRAUD"]
collection = db["creditcardData"]

# Set MLflow tracking URI from environment variable
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_TRACKING_URI

# Global variables for model and preprocessor
model = None
preprocessor = None
store = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, preprocessor, store

    try:
        # # Setup OpenTelemetry tracing
        # setup_opentelemetry_tracing()

        # Initialize Feature Store
        FEATURE_REPO_PATH = os.getenv("FEATURE_REPO_PATH")
        store = FeatureStore(repo_path=FEATURE_REPO_PATH)
        logger.info(f"Feature store initialized with repo path: {FEATURE_REPO_PATH}")


        # Configure MLflow to use EC2-hosted tracking server
        # remote_tracking_uri = "http://ec2-34-207-207-10.compute-1.amazonaws.com:5000"
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        logger.info(f"MLflow Tracking URI set to: {MLFLOW_TRACKING_URI}")

        # Load model and preprocessor once at startup
        client = MlflowClient()

        # Get information about registered models
        try:
            model_names = [m.name for m in client.search_registered_models()]
            logger.info(f"✓ Found registered models: {model_names}")
            
            if 'XGBClassifier' not in model_names or 'feature_preprocessor' not in model_names:
                logger.info("❌ Error: Required models not found in the registry")
                missing = []
                if 'XGBClassifier' not in model_names:
                    missing.append('XGBClassifier')
                if 'feature_preprocessor' not in model_names:
                    missing.append('feature_preprocessor')
                logger.info(f" Missing: {', '.join(missing)}")
                sys.exit(1)
        except Exception as e:
            logger.error(f"❌ Failed to retrieve registered models: {str(e)}")
            sys.exit(1)

        # Get model versions
        model_version = None
        preprocessor_version = None

        try:
            # Get XGBClassifier versions
            model_versions = client.search_model_versions("name='XGBClassifier'")
            if not model_versions:
                logger.info("❌ Error: No versions found for XGBClassifier")
                sys.exit(1)
            

            # Sort by version number (latest first)
            model_versions = sorted(model_versions, key=lambda x: int(x.version), reverse=True)
            model_version = model_versions[0].version
            logger.info(f"✓ Latest XGBClassifier version: {model_version}")
            
            # Get preprocessor versions
            preprocessor_versions = client.search_model_versions("name='feature_preprocessor'")
            if not preprocessor_versions:
                logger.info("❌ Error: No versions found for feature_preprocessor")
                sys.exit(1)
            
            # Sort by version number (latest first)
            preprocessor_versions = sorted(preprocessor_versions, key=lambda x: int(x.version), reverse=True)
            preprocessor_version = preprocessor_versions[0].version
            logger.info(f"✓ Latest feature_preprocessor version: {preprocessor_version}")
        except Exception as e:
            logger.error(f"❌ Failed to retrieve model versions: {str(e)}")
            sys.exit(1)

        # Load preprocessor
        try:
            logger.info("\nLoading feature preprocessor...")
            preprocessor = mlflow.pyfunc.load_model(f"models:/feature_preprocessor/{preprocessor_version}")
            logger.info(f"✓ Successfully loaded preprocessor (version {preprocessor_version})")
            logger.info(f"   Type: {type(preprocessor).__name__}")
        except Exception as e:
            logger.error(f"❌ Failed to load preprocessor: {str(e)}")
            sys.exit(1)
        
        # Load model
        try:
            logger.info("\nLoading XGBClassifier model...")
            model = mlflow.pyfunc.load_model(f"models:/XGBClassifier/{model_version}")
            logger.info(f"✓ Successfully loaded model (version {model_version})")
            logger.info(f"   Type: {type(model).__name__}")
        except Exception as e:
            logger.error(f"❌ Failed to load model: {str(e)}")
            sys.exit(1)

        # Yield control back to the app for async startup tasks
        yield

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        model = None
        preprocessor = None
        yield  # Ensure app can still start even if the model loading fails

# app = FastAPI()
# app = FastAPI(lifespan=lifespan, root_path="/api", openapi_url="/openapi.json", docs_url="/docs",  redoc_url="/redoc")

# Create the trace provider
# trace.set_tracer_provider(TracerProvider())

# Set up the exporter to send traces to OpenTelemetry Collector
# otlp_exporter = OTLPSpanExporter(endpoint="http://otel-collector:4317")
# span_processor = BatchSpanProcessor(otlp_exporter)
# trace.get_tracer_provider().add_span_processor(span_processor)

app = FastAPI(lifespan=lifespan)

# CORS Middleware
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instrument the FastAPI app
FastAPIInstrumentor.instrument_app(app)

# Create a tracer
# tracer = trace.get_tracer(__name__)

# Add Prometheus instrumentation
Instrumentator().instrument(app).expose(app)

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

@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")

@app.get("/train")
@log_performance
async def train_route():
    try:
        train_pipeline = TrainingPipeline()
        train_pipeline.run_pipeline()

        # Log training event
        logger.info(json.dumps({
            "event": "model_training_completed",
            "status": "success"
        }))

        return Response("Training is successful")
    except Exception as e:
        # Log training failure
        logger.error(json.dumps({
            "event": "model_training_failed",
            "error": str(e)
        }))
        raise CreditCardException(e, sys)
    
# Temporary storage for the last transaction_id received from Spark
latest_transaction_id = None  

@app.post("/transaction")
@log_performance
async def create_transaction(transaction: Union[TransactionRequest, SparkTransactionRequest]):
    global latest_transaction_id  # Allow modification of global variable
    tracer = trace.get_tracer(__name__)

    try:
        with tracer.start_as_current_span("create_transaction"):

            # Prometheus metrics tracking
            REQUEST_COUNT.inc()

            insert_result = None  # Avoid uninitialized reference

            # If request is from Spark (contains transaction_id)
            if isinstance(transaction, SparkTransactionRequest):
                latest_transaction_id = transaction.dict().get("transaction_id")
                return {"status": "success", "message": f"Received transaction_id {latest_transaction_id}"}

            # If request is from Streamlit (TransactionRequest)
            if isinstance(transaction, TransactionRequest):

                transaction_dict = transaction.dict()

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
        logger.error(json.dumps({
                "event": "transaction_creation_failed",
                "error": str(e)
            }))
        return {"status": "error", "message": str(e)}

@app.post("/predict")
@log_performance
async def predict(prediction_request: PredictionRequest):
    # try:
    tracer = trace.get_tracer(__name__)
        
    with tracer.start_as_current_span("predict_fraud"):

        #Prometheus metrics and tracing
        PREDICTION_COUNT.inc()

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

        logger.info(f"Fetched feature data: {feature_data}")

        # Explicitly remove 'transaction_id' if it appears in feature_data
        if 'transaction_id' in feature_data:
            del feature_data['transaction_id']

        # Convert features to pandas DataFrame for model input

        feature_df = pd.DataFrame(feature_data)
        
        # Preprocess features
        preprocessed_features = preprocessor.predict(feature_df)
       

        if model is None:
            raise CreditCardException("Model is not loaded", sys)

        # Get prediction
        fraud_label = model.predict(preprocessed_features)[0]
        FRAUD_DETECTION_GAUGE.set(fraud_label)

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

        logger.info(json.dumps({
                "event": "fraud_prediction_completed",
                "transaction_id": prediction_request.transaction_id,
                "fraud_label": fraud_label
            }))
        
        # logger.info(f"Transaction ID: {prediction_request.transaction_id}, Fraud Label: {fraud_label}")
 
        return {
            "transaction_id": prediction_request.transaction_id,
            # "fraud_probability": float(fraud_probability),
            "fraud_label": int(fraud_label)
        }
    # except Exception as e:
    #     logging.error(f"Prediction error: {str(e)}")
    #     raise CreditCardException(e, sys)

if __name__ == "__main__":
    app_run(app, host="0.0.0.0", port=8000)