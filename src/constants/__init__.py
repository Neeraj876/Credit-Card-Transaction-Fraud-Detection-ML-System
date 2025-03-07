import os
import numpy as np


"""
defining common constant variable for training pipeline
"""
TARGET_COLUMN = "is_fraud"
PIPELINE_NAME: str = "CreditCard"
ARTIFACT_DIR_NAME: str = "Artifacts"
FILE_NAME: str = "raw.csv"
PREPROCESSING_OBJECT_FILE_NAME = "preprocessing.pkl"

TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"

SCHEMA_FILE_PATH = os.path.join("data_schema", "schema.yaml")

FINAL_MODEL_DIR_NAME="final_model"
MODEL_FILE_NAME = "model.pkl"

FINAL_PREPROCESSOR_PATH = os.path.join(os.getcwd(),"final_model","preprocessor.pkl")

"""
data ingestion
"""
DATA_INGESTION_DIR_NAME = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DATA_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2
RANDOM_STATE = 42
DATA_INGESTION_COLLECTION_NAME: str = "creditcardData"
DATA_INGESTION_DATABASE_NAME: str = "FRAUD"

"""
data validation
"""
DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_VALID_DIR_NAME: str = "validated"
DATA_VALIDATION_INVALID_DIR_NAME: str = "invalid"
DATA_VALIDATION_DRIFT_REPORT_DIR_NAME: str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "report.yaml"

"""
data transformation
"""

DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"

# knn imputer to replace nan values
DATA_TRANSFORMATION_IMPUTER_PARAMS: dict = {
    "missing_values": np.nan,
    "n_neighbors": 3,
    "weights": "uniform",
}

DATA_TRANSFORMATION_TRAIN_FILE_NAME: str = "train.npy"

DATA_TRANSFORMATION_TEST_FILE_NAME: str = "test.npy"



"""
Model trainer
"""

MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_TRAINER_TRAINED_MODEL_NAME: str = "model.pkl"
MODEL_TRAINER_EXPECTED_SCORE: float = 0.6
MODEL_TRAINER_OVER_FITTING_UNDER_FITTING_THRESHOLD: float = 0.05


