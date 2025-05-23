from datetime import datetime
import os 
from src.constants import *

class TrainingPipelineConfig:
    def __init__(self, timestamp=datetime.now()):
        # Assign timestamp only if it's not provided
        # if timestamp is None:
        #     timestamp = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")        

        timestamp = timestamp.strftime("%m_%d_%Y_%H_%M_%S")        
        self.pipeline_name=PIPELINE_NAME
        self.artifact_name=ARTIFACT_DIR_NAME
        self.artifact_dir=os.path.join(self.artifact_name, timestamp)
        # self.model_dir=os.path.join("final_model")
        self.timestamp: str=timestamp

class DataIngestionConfig:
    def __init__(self, training_pipeline_config:TrainingPipelineConfig):
        self.data_ingestion_dir:str = os.path.join(training_pipeline_config.artifact_dir, DATA_INGESTION_DIR_NAME)

        self.feature_store_file_path: str = os.path.join(self.data_ingestion_dir, DATA_INGESTION_FEATURE_STORE_DIR, FILE_NAME)

        self.training_file_path: str = os.path.join(self.data_ingestion_dir, DATA_INGESTION_INGESTED_DATA_DIR, TRAIN_FILE_NAME)

        self.testing_file_path: str = os.path.join(self.data_ingestion_dir, DATA_INGESTION_INGESTED_DATA_DIR, TEST_FILE_NAME)

        self.train_test_split_ratio: float = DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO

        self.collection_name: str = DATA_INGESTION_COLLECTION_NAME

        self.datbase_name: str = DATA_INGESTION_DATABASE_NAME
    
class DataValidationConfig:
    def __init__(self, training_pipeline_config:TrainingPipelineConfig):
        self.data_validation_dir: str = os.path.join(training_pipeline_config.artifact_dir, DATA_VALIDATION_DIR_NAME)

        self.valid_data_dir: str = os.path.join(self.data_validation_dir, DATA_VALIDATION_VALID_DIR_NAME)

        self.invalid_data_dir: str = os.path.join(self.data_validation_dir, DATA_VALIDATION_INVALID_DIR_NAME)

        self.valid_train_file_path: str = os.path.join(self.valid_data_dir, TRAIN_FILE_NAME)

        self.valid_test_file_path: str = os.path.join(self.valid_data_dir, TEST_FILE_NAME)

        self.invalid_train_file_path: str = os.path.join(self.invalid_data_dir, TRAIN_FILE_NAME)

        self.invalid_test_file_path: str = os.path.join(self.invalid_data_dir, TEST_FILE_NAME)
        
        self.drift_report_file_path: str = os.path.join(self.data_validation_dir, DATA_VALIDATION_DRIFT_REPORT_DIR_NAME, DATA_VALIDATION_DRIFT_REPORT_FILE_NAME)

class DataTransformationConfig:
    def __init__(self, training_pipeline_config:TrainingPipelineConfig):
        self.data_transformation_dir: str = os.path.join(training_pipeline_config.artifact_dir, DATA_TRANSFORMATION_DIR_NAME)

        self.transformed_train_file_path: str = os.path.join( self.data_transformation_dir,DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
        DATA_TRANSFORMATION_TRAIN_FILE_NAME)

        self.transformed_test_file_path: str = os.path.join(self.data_transformation_dir,  DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
        DATA_TRANSFORMATION_TEST_FILE_NAME)

        self.transformed_object_file_path: str = os.path.join(self.data_transformation_dir, DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR, PREPROCESSING_OBJECT_FILE_NAME)

class ModelTrainingConfig:
    def __init__(self, training_pipeline_config:TrainingPipelineConfig):
        self.model_trainer_dir: str = os.path.join(training_pipeline_config.artifact_dir, MODEL_TRAINER_DIR_NAME)

        self.trained_model_file_path: str = os.path.join(self.model_trainer_dir, MODEL_TRAINER_TRAINED_MODEL_DIR, MODEL_TRAINER_TRAINED_MODEL_NAME)
        
        self.expected_accuracy: float = MODEL_TRAINER_EXPECTED_SCORE

        self.overfitting_underfitting_threshold = MODEL_TRAINER_OVER_FITTING_UNDER_FITTING_THRESHOLD
    