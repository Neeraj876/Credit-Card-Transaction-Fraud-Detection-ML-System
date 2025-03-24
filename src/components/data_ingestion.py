import os
import sys
import numpy as np
import pandas as pd
import pymongo
from sklearn.model_selection import train_test_split

from feast import FeatureStore
import pandas as pd
import psycopg2

from src.logging.logger import logging
from src.exception.exception import CreditCardException
from src.entity.config_entity import DataIngestionConfig, TrainingPipelineConfig, DataValidationConfig, DataTransformationConfig, ModelTrainingConfig
from src.entity.artifact_entity import DataIngestionArtifact
from src.constants import *

from src.components.data_validation import DataValidation

from src.components.data_transformation import DataTransformation

from src.components.model_training import ModelTrainer

from dotenv import load_dotenv
load_dotenv

MONGO_DB_URL=os.getenv("MONGO_DB_URL")

class DataIngestion:
    def __init__(self, data_ingestion_config:DataIngestionConfig):
        try:
            self.data_ingestion_config=data_ingestion_config
        except Exception as e:
            raise CreditCardException(e, sys)
        
    def feature_retrieval(self):
            """
            Retrieve historical features from feast offline store postgres
            """
            # Initialize Feature Store
            FEATURE_REPO_PATH = os.getenv("FEATURE_REPO_PATH")
            store = FeatureStore(repo_path=FEATURE_REPO_PATH)

            # list of features to fetch
            features = [
                "creditcard_fraud:cc_num", 
                "creditcard_fraud:amt", 
                "creditcard_fraud:merchant", 
                "creditcard_fraud:category", 
                "creditcard_fraud:gender", 
                "creditcard_fraud:job", 
                "creditcard_fraud:trans_year", 
                "creditcard_fraud:trans_month", 
                "creditcard_fraud:is_fraud", 
                "creditcard_fraud:city_pop"
            ]

            # Entity DataFrame (PostgreSQL Query)
            df = store.get_historical_features(
                features=features, 
                entity_df="SELECT transaction_id, event_timestamp FROM transactions_raw"
            ).to_df()

            df.replace({"na":np.nan}, inplace=True)

            df.to_csv("/mnt/d/real_time_streaming/Notebooks/Dataset/training_features.csv", index=False, header=True)
            
            return df
        
    def export_data_into_feature_store(self, df: pd.DataFrame):
        try:
            feature_store_file_path=self.data_ingestion_config.feature_store_file_path

            # Creating folder
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok=True)

            # Save raw data
            df.to_csv(feature_store_file_path,index=False,header=True)
            logging.info("Raw data saved successfully")

            return df

        except Exception as e:
            raise CreditCardException(e, sys)

    def split_data_as_train_test(self, df: pd.DataFrame):
        try:

            # Step 4: Split dataset into train and test sets
            logging.info("Splitting dataset into train and test sets.")
            train_data, test_data = train_test_split(df, test_size=self.data_ingestion_config.train_test_split_ratio, random_state=RANDOM_STATE)

            logging.info("Performed train test split on the dataframe")

            # Step 5: Create necessary directories
            logging.info("Creating directories for saving datasets.")
            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            
            os.makedirs(dir_path, exist_ok=True)

            # Step 6: Save datasets
            logging.info("Saving train and test datasets.")
            
            train_data.to_csv(self.data_ingestion_config.training_file_path, index=False, header=True)

            test_data.to_csv(self.data_ingestion_config.testing_file_path, index=False, header=True)

            logging.info(f"Exported train and test file path.")

        except Exception as e:
            raise CreditCardException(e, sys)
        
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            # df=self.export_collection_as_dataframe()
            # print(df.shape)

            df=self.feature_retrieval()

            df=self.export_data_into_feature_store(df)

            self.split_data_as_train_test(df)

            dataingestionartifact=DataIngestionArtifact(
                trained_file_path=self.data_ingestion_config.training_file_path, 
                test_file_path=self.data_ingestion_config.testing_file_path
            )

            logging.info("Data ingestion completed successfully.")
        
            return dataingestionartifact  
          
        except Exception as e:
            raise CreditCardException(e, sys)
        
if __name__=="__main__":
    training_pipeline_config=TrainingPipelineConfig()

    data_ingestion_config=DataIngestionConfig(training_pipeline_config)
    data_ingestion=DataIngestion(data_ingestion_config)
    data_ingestion_artifact=data_ingestion.initiate_data_ingestion()

    # data_validation_config=DataValidationConfig(training_pipeline_config)
    # data_validation=DataValidation(data_ingestion_artifact, data_validation_config)
    # data_validation_artifact=data_validation.initiate_data_validation()

    # data_transformation_config=DataTransformationConfig(training_pipeline_config)
    # data_transformation=DataTransformation(data_validation_artifact, data_transformation_config)
    # data_transformation_artifact=data_transformation.initiate_data_transformation()

    # model_trainer_config=ModelTrainingConfig(training_pipeline_config)
    # model_trainer=ModelTrainer(data_transformation_artifact, model_trainer_config)
    # model_trainer_artifact=model_trainer.initiate_model_trainer()

        