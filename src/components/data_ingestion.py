import os
import sys
import numpy as np
import pandas as pd
import pymongo
from sklearn.model_selection import train_test_split
from feast import FeatureStore


from src.logging.logger import logging
from src.exception.exception import CreditCardException
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact
from src.constants import *

# from src.components.data_transformation import DataTransformation

# from src.components.model_trainer import ModelTrainer

from dotenv import load_dotenv
load_dotenv

MONGO_DB_URL=os.getenv("MONGO_DB_URL")

class DataIngestion:
    def __init__(self, data_ingestion_config:DataIngestionConfig):
        try:
            self.data_ingestion_config=data_ingestion_config
        except Exception as e:
            raise CreditCardException(e, sys)
        
        def advanced_historical_retrieval():
            """
            More flexible historical feature retrieval
            """
            feature_store = FeatureStore(repo_path=".")
            
            # Direct query to PostgreSQL source
            # This uses the source defined in your PostgreSQL Source configuration
            historical_features = feature_store.get_historical_features(
                entity_df=pd.DataFrame({
                    "transaction_id": [],  # Empty DataFrame allows full source retrieval
                    "event_timestamp": []
                }),
                feature_refs=[
                    "fraud_features:cc_num",
                    "fraud_features:amt",
                    "fraud_features:is_fraud"
                ]
            )
            
            return historical_features.to_df()
                
    # def export_collection_as_dataframe(self):
    #     """
    #     Read data from mongodb
    #     """
    #     try:
    #         database_name=self.data_ingestion_config.datbase_name
    #         collection_name=self.data_ingestion_config.collection_name
    #         self.mongo_client=pymongo.MongoClient(MONGO_DB_URL)
    #         collection=self.mongo_client[database_name][collection_name]

    #         print(collection.find().limit(1))

    #         df=pd.DataFrame(list(collection.find))
    #         if "_id" in df.columns.to_list():
    #             df=df.drop(columns=["id"], axis=1)

    #         df.replace({'na': np.nan}, inplace=True)
    #         return df
        
    #     except Exception as e:
    #         raise CreditCardException(e, sys)
        
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
            # Step 3: Data cleaning (drop unwanted columns)
            logging.info("Dropping unnecessary columns if present.")
            df.drop(columns=['Unnamed: 0'], errors='ignore', inplace=True)
            df.reset_index(drop=True, inplace=True)

            # Step 4: Split dataset into train and test sets
            logging.info("Splitting dataset into train and test sets.")
            train_data, test_data = train_test_split(df, test_size=DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO, random_state=RANDOM_STATE)

            # Step 5: Create necessary directories
            logging.info("Creating directories for saving datasets.")
            os.makedirs(self.data_ingestion_config.data_ingestion_dir, exist_ok=True)

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
    pass

        