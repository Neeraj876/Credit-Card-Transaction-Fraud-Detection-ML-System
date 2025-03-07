import os
import sys
import logging
import numpy as np
import pandas as pd
import yaml

from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from src.entity.config_entity import DataValidationConfig
from src.exception.exception import CreditCardException
from src.logging.logger import logging
from utils.main_utils.utils import read_yaml_file, write_yaml_file
from src.constants import *

from scipy.stats import ks_2samp

class DataValidation:
    def __init__(self, data_ingestion_artifact:DataIngestionArtifact, 
    data_validation_config:DataValidationConfig
    ):
        try:
            self.data_ingestion_artifact=data_ingestion_artifact
            self.data_validation_config=data_validation_config
            self._schema_config=read_yaml_file(SCHEMA_FILE_PATH)

        except Exception as e:
            raise CreditCardException(e, sys)
        
    @staticmethod
    def read_data(file_path)->pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise CreditCardException(e, sys)
        
    def validate_number_of_columns(self,dataframe:pd.DataFrame)->bool:
        try:
            number_of_columns=len(self._schema_config)
            logging.info(f"Required number of columns:{number_of_columns}")
            logging.info(f"Data frame has columns:{len(dataframe.columns)}")
            if len(dataframe.columns)==number_of_columns:
                return True
            return False

        except Exception as e:
            raise CreditCardException(e, sys)

    def numerical_exists(self, df: pd.DataFrame) -> bool:
        """
        Checks if all required numerical columns exist in the dataset.
        
        Args:
            df (pd.DataFrame): Input dataframe.
        
        Returns:
            bool: True if all numeric columns exist, else raises exception.
        """
        try:
            expected_columns = df.select_dtypes(exclude='object')

            if len(self._schema_config.columns) != len(df.columns.to_list()):
                logging.error("Required and Expected columns are not equal")
            
            if len(self._schema_config.numeric_columns) == expected_columns.shape[1]:
                for col in self._schema_config.numeric_columns:
                    if col not in expected_columns:
                        logging.error(f"Missing column: {col}")
                        return False
                return True
            else:
                logging.error("Mismatch in expected numerical columns count.")
        except Exception as e:
            raise CreditCardException(e, sys)

    def categorical_exists(self, df: pd.DataFrame) -> bool:
        """
        Checks if all required categorical columns exist in the dataset.
        
        Args:
            df (pd.DataFrame): Input dataframe.
        
        Returns:
            bool: True if all categorical columns exist, else raises exception.
        """
        try:
            expected_columns = df.select_dtypes(include='object')

            if len(self._schema_config.columns) != len(df.columns.to_list()):
                logging.error("Required and Expected columns are not equal")
            
            if len(self._schema_config.categorical_columns) == expected_columns.shape[1]:
                for col in self._config.categorical_columns:
                    if col not in expected_columns:
                        logging.error(f"Missing column: {col}")
                        return False
                return True
            else:
               logging.error("Mismatch in expected categorical columns count.")
        except Exception as e:
            raise CreditCardException(e, sys)
        
    def detect_dataset_drift(self, base_df: pd.DataFrame, current_df: pd.DataFrame, threshold=0.05) -> tuple:
        """
        Checks for data drift using the KS test.
        
        Args:
            base_df (pd.DataFrame): Reference dataset.
            current_df (pd.DataFrame): New dataset for comparison.
            threshold (float): P-value threshold for drift detection.
        
        Returns:
            tuple: Drift report (dict) and status (bool).
        """
        try:
            logging.info("Checking for data drift.")
            report = {}
            status = True

            for column in base_df.columns:
                d1 = base_df[column]
                d2 = current_df[column]
                is_same_dist = ks_2samp(d1, d2)
                if is_same_dist.pvalue >= threshold:
                    is_found=False
                else:
                    is_found=True
                    status=False
                report.update({column:{
                    "p_value":float(is_same_dist.pvalue),
                    "drift_status":is_found
                    
                    }})
            drift_report_file_path = self.data_validation_config.drift_report_file_path

            #Create directory
            dir_path = os.path.dirname(drift_report_file_path)
            os.makedirs(dir_path,exist_ok=True)
            write_yaml_file(file_path=drift_report_file_path,content=report)

        except Exception as e:
            raise CreditCardException(e, sys)
        
    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            train_file_path=self.data_ingestion_artifact.trained_file_path
            test_file_path=self.data_ingestion_artifact.test_file_path

            # Read the data from train and test
            train_dataframe=DataValidation.read_data(train_file_path)
            test_dataframe=DataValidation.read_data(test_file_path)

            # Validate number of columns in train dataframe and test dataframe
            status=self.validate_number_of_columns(dataframe=train_dataframe)
            if not status:
                error_message=f"Train dataframe does not contain all columns.\n"
            status = self.validate_number_of_columns(dataframe=test_dataframe)
            if not status:
                error_message=f"Test dataframe does not contain all columns.\n"   

            logging.info("Checking for numerical Columns for train and test data")
            self.numerical_exists(train_dataframe)
            self.numerical_exists(test_dataframe)

            logging.info("Checking for Categorical Columns for train and test data")
            self.categorical_exists(train_dataframe)
            self.categorical_exists(test_dataframe)

            # Check data drift
            status=self.detect_dataset_drift(base_df=train_dataframe,current_df=test_dataframe)
            dir_path=os.path.dirname(self.data_validation_config.valid_train_file_path)
            os.makedirs(dir_path,exist_ok=True)
            dir_path=os.path.dirname(self.data_validation_config.valid_test_file_path)
            os.makedirs(dir_path,exist_ok=True)

            if status:
                train_dataframe.to_csv(self.data_validation_config.valid_train_file_path, index=False)
                test_dataframe.to_csv(self.data_validation_config.valid_test_file_path, index=False)
                logging.info("Validation successful, data stored in valid directory.")
                return DataValidationArtifact(
                    valid_train_path=self.data_validation_config.valid_train_file_path,
                    valid_test_path=self.data_validation_config.valid_test_file_path,
                    invalid_train_path=None,
                    invalid_test_path=None,
                    validation_status=status
                )
            else:
                train_dataframe.to_csv(self.data_validation_config.invalid_train_file_path, index=False)
                test_dataframe.to_csv(self.data_validation_config.invalid_test_file_path, index=False)
                logging.warning("Validation failed, data stored in invalid directory.")
                return DataValidationArtifact(
                    valid_train_path=None,
                    valid_test_path=None,
                    invalid_train_path=self.data_validation_config.invalid_train_file_path,
                    invalid_test_path=self.data_validation_config.invalid_test_file_path,
                    validation_status=status
                )

        except Exception as e:
            raise CreditCardException(e, sys)