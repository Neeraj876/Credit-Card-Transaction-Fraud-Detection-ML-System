import os
import sys
import logging
import numpy as np
import pandas as pd
import yaml
import mlflow

from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from src.entity.config_entity import DataValidationConfig
from src.exception.exception import CreditCardException
from src.logging.logger import logging
from src.utils.main_utils.utils import read_yaml_file, write_yaml_file
from src.constants import *

from scipy.stats import ks_2samp
from evidently.report import Report
from evidently.metrics import DataDriftTable

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
            # Extract column information from schema
            schema_columns = self._schema_config.get("columns", {})
            
            number_of_columns=len(schema_columns)
            logging.info(f"Required number of columns: {number_of_columns}")
            logging.info(f"Data frame has columns: {len(dataframe.columns)}")
            
            return len(dataframe.columns) == number_of_columns

        except Exception as e:
            logging.error(f"Error in column validation: {str(e)}")
            raise CreditCardException(e, sys)
    
    def validate_column_names(self, dataframe: pd.DataFrame) -> bool:
        """
        Validates that all required column names from schema are present in dataframe.
        Handles schema structure where columns is a list of single-key dictionaries.
        """
        try:
            # Extract column names from schema correctly
            schema_columns_list = self._schema_config.get("columns", [])
            schema_columns = []
            
            # Extract column names from the list of dictionaries
            for col_dict in schema_columns_list:
                if isinstance(col_dict, dict):
                    schema_columns.extend(col_dict.keys())
            
            df_columns = dataframe.columns.tolist()
            
            # Check if all required columns exist (ignoring order)
            missing_columns = [col for col in schema_columns if col not in df_columns]
            
            if missing_columns:
                logging.error(f"Missing columns: {missing_columns}")
                return False
            
            # Log column order mismatch as warning but don't fail validation
            if sorted(schema_columns) != sorted(df_columns):
                logging.warning(f"Column mismatch or order difference. Schema: {schema_columns}, DataFrame: {df_columns}")
            
            return True
                
        except Exception as e:
            logging.error(f"Error in column name validation: {str(e)}")
            raise CreditCardException(e, sys)
            
        # def validate_numerical_columns(self, df: pd.DataFrame) -> bool:
    #     """
    #     Checks if all required numerical columns exist in the dataset.
        
    #     Args:
    #         df (pd.DataFrame): Input dataframe.
        
    #     Returns:
    #         bool: True if all numeric columns exist, else False.
    #     """
    #     try:
    #         # Get numerical columns from schema
    #         numerical_columns = self._schema_config.get("numerical_columns", [])
            
    #         # Get actual numerical columns from dataframe
    #         actual_numerical_columns = df.select_dtypes(include=['number']).columns.tolist()
            
    #         logging.info(f"Required numerical columns: {numerical_columns}")
    #         logging.info(f"Data frame has numerical columns: {actual_numerical_columns}")
            
    #         # Check if all required numerical columns are present
    #         missing_columns = set(numerical_columns) - set(actual_numerical_columns)
    #         if missing_columns:
    #             logging.error(f"Missing numerical columns: {missing_columns}")
    #             return False
                
    #         return True

    #     except Exception as e:
    #         logging.error(f"Error in numerical column validation: {str(e)}")
    #         raise CreditCardException(e, sys)

    # def validate_categorical_columns(self, df: pd.DataFrame) -> bool:
    #     """
    #     Checks if all required categorical columns exist in the dataset.
        
    #     Args:
    #         df (pd.DataFrame): Input dataframe.
        
    #     Returns:
    #         bool: True if all categorical columns exist, else False.
    #     """
    #     try:
    #         # Get categorical columns from schema
    #         categorical_columns = self._schema_config.get("categorical_columns", [])
            
    #         # Get actual categorical columns from dataframe
    #         actual_categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
            
    #         logging.info(f"Required categorical columns: {categorical_columns}")
    #         logging.info(f"Data frame has categorical columns: {actual_categorical_columns}")
            
    #         # Check if all required categorical columns are present
    #         missing_columns = set(categorical_columns) - set(actual_categorical_columns)
    #         if missing_columns:
    #             logging.error(f"Missing categorical columns: {missing_columns}")
    #             return False
                
    #         return True

    #     except Exception as e:
    #         logging.error(f"Error in categorical column validation: {str(e)}")
    #         raise CreditCardException(e, sys)

    def detect_dataset_drift(self, base_df: pd.DataFrame, current_df: pd.DataFrame, phase="training", threshold=0.05) -> bool:
        """
        Checks for data drift between reference (training) and current (production) dataset using KS test and Evidently AI.
        
        Args:
            base_df (pd.DataFrame): Reference dataset (training or previous production data).
            current_df (pd.DataFrame): New dataset (test set or real-time data).
            phase (str): Phase of data validation ("training" or "production").
            threshold (float): P-value threshold for drift detection.
        
        Returns:
            bool: True if no significant drift is detected, False otherwise.
        """
        try:
            logging.info(f"Checking for data drift during {phase}.")
            
            report = {}
            status = True

            for column in base_df.columns:
                # Skip non-numeric columns for KS test
                if not np.issubdtype(base_df[column].dtype, np.number):
                    logging.info(f"Skipping drift detection for non-numeric column: {column}")
                    continue
                    
                d1 = base_df[column]
                d2 = current_df[column]
                
                # Remove NaN values before KS test
                d1 = d1.dropna()
                d2 = d2.dropna()
                
                # Skip if not enough data
                if len(d1) < 5 or len(d2) < 5:
                    logging.warning(f"Not enough data for KS test on column {column}. Skipping.")
                    continue
                
                ks_stat, p_value = ks_2samp(d1, d2)

                drift_detected = p_value < threshold
                if drift_detected:
                    status = False  # If any feature drifts, set status to False

                report[column] = {
                    "p_value": float(p_value),
                    "drift_detected": drift_detected
                }

                # Log KS Test results in MLflow
                mlflow.log_metric(f"{phase}_ks_test_p_value_{column}", p_value)

            # Create directory for drift report
            drift_report_file_path = self.data_validation_config.drift_report_file_path
            dir_path = os.path.dirname(drift_report_file_path)
            os.makedirs(dir_path, exist_ok=True)
            
            # Save drift report
            write_yaml_file(file_path=drift_report_file_path, content=report)

            # Save drift report to MLflow
            drift_report_file = f"{phase}_drift_report.yaml"
            with open(drift_report_file, "w") as f:
                yaml.dump(report, f)
            mlflow.log_artifact(drift_report_file, artifact_path="drift_reports")

            logging.info(f"Generating Evidently AI Data Drift Report for {phase} data...")

            # # Generate Evidently AI Data Drift Report
            # drift_report = Report(metrics=[DataDriftTable()])
            # drift_report.run(reference_data=base_df, current_data=current_df)

            # # Define HTML file path in the same directory
            # drift_report_html_path = os.path.join(dir_path, f"{phase}_evidently_drift_report.html")

            # # Save the report
            # drift_report.save_html(drift_report_html_path)
            # logging.info(f"Drift report saved at: {drift_report_html_path}")

            # # Log to mlflow
            # mlflow.log_artifact(drift_report_html_path, artifact_path="drift_reports")

            # # Log dataset drift metric
            # drift_score = drift_report.as_dict()["metrics"][0]["result"]["dataset_drift"]
            # mlflow.log_metric(f"{phase}_dataset_drift", drift_score)
            # logging.info(f"Dataset drift score ({phase}): {drift_score}")

            return status

        except Exception as e:
            logging.error(f"Error in detecting dataset drift during {phase}: {str(e)}")
            raise CreditCardException(e, sys)
            
    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            train_file_path = self.data_ingestion_artifact.trained_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            # Read the data from train and test
            train_dataframe = DataValidation.read_data(train_file_path)
            test_dataframe = DataValidation.read_data(test_file_path)

            validation_errors = []

            # Validate number of columns in train dataframe
            status = self.validate_number_of_columns(dataframe=train_dataframe)
            if not status:
                error_message = "Train dataframe does not contain the expected number of columns."
                validation_errors.append(error_message)
                logging.error(error_message)

            # Validate number of columns in test dataframe
            status = self.validate_number_of_columns(dataframe=test_dataframe)
            if not status:
                error_message = "Test dataframe does not contain the expected number of columns."
                validation_errors.append(error_message)   
                logging.error(error_message)
                
            # Validate column names in train dataframe
            status = self.validate_column_names(dataframe=train_dataframe)
            if not status:
                error_message = "Train dataframe does not contain all required columns."
                validation_errors.append(error_message)
                logging.error(error_message)

            # Validate column names in test dataframe
            status = self.validate_column_names(dataframe=test_dataframe)
            if not status:
                error_message = "Test dataframe does not contain all required columns."
                validation_errors.append(error_message)   
                logging.error(error_message)

            # If any validation errors, fail early
            if validation_errors:
                # Create directories for invalid files
                invalid_train_dir = os.path.dirname(self.data_validation_config.invalid_train_file_path)
                invalid_test_dir = os.path.dirname(self.data_validation_config.invalid_test_file_path)
                os.makedirs(invalid_train_dir, exist_ok=True)
                os.makedirs(invalid_test_dir, exist_ok=True)
                
                # Save invalid data
                train_dataframe.to_csv(self.data_validation_config.invalid_train_file_path, index=False)
                test_dataframe.to_csv(self.data_validation_config.invalid_test_file_path, index=False)
                
                logging.warning(f"Validation failed with errors: {validation_errors}")
                return DataValidationArtifact(
                    valid_train_file_path=None,
                    valid_test_file_path=None,
                    invalid_train_file_path=self.data_validation_config.invalid_train_file_path,
                    invalid_test_file_path=self.data_validation_config.invalid_test_file_path,
                    validation_status=False,
                    drift_report_file_path=self.data_validation_config.drift_report_file_path
                )

            # Check data drift - only if basic validations passed
            drift_status = self.detect_dataset_drift(base_df=train_dataframe, current_df=test_dataframe, phase="training")

            # Create directories for valid files
            valid_train_dir = os.path.dirname(self.data_validation_config.valid_train_file_path)
            valid_test_dir = os.path.dirname(self.data_validation_config.valid_test_file_path)
            os.makedirs(valid_train_dir, exist_ok=True)
            os.makedirs(valid_test_dir, exist_ok=True)

            # Always save files to valid directory, even if drift is detected
            # This ensures downstream components have files to work with
            train_dataframe.to_csv(self.data_validation_config.valid_train_file_path, index=False)
            test_dataframe.to_csv(self.data_validation_config.valid_test_file_path, index=False)

            if drift_status:
                logging.info("Validation successful, no significant drift detected.")
                return DataValidationArtifact(
                    valid_train_file_path=self.data_validation_config.valid_train_file_path,
                    valid_test_file_path=self.data_validation_config.valid_test_file_path,
                    invalid_train_file_path=None,
                    invalid_test_file_path=None,
                    validation_status=True,
                    drift_report_file_path=self.data_validation_config.drift_report_file_path,
                )
            else:
                # Also save files to invalid directory for reference
                invalid_train_dir = os.path.dirname(self.data_validation_config.invalid_train_file_path)
                invalid_test_dir = os.path.dirname(self.data_validation_config.invalid_test_file_path)
                os.makedirs(invalid_train_dir, exist_ok=True)
                os.makedirs(invalid_test_dir, exist_ok=True)
                
                train_dataframe.to_csv(self.data_validation_config.invalid_train_file_path, index=False)
                test_dataframe.to_csv(self.data_validation_config.invalid_test_file_path, index=False)
                
                logging.warning("Data drift detected, but continuing with processing. Data stored in both valid and invalid directories.")
                return DataValidationArtifact(
                    valid_train_file_path=self.data_validation_config.valid_train_file_path,
                    valid_test_file_path=self.data_validation_config.valid_test_file_path,
                    invalid_train_file_path=self.data_validation_config.invalid_train_file_path,
                    invalid_test_file_path=self.data_validation_config.invalid_test_file_path,
                    validation_status=False,
                    drift_report_file_path=self.data_validation_config.drift_report_file_path
                )

        except Exception as e:
            raise CreditCardException(e, sys)