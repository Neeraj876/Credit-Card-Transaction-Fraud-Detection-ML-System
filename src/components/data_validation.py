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
            schema_columns = self._schema_config.get("columns", {})  # Extract 'columns' section
            
            number_of_columns=len(schema_columns)
            logging.info(f"Required number of columns:{number_of_columns}")
            logging.info(f"Data frame has columns:{len(dataframe.columns)}")
            if len(dataframe.columns)==number_of_columns:
                return True
            return False

        except Exception as e:
            logging.error(f"Error in column validation: {str(e)}")
            raise CreditCardException(e, sys)

    # def numerical_exists(self, df: pd.DataFrame) -> bool:
    #     """
    #     Checks if all required numerical columns exist in the dataset.
        
    #     Args:
    #         df (pd.DataFrame): Input dataframe.
        
    #     Returns:
    #         bool: True if all numeric columns exist, else raises exception.
    #     """
    #     try:
    #         expected_columns = df.select_dtypes(exclude='object')

    #         number_of_columns=len(self._schema_config)
    #         logging.info(f"Required number of columns:{number_of_columns}")
    #         logging.info(f"Data frame has numerical columns:{expected_columns}")
            
    #         if expected_columns==number_of_columns:
    #             return True
    #         return False

    #     except Exception as e:
    #         raise CreditCardException(e, sys)

    # def categorical_exists(self, df: pd.DataFrame) -> bool:
    #     """
    #     Checks if all required categorical columns exist in the dataset.
        
    #     Args:
    #         df (pd.DataFrame): Input dataframe.
        
    #     Returns:
    #         bool: True if all categorical columns exist, else raises exception.
    #     """
    #     try:
    #         expected_columns = df.select_dtypes(include='object')

    #         if len(self._schema_config.columns) != len(df.columns.to_list()):
    #             logging.error("Required and Expected columns are not equal")
            
    #         if len(self._schema_config.categorical_columns) == expected_columns.shape[1]:
    #             for col in self._schema_config.categorical_columns:
    #                 if col not in expected_columns:
    #                     logging.error(f"Missing column: {col}")
    #                     return False
    #             return True
    #         else:
    #            logging.error("Mismatch in expected categorical columns count.")
    #     except Exception as e:
    #         raise CreditCardException(e, sys)
        
    def detect_dataset_drift(self, base_df: pd.DataFrame, current_df: pd.DataFrame, phase="training", threshold=0.05) -> tuple:
        """
        Checks for data drift between reference (training) and current (production) dataset using KS test and Evidently AI.
        
        Args:
            base_df (pd.DataFrame): Reference dataset (training or previous production data).
            current_df (pd.DataFrame): New dataset (test set or real-time data).
            phase (str): Phase of data validation ("training" or "production").
            threshold (float): P-value threshold for drift detection.
        
        Returns:
            tuple: Drift report (dict) and status (bool).
        """
        try:
            logging.info(f"Checking for data drift during {phase}.")
            
            report = {}
            status = True

            for column in base_df.columns:
                d1 = base_df[column]
                d2 = current_df[column]
                ks_stat, p_value = ks_2samp(d1, d2)

                drift_detected = p_value < threshold
                if drift_detected:
                    status = False  # If any feature drifts, set status to False

                report[column] = {
                    "p_value": float(p_value),
                    "drift_detected": drift_detected
                }

                # Log KS Test results in MLflow
                # mlflow.log_metric(f"{phase}_ks_test_p_value_{column}", p_value)

            drift_report_file_path = self.data_validation_config.drift_report_file_path

            # Create directory
            dir_path = os.path.dirname(drift_report_file_path)
            os.makedirs(dir_path,exist_ok=True)
            write_yaml_file(file_path=drift_report_file_path,content=report)

            # # Log KS Test results in MLflow
            # mlflow.log_metric(f"{phase}_ks_test_p_value_{column}", p_value)

            # Save drift report to MLflow
            drift_report_file = f"{phase}_drift_report.yaml"
            with open(drift_report_file, "w") as f:
                yaml.dump(report, f)
            # mlflow.log_artifact(drift_report_file, artifact_path="drift_reports")

            logging.info(f"Generating Evidently AI Data Drift Report for {phase} data...")

            # Generate Evidently AI Data Drift Report
            drift_report = Report(metrics=[DataDriftTable()])
            drift_report.run(reference_data=base_df, current_data=current_df)

            # Save and log the Evidently drift report
            dir_path = os.path.dirname(drift_report_file_path)
            os.makedirs(dir_path, exist_ok=True)

            # Define HTML file path in the same directory
            drift_report_html_path = os.path.join(dir_path, f"{phase}_evidently_drift_report.html")

            # Save the report
            drift_report.save_html(drift_report_html_path)
            print(f"Drift report saved at: {drift_report_html_path}")

            # Log to mlflow
            # mlflow.log_artifact(drift_report_html_path, artifact_path="drift_reports")

            # Log dataset drift metric
            drift_score = drift_report.as_dict()["metrics"][0]["result"]["dataset_drift"]
            # mlflow.log_metric(f"{phase}_dataset_drift", drift_score)
            logging.info(f"Dataset drift score ({phase}): {drift_score}")

            # Raise alert if drift is too high
            DRIFT_THRESHOLD = 0.05
            if drift_score > DRIFT_THRESHOLD:
                if phase == "training":
                    logging.warning("🚨 High Data Drift Detected in Training! Consider retraining the model.")
                else:
                    logging.warning("🚨 Real-Time Data Drift Detected! Investigate the incoming data.")

            return status

        except Exception as e:
            logging.error(f"Error in detecting dataset drift during {phase}: {e}")
            return None, False
            
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
                logging.error(error_message)

                
            status = self.validate_number_of_columns(dataframe=test_dataframe)
            if not status:
                error_message=f"Test dataframe does not contain all columns.\n"   
                logging.error(error_message)


            # logging.info("Checking for numerical Columns for train and test data")
            # self.numerical_exists(train_dataframe)
            # self.numerical_exists(test_dataframe)

            # logging.info("Checking for Categorical Columns for train and test data")
            # self.categorical_exists(train_dataframe)
            # self.categorical_exists(test_dataframe)

            # Check data drift
            status=self.detect_dataset_drift(base_df=train_dataframe,current_df=test_dataframe, phase="training")
            
            dir_path=os.path.dirname(self.data_validation_config.valid_train_file_path)
            os.makedirs(dir_path,exist_ok=True)

            dir_path=os.path.dirname(self.data_validation_config.valid_test_file_path)
            os.makedirs(dir_path,exist_ok=True)

            if status:
                train_dataframe.to_csv(self.data_validation_config.valid_train_file_path, index=False)
                test_dataframe.to_csv(self.data_validation_config.valid_test_file_path, index=False)
                logging.info("Validation successful, data stored in valid directory.")
                return DataValidationArtifact(
                    valid_train_file_path=self.data_validation_config.valid_train_file_path,
                    valid_test_file_path=self.data_validation_config.valid_test_file_path,
                    invalid_train_file_path=None,
                    invalid_test_file_path=None,
                    validation_status=status,
                    drift_report_file_path=self.data_validation_config.drift_report_file_path,
                )
            else:
                train_dataframe.to_csv(self.data_validation_config.invalid_train_file_path, index=False)
                test_dataframe.to_csv(self.data_validation_config.invalid_test_file_path, index=False)
                logging.warning("Validation failed, data stored in invalid directory.")
                return DataValidationArtifact(
                    valid_train_file_path=None,
                    valid_test_file_path=None,
                    invalid_train_file_path=self.data_validation_config.invalid_train_file_path,
                    invalid_test_file_path=self.data_validation_config.invalid_test_file_path,
                    validation_status=status,
                    drift_report_file_path=self.data_validation_config.drift_report_file_path
                )

        except Exception as e:
            raise CreditCardException(e, sys)