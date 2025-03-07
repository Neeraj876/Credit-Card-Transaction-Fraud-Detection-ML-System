import os
import sys
import numpy as np
import pandas as pd
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

from src.exception.exception import CreditCardException
from src.logging.logger import logging
from src.constants import *
from src.entity.artifact_entity import DataTransformationArtifact, DataValidationArtifact
from src.entity.config_entity import DataTransformationConfig
from src.utils.main_utils.utils import read_yaml_file, save_numpy_array_data, save_object
from src.constants import *

from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from category_encoders import TargetEncoder

class DataTransformation:
    def __init__(self, data_validation_artifact: DataValidationArtifact, data_transformation_config: DataTransformationConfig):
        """
        Initializes DataTransformation class.
        Args:
            data_validation_artifact (DataValidationArtifact): Contains validated data paths and status.
            data_transformation_config (DataTransformationConfig): Configuration settings for data transformation.
        """
        self.data_validation_artifact= data_validation_artifact
        self.data_transformation_config=data_transformation_config
        self._schema_config=read_yaml_file(SCHEMA_FILE_PATH)

    def get_transformed_pipeline(self, num_features, cat_features):
        """
        Creates a data transformation pipeline for numerical and categorical features.
        Args:
            num_features (list): List of numerical feature names.
            cat_features (list): List of categorical feature names.
        Returns:
            ColumnTransformer: Preprocessing pipeline.
        """

        num_pipeline = Pipeline(
            steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', MinMaxScaler())
        ])

        cat_pipeline = Pipeline(
            steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('encoder', TargetEncoder())
        ])

        preprocessor = ColumnTransformer(
            transformers=[
            ('num', num_pipeline, num_features),
            ('cat', cat_pipeline, cat_features)
        ])

        return preprocessor
    
    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Performs feature engineering on the dataset by adding new features.
        Args:
            df (pd.DataFrame): Input dataset.
        Returns:
            pd.DataFrame: Transformed dataset with new features.
        """
        try:
            
            df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
            # Extract date and time separately
            df['trans_date'] = df['trans_date_trans_time'].dt.strftime("%Y-%m-%d")
            df['trans_date'] = pd.to_datetime(df['trans_date'])
            df['dob']=pd.to_datetime(df['dob'])
            df['trans_month'] = pd.DatetimeIndex(df['trans_date']).month
            df['trans_year'] = pd.DatetimeIndex(df['trans_date']).year
            # df['latitude_distance'] = abs(round(df['merch_lat'] - df['lat'], 2))
            # df['longitude_distance'] = abs(round(df['merch_long'] - df['long'], 2))
            # df['gender'] = df['gender'].replace({'F': 0, 'M': 1}).astype("int64")
            df["event_timestamp"] = pd.to_datetime(df["trans_date_trans_time"])
            
            return df
        except Exception as e:
            logging.error(f"Error in feature engineering: {e}")
            raise CreditCardException(e, sys)
        
    def drop_columns(self, df: pd.DataFrame, cols: list) -> pd.DataFrame:
        """
        Drops specified columns from the dataset.
        Args:
            df (pd.DataFrame): Input dataset.
            cols (list): List of column names to be dropped.
        Returns:
            pd.DataFrame: Dataset after dropping specified columns.
        """
        try:

            df = df.drop(columns=cols, axis=1)

            return df
        except Exception as e:
            logging.error(f"Error in dropping columns: {e}")
            raise CreditCardException(e, sys)
        
    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Initiates the data transformation process, including:
        - Handling missing values & duplicates
        - Applying feature engineering
        - Scaling features
        - Saving transformed datasets
        Returns:
            DataTransformationArtifact: Paths to transformed data and preprocessor object.
        """
        try:
            if not self.data_validation_artifact.validation_status:
                logging.error("Data Validation failed. Stopping execution.")
                raise ValueError("Invalid training data. Exiting pipeline.")  # Stop further processing

            train_data_path = self.data_validation_artifact.valid_train_file_path  # Proceed only if valid

            
            if not self.data_validation_artifact.validation_status:
                logging.error("Data Validation failed. Stopping execution.")
                raise ValueError("Invalid testing data. Exiting pipeline.")  # Stop further processing

            test_data_path = self.data_validation_artifact.valid_test_file_path  # Proceed only if valid


            train_data = pd.read_csv(train_data_path)
            test_data = pd.read_csv(test_data_path)
            logging.info("Read the train and test validated data")
            
            train_data = self.feature_engineering(train_data)
            test_data = self.feature_engineering(test_data)
            logging.info("Performed feature engineering for train and test data successfully")

            X_train, y_train = train_data.drop(columns=[TARGET_COLUMN]), train_data[TARGET_COLUMN]
            X_test, y_test = test_data.drop(columns=[TARGET_COLUMN]), test_data[TARGET_COLUMN]

            num_features = [column for column in X_train.columns if X_train[column].dtype != 'object']
            cat_features = [column for column in X_train.columns if column not in num_features]

            pre_processor = self.get_transformed_pipeline(num_features, cat_features)
            logging.info("Data transformation pipeline created successfully.")

            X_train_processed = pre_processor.fit_transform(X_train, y_train)
            X_test_processed = pre_processor.transform(X_test)
            logging.info("Fit transformation pipeline to training data and transformation to testing data")

            smote = SMOTE(sampling_strategy="minority")
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)
            X_test_resampled, y_test_resampled = smote.fit_resample(X_test_processed, y_test)
            logging.info("Applied SMOTE for class balancing.")

            train_arr = np.c_[X_train_resampled, y_train_resampled]
            test_arr = np.c_[X_test_resampled, y_test_resampled]

            os.makedirs(self.data_transformation_config.data_transformation_dir, exist_ok=True)
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)
            save_object(
                self.data_transformation_config.transformed_object_file_path,
                pre_processor
            )
            logging.info(f"Saved the arrays and preprocessor object.")

            save_object(FINAL_PREPROCESSOR_PATH, pre_processor)

            logging.info("Data transformation completed successfully.")
            return DataTransformationArtifact(
                self.data_transformation_config.transformed_train_file_path,
                self.data_transformation_config.transformed_test_file_path,
                self.data_transformation_config.transformed_object_file_path
            )

        except Exception as e:
            logging.error(f"Error in data transformation: {e}")
            raise CreditCardException(e, sys)


