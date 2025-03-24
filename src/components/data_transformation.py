import os
import sys
import numpy as np
import pandas as pd
import warnings
import mlflow
from mlflow.pyfunc import PythonModel
from mlflow.models.signature import infer_signature
from typing import Any

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
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from category_encoders import TargetEncoder

# Ensure TargetEncoder outputs numeric values
# def ensure_numeric(X):
#     return X.astype(np.float32)

# class TransformerWithPredict:
#     def __init__(self, transformer):
#         self.transformer = transformer
    
#     def predict(self, X):
#         return self.transformer.transform(X)
    
#     def transform(self, X):
#         return self.transformer.transform(X)
    
#     def fit(self, X, y=None):
#         self.transformer.fit(X, y)
#         return self
    
#     # Include this to ensure all methods are passed through
#     def __getattr__(self, name):
#         if hasattr(self.transformer, name):
#             return getattr(self.transformer, name)
#         raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

# Create a PythonModel class for the wrapper
class PreprocessorWrapper(PythonModel):
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor
    
    def predict(self, context: Any, model_input: pd.DataFrame) -> np.ndarray:
        """
        Transform data using the wrapped preprocessor.
        
        Args:
            context: MLflow model context (not used but required by MLflow)
            model_input: Pandas DataFrame to transform
            
        Returns:
            Transformed data from the preprocessor
        """
        # Ensure model_input is a pandas DataFrame
        import pandas as pd
        if not isinstance(model_input, pd.DataFrame):
            if isinstance(model_input, np.ndarray):
                # Convert numpy array to DataFrame if needed
                model_input = pd.DataFrame(model_input)
            else:
                raise TypeError("Input must be a pandas DataFrame or numpy array")
        
        # Apply the transform method of the preprocessor
        return self.preprocessor.transform(model_input)

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
            ('scaler', StandardScaler())
        ])

        cat_pipeline = Pipeline(
            steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', TargetEncoder())
        ])

        preprocessor = ColumnTransformer(
            transformers=[
            ('num', num_pipeline, num_features),
            ('cat', cat_pipeline, cat_features)
        ])

        return preprocessor
        
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

            train_data = pd.read_csv(self.data_validation_artifact.valid_train_file_path)
            test_data = pd.read_csv(self.data_validation_artifact.valid_test_file_path)
            logging.info("Read the train and test validated data")

            # X_train, y_train = train_data.drop(columns=["is_fraud", "transaction_id", "event_timestamp"]), train_data[TARGET_COLUMN]
            # X_test, y_test = test_data.drop(columns=["is_fraud", "transaction_id", "event_timestamp"]), test_data[TARGET_COLUMN]
            # print("X_train: ", X_train.head(2))
            # print("X_test: ", X_train.head(2))

            ## training dataframe
            X_train = train_data.drop(columns=["is_fraud", "transaction_id", "event_timestamp"],axis=1)
            y_train = train_data[TARGET_COLUMN]
            print("X-train: ", X_train)
            # y_train = y_train.replace(-1, 0).values

            ## testing dataframe
            X_test = test_data.drop(columns=["is_fraud", "transaction_id", "event_timestamp"],axis=1)
            y_test = test_data[TARGET_COLUMN]
            print("y-train: ", y_train)
            # y_test = y_test.replace(-1, 0).values

            num_features = [column for column in X_train.columns if X_train[column].dtype != 'object']
            print("Numerical: ", num_features)
            cat_features = [column for column in X_train.columns if column not in num_features]
            print("Categorical", cat_features)

            pre_processor = self.get_transformed_pipeline(num_features, cat_features)
            logging.info("Data transformation pipeline created successfully.")

            X_train_arr = pre_processor.fit_transform(X_train, y_train).astype(np.float32)
            X_test_arr = pre_processor.transform(X_test).astype(np.float32)
            logging.info("Fit transformation pipeline to training data and transformation to testing data")

            # train_arr = np.c_[X_train_processed, np.array(y_train)]
            # test_arr = np.c_[X_test_processed, np.array(y_test)]

            # Debugging: Ensure transformation output is numeric
            print("X_train_processed dtype:", X_train_arr.dtype)
            print("X_test_processed dtype:", X_test_arr.dtype)

            smote = SMOTE(sampling_strategy="minority")
            X_train_resampled_arr, y_train_resampled_arr = smote.fit_resample(X_train_arr, np.array(y_train))
            logging.info("Applied SMOTE for class balancing.")

            # Save transformed data
            train_arr = np.c_[X_train_resampled_arr, y_train_resampled_arr]
            test_arr = np.c_[X_test_arr, np.array(y_test)]

            os.makedirs(self.data_transformation_config.data_transformation_dir, exist_ok=True)
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)
            save_object(
                self.data_transformation_config.transformed_object_file_path,
                pre_processor
            )
            logging.info(f"Saved the arrays and preprocessor object.")

            save_object(FINAL_PREPROCESSOR_PATH, pre_processor)
            # mlflow.sklearn.log_model(
            #     pre_processor, 
            #     "preprocessor", 
            #     registered_model_name="feature_preprocessor"
            # ) 

            # # wrap the fitted preprocessor for MLflow logging
            # wrapped_preprocessor = TransformerWithPredict(pre_processor)

            # # Log the wrapped preprocessor to MLflow
            # mlflow.sklearn.log_model(
            #     wrapped_preprocessor,
            #     "preprocessor", 
            #     registered_model_name="feature_preprocessor"
            # )

            # Then in your training code
            wrapped_model = PreprocessorWrapper(pre_processor)

            # Infer the model signature using test data
            signature = infer_signature(X_test, pre_processor.transform(X_test))

            # Log the model with pyfunc
            mlflow.pyfunc.log_model(
                artifact_path="preprocessor",
                python_model=wrapped_model,
                signature=signature,
                registered_model_name="feature_preprocessor"
            )

            logging.info("Data transformation completed successfully.")
            return DataTransformationArtifact(
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path
            )

        except Exception as e:
            logging.error(f"Error in data transformation: {e}")
            raise CreditCardException(e, sys)


