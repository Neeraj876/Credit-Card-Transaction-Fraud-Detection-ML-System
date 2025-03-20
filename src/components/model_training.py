import os
import sys
import numpy as np
import pandas as pd
import xgboost as xgb

from src.exception.exception import CreditCardException
from src.logging.logger import logging
from src.entity.config_entity import ModelTrainingConfig
from src.entity.artifact_entity import ModelTrainerArtifact, DataTransformationArtifact
from src.constants import *

from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import f1_score

from src.utils.ml_utils.model.estimator import FraudModel
from src.utils.main_utils.utils import save_object, load_numpy_array_data, load_object, evaluate_models
from src.utils.ml_utils.metric.classification_metric import get_classification_score

import mlflow
from mlflow.models import infer_signature
from sklearn.base import BaseEstimator
from urllib.parse import urlparse

# Set MLflow tracking URI from environment variable
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_TRACKING_URI

class ModelTrainer:
    def __init__(self, data_transformation_artifact:DataTransformationArtifact, 
    model_trainer_config:ModelTrainingConfig):
        try:
            self.model_trainer_config=model_trainer_config
            self.data_transformation_artifact=data_transformation_artifact
        except Exception as e:
            raise CreditCardException(e, sys)

    def track_mlflow(self, best_model, best_model_name, classification_train_metric=None, classification_test_metric=None, X_train=None):

        try:
            # remote_server_uri = "http://ec2-34-207-207-10.compute-1.amazonaws.com:5000/"
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

            # End any existing active run before starting a new one
            if mlflow.active_run():
                logging.warning(f"Ending previous MLflow run: {mlflow.active_run().info.run_id}")
                mlflow.end_run()

            with mlflow.start_run(run_name=best_model_name):
                logging.info(f"Starting MLflow run: {mlflow.active_run().info.run_id}")

                # Log model parameters if available
                if hasattr(best_model, 'get_params'):
                    mlflow.log_params(best_model.get_params())

                # Log training metrics if available
                if classification_train_metric:
                    mlflow.log_metrics({
                        "train_f1_score": classification_train_metric.f1_score,
                        "train_precision": classification_train_metric.precision_score,
                        "train_recall": classification_train_metric.recall_score
                    })

                # Log test metrics if available
                if classification_test_metric:
                    mlflow.log_metrics({
                        "test_f1_score": classification_test_metric.f1_score,
                        "test_precision": classification_test_metric.precision_score,
                        "test_recall": classification_test_metric.recall_score
                    })

                # Take a sample input example for logging
                input_example = X_train[:1] if X_train is not None else None
                signature = infer_signature(X_train, best_model.predict(X_train)) if X_train is not None else None

                # Determine the appropriate logging function for the model
                if isinstance(best_model, xgb.Booster):
                    log_model_func = mlflow.xgboost.log_model
                elif isinstance(best_model, xgb.XGBModel):
                    log_model_func = mlflow.xgboost.log_model
                else:
                    log_model_func = mlflow.sklearn.log_model

                # Check tracking storage type
                tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

                # Register model only if not using file-based storage
                if tracking_url_type_store != "file":
                    log_model_func(
                        best_model, 
                        "model", 
                        registered_model_name=best_model_name,
                        signature=signature,
                        input_example=input_example
                    )
                    logging.info(f"Model registered as: {best_model_name}")
                else:
                    log_model_func(
                        best_model, 
                        "model",
                        signature=signature,
                        input_example=input_example
                    )
                    logging.info("Model logged but not registered (file-based storage)")

                logging.info(f"MLflow run completed with ID: {mlflow.active_run().info.run_id}")

        except Exception as e:
            logging.error(f"Error during MLflow tracking: {e}")
            raise

        finally:
            # Ensure the run is properly ended
            if mlflow.active_run():
                mlflow.end_run()
            logging.info("MLflow run ended successfully.")
                
    def train_model(self, X_train, y_train, X_test, y_test):
        try:
            models = {
            "Logistic Regression": LogisticRegression(max_iter=5000, class_weight="balanced", verbose=1),
            "XGBClassifier":  XGBClassifier(min_child_weight=3, subsample=0.7,  colsample_bytree=0.7,  reg_lambda=1, reg_alpha=1, objective="binary:logistic",  eval_metric="logloss", verbosity=1, random_state=42), 
            # "SVC": SVC(verbose=1),
            "RandomForestClassifier": RandomForestClassifier(  class_weight="balanced", random_state=42, verbose=1),
            }

            params = {
                "Logistic Regression": {
                    'C': [0.1, 1, 5],  # Regularization strength. Smaller values apply stronger regularization.

                    'solver': ['lbfgs', 'liblinear', 'saga'],  # Optimization algorithms. 
                },

                "XGBClassifier": {
                    'n_estimators': [50, 100, 200],  # Number of boosting rounds. More rounds improve performance but can lead to overfitting.

                    'learning_rate': [0.01, 0.1, 0.05],  # Step size for each boosting round. 

                    'max_depth': [3, 6],  # Maximum depth of each tree. 
                },

                "RandomForestClassifier": {
                    'n_estimators': [50, 100],  # Number of trees in the forest. More trees generally lead to better performance but increase computation time.

                    'max_depth': [10, 20],  # Maximum depth of trees. 
                },

                # "SVC": {
                #     'C': [0.1, 1],  # Regularization strength. Larger values make the decision boundary more complex, smaller values create a smoother boundary.

                #     'kernel': ['linear', 'rbf'],  # Types of kernels. 
                # }    
            }

            logging.info("Entered into evaluate models")

            model_report:dict=evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, param=params)

            # To get best model score from dict
            best_model_score = max(sorted(model_report.values()))
            print(best_model_score)

            # To get best model name from dict
            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            print(best_model_name)

            best_model=models[best_model_name]

            logging.info("Got the best model")

            if best_model_score < self.model_trainer_config.expected_accuracy:
                logging.info("No best model found with score more than base score")
                raise Exception("No best model found with score more than base score")
            
            y_train_pred=best_model.predict(X_train)

            classification_train_metric=get_classification_score(y_true=y_train,y_pred=y_train_pred)

            # Track the experiments with mlflow
            # self.track_mlflow(best_model=best_model, best_model_name=best_model_name, classification_train_metric=classification_train_metric, X_train=X_train)

            y_test_pred=best_model.predict(X_test)

            classification_test_metric=get_classification_score(y_true=y_test,y_pred=y_test_pred)

            # Track the experiments with mlflow
            # self.track_mlflow(best_model=best_model, best_model_name=best_model_name, classification_test_metric=classification_test_metric, X_train=X_test)

            # Track the experiments with mlflow - single run with both train and test metrics
            self.track_mlflow(
                best_model=best_model, 
                best_model_name=best_model_name, 
                classification_train_metric=classification_train_metric, 
                classification_test_metric=classification_test_metric, 
                X_train=X_train
            )

            preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)

            model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir_path,exist_ok=True)

            fraud_model=FraudModel(preprocessor=preprocessor,model=best_model)
            save_object(self.model_trainer_config.trained_model_file_path,obj=fraud_model)

            # Model pusher
            save_object("final_model/model.pkl",best_model)

            # Model Trainer Artifact
            model_trainer_artifact=ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=classification_train_metric,
                test_metric_artifact=classification_test_metric
            )

            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact

        except Exception as e:
            raise CreditCardException(e, sys)
        
    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            train_file_path=self.data_transformation_artifact.transformed_train_file_path
            test_file_path=self.data_transformation_artifact.transformed_test_file_path

            # Loading training array and testing array
            train_array = load_numpy_array_data(train_file_path)
            test_array=load_numpy_array_data(test_file_path)

            logging.info("Splitting train and test input data")
            X_train, y_train, X_test, y_test=(
                train_array[:, :-1], 
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            model_trainer_artifact=self.train_model(X_train,y_train,X_test,y_test)
            return model_trainer_artifact

        except Exception as e:
            raise CreditCardException(e,sys)
