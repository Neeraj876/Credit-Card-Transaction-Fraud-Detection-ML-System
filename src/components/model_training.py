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

import dagshub
# Automatically configure MLflow tracking with DagsHub
dagshub.init(repo_owner='neerajjj6785', repo_name='real-time-credit-card-transaction-fraud-detection-mlops', mlflow=True)

class ModelTrainer:
    def __init__(self, data_transformation_artifact:DataTransformationArtifact, 
    model_trainer_config:ModelTrainingConfig):
        try:
            self.model_trainer_config=model_trainer_config
            self.data_transformation_artifact=data_transformation_artifact
        except Exception as e:
            raise CreditCardException(e, sys)

    def track_mlflow(self,best_model,best_model_name, classification_train_metric=None, classification_test_metric=None, X_train=None):

            # try:   
                # Check if an MLflow run is already active
                # if mlflow.active_run() is None:
                #     run = mlflow.start_run()
                # else:
                #     run = mlflow.active_run()

            if mlflow.active_run():
                mlflow.end_run()  # Ensure no active runs before starting a new one

            with mlflow.start_run(run_name=best_model_name):

                # Log parameters from the model
                model_params = best_model.get_params() if hasattr(best_model, 'get_params') else {}
                mlflow.log_params(model_params)

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

                # f1_score=classificationmetric.f1_score
                # precision_score=classificationmetric.precision_score
                # recall_score=classificationmetric.recall_score


                # mlflow.log_metric("f1_score",f1_score)
                # mlflow.log_metric("precision",precision_score)
                # mlflow.log_metric("recall_score",recall_score)
                # mlflow.sklearn.log_model(best_model,"model")

                # Log model based on type
                # if isinstance(best_model, xgboost.XGBModel):
                #     mlflow.xgboost.log_model(best_model, "model")
                # elif isinstance(best_model, BaseEstimator):
                #     mlflow.sklearn.log_model(best_model, "model")
                # else:
                #     raise ValueError("Unsupported model type")

                # Take one sample input (assuming X_train is a DataFrame or NumPy array)
                input_example = X_train[:1]  

                # Infer the model signature
                signature = infer_signature(X_train, best_model.predict(X_train))

                # Determine model type and log accordingly
                if hasattr(xgb, 'XGBModel') and isinstance(best_model, xgb.XGBModel):
                    log_model_func = mlflow.xgboost.log_model
                elif hasattr(xgb, 'Booster') and isinstance(best_model, xgb.Booster):
                    log_model_func = mlflow.xgboost.log_model
                else:
                    # Default to scikit-learn for other model types
                    log_model_func = mlflow.sklearn.log_model
                    
                # Check tracking storage type
                tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

                 # Model registry does not work with file store
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
                
                        
                # Model registry does not work with file store
                # Register the model only if it's NOT file-based
                # if tracking_url_type_store != "file":
                
                #     model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
                #     mlflow.register_model(model_uri, best_model_name)
                #     print(f"Model registered as: {best_model_name}")
                # else:
                #     print("Model Registry is not available when using file-based storage.")

            # finally:
            #     # Ensure the run is properly closed if we started it
            #     if mlflow.active_run() and run is not None:
            #         mlflow.end_run()

    def train_model(self, X_train, y_train, X_test, y_test):
        try:
            models = {
            "Logistic Regression": LogisticRegression(max_iter=5000, class_weight="balanced", verbose=1),
            "XGBClassifier":  XGBClassifier(min_child_weight=3, subsample=0.7,  colsample_bytree=0.7,  reg_lambda=1, reg_alpha=1, objective="binary:logistic",  eval_metric="logloss", verbosity=1, random_state=42), 
            # "SVC": SVC(verbose=1),
            "RandomForestClassifier": RandomForestClassifier(  class_weight="balanced", random_state=42, verbose=1),
            }

            # params = {
            #     "Logistic Regression": {
            #         'C': [0.01, 0.1, 1, 10],  # Regularization strength
            #         'solver': ['lbfgs', 'liblinear', 'saga'],  # Saga works well for large datasets
            #     },

            #     "SVC": {
            #         'C': [0.01, 0.1, 1, 5],  # Prevents convergence issues
            #         'kernel': ['linear', 'rbf'],  # Test both
            #         'gamma': ['scale', 'auto'],  # Important for rbf
            #     },

            #     "RandomForestClassifier": {
            #         'n_estimators': [100, 200],  # More trees improve stability
            #         'max_depth': [None, 10, 20],  # None allows full growth
            #         'min_samples_split': [2, 5, 10],  # Regularization for better generalization
            #         'min_samples_leaf': [1, 2, 4],  # Controls overfitting
            #     },

            #     # "XGBClassifier": {
            #     #     'n_estimators': [100, 200],  # More boosting rounds
            #     #     'learning_rate': [0.01, 0.1, 0.2],  # Step size
            #     #     'max_depth': [3, 6, 9],  # Controls tree complexity
            #     #     'subsample': [0.7, 1.0],  # Prevents overfitting
            #     #     'colsample_bytree': [0.7, 1.0],  # Feature selection
            #     # }
            # }


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
