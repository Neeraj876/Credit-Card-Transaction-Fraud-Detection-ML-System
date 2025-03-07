import os
import sys
import numpy as np
import pandas as pd

from src.exception.exception import CreditCardException
from src.logging.logger import logging
from src.entity.config_entity import ModelTrainingConfig
from src.entity.artifact_entity import ModelTrainerArtifact, DataTransformationArtifact
from src.constants import *

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import f1_score

from src.utils.ml_utils.model.estimator import FraudModel
from src.utils.main_utils.utils import save_object, load_numpy_array_data, load_object, evaluate_models
from src.utils.ml_utils.metric.classification_metric import get_classification_score

class ModelTrainer:
    def __init__(self, data_transformation_artifact:DataTransformationArtifact, 
    model_trainer_config:ModelTrainingConfig):
        try:
            self.model_trainer_config=model_trainer_config
            self.data_transformation_artifact=data_transformation_artifact
        except Exception as e:
            raise CreditCardException(e, sys)

    def train_model(self, X_train, y_train, X_test)

