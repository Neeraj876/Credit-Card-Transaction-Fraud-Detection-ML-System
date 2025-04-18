import yaml
from src.exception.exception import CreditCardException
from src.logging.logger import logging
import os,sys
import numpy as np
import pickle

from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV



def read_yaml_file(file_path: str) -> dict:
    """
    Reads the schema.yaml file
    """
    try:
        with open(file_path, "r") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise CreditCardException(e, sys)
    
def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content, file)
    except Exception as e:
        raise CreditCardException(e, sys)
    
def save_numpy_array_data(file_path: str, array: np.array):
    """
    Save numpy array data to file
    file_path: str location of file to save
    array: np.array data to save
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise CreditCardException(e, sys) from e
    
def save_object(file_path: str, obj: object) -> None:
    try:
        logging.info("Entered the save_object method of MainUtils class")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logging.info("Exited the save_object method of MainUtils class")
    except Exception as e:
        raise CreditCardException(e, sys) from e
    
def load_object(file_path: str, ) -> object:
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file: {file_path} is not exists")
        with open(file_path, "rb") as file_obj:
            print(file_obj)
            return pickle.load(file_obj)
    except Exception as e:
        raise CreditCardException(e, sys) from e
    
def load_numpy_array_data(file_path: str) -> np.array:
    """
    load numpy array data from file
    file_path: str location of file to load
    return: np.array data loaded
    """
    try:
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise CreditCardException(e, sys) from e
    
def evaluate_models(X_train, y_train, X_test, y_test,models,param):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]
            
            # logging.info(f"Param is: {param}")

            # Log the current model name being evaluated
            logging.info(f"Evaluating model: {model}")
            logging.info(f"Model parameters for {model}: {para}")

            gs = GridSearchCV(model, para, cv=3, n_jobs=-1)
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            logging.info(f"Best parameters for {model}: {gs.best_params_}")
            model.fit(X_train,y_train)

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = f1_score(y_train, y_train_pred)

            test_model_score = f1_score(y_test, y_test_pred)

            logging.info(f"Model {model} - Train F1 Score: {train_model_score}, Test F1 Score: {test_model_score}")

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CreditCardException(e, sys)