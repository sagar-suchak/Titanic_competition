import os
import sys
import dill
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from src.exception import custom_exception
from src.logger import logging

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
    
    except Exception as e:
        raise custom_exception(e,sys)
    

def evaluate_model(X, y, models):
    def evaluate_model(model, X_test, y_test):
        try:
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            return acc
        except Exception as e:
            raise custom_exception(e, sys)
        

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise custom_exception(e, sys)