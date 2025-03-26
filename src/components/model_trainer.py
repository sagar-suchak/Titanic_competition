import os
import sys
from dataclasses import dataclass
import pandas as pd
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score

from src.exception import custom_exception
from src.logger import logging
from src.utils import save_object, evaluate_model


@dataclass
class model_trainer_config:
    trained_model_path = os.path.join('artifacts', 'model.pkl')
    logging.info('Pickle file loaded successfully')

class ModelTrainer:
    def __init__(self):
        self.train_path = os.path.join('artifacts', 'train.csv')
        self.test_path = os.path.join('artifacts','test.csv')

    def initiate_model_training(self, train_arr,test_arr):
        try:
            logging.info('Reading Train and Test Dataset')
            X_train, y_train, X_test= (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1]
            )
            
            models={
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "KNN": KNeighborsClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(),
                "AdaBoost": AdaBoostClassifier(),
                "CatBoost": CatBoostClassifier(verbose=0),
                "XGBoost": XGBClassifier(use_label_encoder = False, eval_metric = 'logloss'),
                "Naive Bayes" : GaussianNB()

            }

            params = {
                "Logistic Regression": {
                    "C": [0.1, 1, 10],
                    "solver": ['liblinear', 'lbfgs']
                },
                "KNN": {
                    "n_neighbors": [3, 5, 7],
                    "weights": ['uniform', 'distance']
                },
                "Decision Tree": {
                    "max_depth": [3, 5, 10],
                    "criterion": ['gini', 'entropy']
                },
                "Random Forest": {
                    "n_estimators": [100, 200],
                    "max_depth": [5, 10, None]
                },
                "AdaBoost": {
                    "n_estimators": [50, 100],
                    "learning_rate": [0.5, 1.0]
                },
                "CatBoost": {
                    "iterations": [100, 200],
                    "depth": [3, 6],
                    "learning_rate": [0.03, 0.1]
                },
                "XGBoost": {
                    "n_estimators": [100, 200],
                    "learning_rate": [0.05, 0.1],
                    "max_depth": [3, 5]
                },
                "Naive Bayes": {}  # No hyperparameters to tune
            }

            best_models = {}
            scores = {}

            for name, model in models.items():
                logging.info(f"Hyperparameter Tuning for {name}")
                if params[name]:
                    grid = GridSearchCV(model, params[name], cv=5, scoring='accuracy', n_jobs=-1)
                    grid.fit(X_train,y_train)
                    best_model = grid.best_estimator_
                    logging.info(f"{name} best params : {grid.best_params_}")

                else:
                    model.fit(X_train, y_train)
                    best_model = model

                acc = cross_val_score(best_model, X_train, y_train, cv=5, scoring='accuracy').mean()
                best_models[name] =best_model
                scores[name] = acc
                logging.info(f"{name} Accuracy: {acc}")
            
            logging.info('Training and Evaluation completed')
            # Find the best model based on highest accuracy
            best_model_name = max(scores, key=scores.get)
            final_model = best_models[best_model_name]
            logging.info(f"Best Model: {best_model_name} with Accuracy: {scores[best_model_name]:.4f}")

            print("Model Accuracies :\n", scores)
            save_object(
                model_trainer_config().trained_model_path, final_model
                )
            
            
            logging.info(f"Final model saved at: {model_trainer_config().trained_model_path}")

            return final_model, best_model_name, scores[best_model_name]

        
        except Exception as e:
            raise custom_exception(e, sys)
        



