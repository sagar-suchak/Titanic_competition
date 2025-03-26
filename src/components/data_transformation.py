import sys
import os
from dataclasses import dataclass


import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.exception import custom_exception
from src.logger import logging
from src.utils import save_object

@dataclass
class data_transformation_config:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')
    logging.info('Preprocessor Pickel file loaded successfully')

class data_transformation:
    def __init__(self):
        self.data_transformation_config = data_transformation_config()

    def get_data_transformer(self):

        ### This function is responsible for data transformation

        try:
            numerical_features = ['Age','SibSp','Parch','Fare']
            cat_features = ['Pclass','Sex','Embarked']

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy='mean')),
                    ("Standard Scaler", StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy='most_frequent')),
                    ("One Hot Encoder", OneHotEncoder())
                ]
            )

            logging.info('Numerical & Categorical Pipeline Created')

            preprocessor = ColumnTransformer(
                [
                    ("num pipeline", num_pipeline, numerical_features),
                    ("categorical pipeline", cat_pipeline, cat_features)
                ]
            )

            logging.info('Column Transformer created')

            return preprocessor
        
        except Exception as e:
            raise custom_exception(e, sys)
        

    def initiate_data_transformation(self, train_path,test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Initiating Data Transformation')

            preprocessor_obj = self.get_data_transformer()

            train_df['FamilySize'] = train_df['SibSp'] + train_df['Parch'] + 1
            train_df['IsAlone'] = train_df['FamilySize'].apply(lambda x: 1 if x == 1 else 0)

            test_df['FamilySize'] = test_df['SibSp'] + test_df['Parch'] + 1
            test_df['IsAlone'] = test_df['FamilySize'].apply(lambda x: 1 if x==1 else 0)

            target_column = 'Survived'
            numerical_features = ['Age','SibSp','Parch','Fare']
            cat_features = ['Pclass','Sex','Embarked']
            
            columns_to_drop = ['Name','Ticket','Cabin']

            train_df.drop(columns=columns_to_drop, inplace=True,errors='ignore')
            test_df.drop(columns=columns_to_drop, inplace=True,errors='ignore')

            input_features_train_df = train_df.drop(columns=[target_column], axis=1)
            target_feature_train_df = train_df[target_column]

            logging.info('Applying the preprocessor on the train and test data')
            
            input_features_train_arr = preprocessor_obj.fit_transform(input_features_train_df)

            train_arr = np.c_[
                input_features_train_arr, np.array(target_feature_train_df)
            ]

            if target_column in test_df.columns:
                input_features_test_df = test_df.drop(columns=target_column)
                target_feature_test_df = test_df[target_column]
                
                input_features_test_arr = preprocessor_obj.fit_transform(input_features_test_test)

                test_arr = np.c_[
                input_features_test_arr, np.array(target_features_test_arr)]
            else:
                input_features_test_df = test_df
                input_features_test_arr = preprocessor_obj.transform(input_features_test_df)
                test_arr = input_features_test_arr
                        
            
            logging.info('Data Transformation completed Successfully')

            save_object(
                self.data_transformation_config.preprocessor_obj_file_path,
                preprocessor_obj
            )

            return(
                train_arr, 
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        
        except Exception as e:
            raise custom_exception(e, sys)
        
