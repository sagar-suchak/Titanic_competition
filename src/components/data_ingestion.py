import os
import sys
from src.exception import custom_exception
from src.logger import logging
from src.components.data_transformation import data_transformation_config
from src.components.data_transformation import data_transformation
from src.components.model_trainer import model_trainer_config
from src.components.model_trainer import ModelTrainer
from src.utils import load_object

import pandas as pd
from dataclasses import dataclass

@dataclass
class data_ingestion_config:
    train_data_path = os.path.join('artifacts', 'train.csv')
    test_data_path = os.path.join('artifacts','test.csv')
    
class data_ingestion:
    def __init__(self):
        self.ingestion_config = data_ingestion_config()

    def initiate_data_ingestion(self):
        logging.info('Entered Data Ingestion Module')
        try:
            df_train = pd.read_csv('notebook/data/train.csv')
            df_test = pd.read_csv('notebook/data/test.csv')

            logging.info('Read the train and test data successfully')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df_train.to_csv(self.ingestion_config.train_data_path, index=False, header=True)

            df_test.to_csv(self.ingestion_config.test_data_path, index=False, header=True) 

            logging.info('Ingestion Completed Successfully')

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        

        except Exception as e:
            raise custom_exception(e, sys)
        

if __name__ == '__main__':
    obj = data_ingestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation_obj = data_transformation()
    train_arr, test_arr, _ = data_transformation_obj.initiate_data_transformation(train_data, test_data)

    model_Train_obj = ModelTrainer()
    print(model_Train_obj.initiate_model_training(train_arr,test_arr))



    # Load original test dataset to retrieve PassengerId
    original_test_df = pd.read_csv('notebook/data/test.csv')
    passenger_ids = original_test_df['PassengerId']

    # Load the preprocessor
    preprocessor = load_object(data_transformation_config().preprocessor_obj_file_path)

    # Load the same test.csv used in training (processed version)
    test_df = pd.read_csv('artifacts/test.csv')

    # Apply same feature engineering
    test_df['FamilySize'] = test_df['SibSp'] + test_df['Parch'] + 1
    test_df['IsAlone'] = test_df['FamilySize'].apply(lambda x: 1 if x == 1 else 0)

    # Drop unnecessary columns (same as training)
    columns_to_drop = ['Name', 'Ticket', 'Cabin']
    test_df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

    # Apply transformation using saved preprocessor
    X_test_final = preprocessor.transform(test_df)

    # Load the best model (saved from training)
    best_model = load_object(model_trainer_config().trained_model_path)

    # Make predictions
    y_pred = best_model.predict(X_test_final)

    # Create submission DataFrame
    submission_df = pd.DataFrame({
        'PassengerId': passenger_ids,
        'Survived': y_pred.astype(int)
    })

    # Save to CSV
    submission_path = 'artifacts/submission.csv'
    submission_df.to_csv(submission_path, index=False)
    print(f"✅ Submission saved at: {submission_path}")