import sys,os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig():
    preprocessor_obj_file = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformer(self):
        try:
            categorical_columns = ['Company', 'TypeName', 'Gpu', 'CPU_Brand', 'OS']
            numerical_columns = ['Ram', 'Weight', 'PPI', 'HDD', 'SSD']
            trf = ColumnTransformer(transformers=[
                    ('OHE',OneHotEncoder(sparse_output=False,drop='first',dtype='int16'),categorical_columns),
                    ('scaler',StandardScaler(),numerical_columns)
                    ]
                  ,remainder='passthrough')

            trf.set_output(transform='pandas')

            logging.info('preprocessing completed.')

            return trf

        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):

        try:
            logging.info('Reading the train and test data')
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info('Reading the train and test data completed')

            logging.info('Obtaining the preprocessed data')

            preprocessing_obj = self.get_data_transformer()

            target_column_name = "Price"

            input_features_train_df = train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_features_test_df = test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(
                f"Applying the preprocessing object on train and test df"
            )

            input_features_train = preprocessing_obj.fit_transform(input_features_train_df)
            input_features_test = preprocessing_obj.transform(input_features_test_df)

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file,
                obj = preprocessing_obj
            )
            
            return (input_features_train,target_feature_train_df,input_features_test,target_feature_test_df)


        except Exception as e:
            raise CustomException(e,sys)
