import os,sys
from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException

from sklearn.linear_model import LinearRegression,Lasso,LassoCV,Ridge,RidgeCV,ElasticNet,ElasticNetCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostRegressor,RandomForestRegressor,GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score

from src.utils import save_object
from src.utils import evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    
    def initiate_model_training(self,input_features_train,target_feature_train_df,input_features_test,target_feature_test_df):
        try:
            logging.info('model training started')

            models ={
                    "LinearRegression" : LinearRegression(),
                    "Lasso":Lasso(),
                    "Ridge" : Ridge(),
                    "ElasticNet" : ElasticNet(),
                    "SupportVectorRegressor" : SVR(),
                    "KNeighborsRegressor" : KNeighborsRegressor(),
                    "DecisionTreeRegressor" : DecisionTreeRegressor(),
                    "RandomForestRegressor" : RandomForestRegressor(),
                    "AdaBoostRegressor" : AdaBoostRegressor(),
                    "XGBRegressor" : XGBRegressor()
                    }
            
            model_report:dict = evaluate_models(x_train=input_features_train,y_train=target_feature_train_df,
                                               x_test=input_features_test,y_test=target_feature_test_df,
                                               models = models)
            
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score < 0.7:
                raise CustomException('No best model found')
            logging.info('Best model found on both train and test data')

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )
            
            predicted = best_model.predict(input_features_test)

            r2_Score = r2_score(target_feature_test_df,predicted)

            return r2_Score,model_report


        except Exception as e:
            raise CustomException(e,sys)
    

