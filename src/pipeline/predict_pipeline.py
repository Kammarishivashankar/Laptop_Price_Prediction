import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = 'artifacts/model.pkl'
            preprocessor_path = 'artifacts/preprocessor.pkl'

            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            transformed_data = preprocessor.transform(features)
            pred = model.predict(transformed_data)
            return pred
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self,
                 Company: str,
                 TypeName: str,
                 Ram: int,
                 Gpu: str,
                 Touchscreen: int,
                 IPS_screen: int,
                 CPU_Brand: str,
                 Weight: float,
                 HDD: int,
                 SSD: int,
                 OS: str,
                 PPI: float):  # Added PPI as a parameter

        self.Company = Company
        self.TypeName = TypeName
        self.Ram = Ram
        self.Gpu = Gpu
        self.Touchscreen = Touchscreen
        self.IPS_screen = IPS_screen
        self.CPU_Brand = CPU_Brand
        self.Weight = Weight
        self.HDD = HDD
        self.SSD = SSD
        self.OS = OS
        self.PPI = PPI 
    
    def get_data_as_frame(self):
        try:
            input_data = {
                'company': [self.Company],  # Changed to list
                'typeName': [self.TypeName],  # Changed to list
                'ram': [self.Ram],  # Changed to list
                'gpu': [self.Gpu],  # Changed to list
                'touchscreen': [self.Touchscreen],  # Changed to list
                'IPS_screen': [self.IPS_screen],  # Changed to list
                'processor': [self.CPU_Brand],  # Changed to list
                'weight': [self.Weight],  # Changed to list
                'ppi': [self.PPI],
                'HDD': [self.HDD],  # Changed to list
                'SSD': [self.SSD],  # Changed to list
                'OS': [self.OS]  # Changed to list
            }
            return pd.DataFrame(input_data)
        
        except Exception as e:
            raise CustomException(e, sys)
