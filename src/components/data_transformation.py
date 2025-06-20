# All the data manipulation and transformation tasks are handled in this file
import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.exception import CustomException
from src.logger import logging

from src.util import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        this function is responsible for creating a data transformation pipeline
        """
        try:
            numerical_features = ['writing_score', 'reading_score']
            categorical_features = ['gender', 'race_ethnicity', 'parental_level_of_education', 
                                    'lunch', 'test_preparation_course']
            logging.info("Numerical and categorical features identified")
            
            # Numerical pipeline
            num_pipeline = Pipeline(
                steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            logging.info("Numerical pipeline created with imputer and scaler")
            
            # Categorical pipeline
            cat_pipeline = Pipeline(
                steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehotencoder', OneHotEncoder(handle_unknown='ignore')),
                ('scaler', StandardScaler(with_mean=False))
            ])
            logging.info("Categorical pipeline created with imputer, one-hot encoder, and scaler")

            # Combine both pipelines
            preprocessor = ColumnTransformer(
                transformers=[
                    ('numerical_pipeline', num_pipeline, numerical_features),
                    ('categorical_pipeline', cat_pipeline, categorical_features)
                ]
            )
            logging.info("ColumnTransformer created to combine numerical and categorical pipelines")
            
            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_data_transformation(self, train_path, test_path):
        """
        this function is responsible for transforming the data
        """
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Train and test data loaded successfully")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = 'math_score'
            numerical_features = ['writing_score', 'reading_score']

            input_features_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_column_train_df = train_df[target_column_name]

            input_features_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_column_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training and testing dataframes")

            input_features_train_arr = preprocessing_obj.fit_transform(input_features_train_df)
            input_features_test_arr = preprocessing_obj.transform(input_features_test_df)

            train_arr = np.c_[input_features_train_arr, target_column_train_df]
            test_arr = np.c_[input_features_test_arr, target_column_test_df]

            logging.info("Preprocessing completed successfully")
            # Save the preprocessor object

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
            
        except Exception as e:
            raise CustomException(e, sys)
            
            