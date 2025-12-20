import pandas as pd
import numpy as np
import warnings
import os
os.chdir('C:/Users/chand/Desktop/Learn_ML')

# Sklearn Imports
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler, PowerTransformer
from sklearn.impute import KNNImputer
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split

# Model Imports
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# Suppress warnings
warnings.filterwarnings('ignore')

class AdvancedFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):

        if 'Item_Identifier' in X.columns and 'Item_Visibility' in X.columns:
            self.visibility_map_ = X.groupby('Item_Identifier')['Item_Visibility'].mean().to_dict()
            self.global_vis_mean_ = X['Item_Visibility'].mean()
        return self
    
    def transform(self, X):
        X = X.copy()

        if 'Item_Visibility' in X.columns and hasattr(self, 'visibility_map_'):
            def fill_vis(row):
                if row['Item_Visibility'] == 0:
                    return self.visibility_map_.get(row['Item_Identifier'], self.global_vis_mean_)
                return row['Item_Visibility']
            
            X['Item_Visibility'] = X.apply(fill_vis, axis=1)

        if 'Item_Identifier' in X.columns:
            X['Item_Type_New'] = X['Item_Identifier'].apply(lambda x: x[:2])
            X['Item_Type_New'] = X['Item_Type_New'].map({
                'FD': 'Food',
                'NC': 'Non-Consumable',
                'DR': 'Drinks'
            })

        if 'Outlet_Establishment_Year' in X.columns:
            X['Outlet_Age'] = 2013 - X['Outlet_Establishment_Year']

        if 'Item_Fat_Content' in X.columns:
            X['Item_Fat_Content'] = X['Item_Fat_Content'].replace({
                'LF': 'Low Fat', 'low fat': 'Low Fat', 'reg': 'Regular'
            })

            if 'Item_Type_New' in X.columns:
                mask = X['Item_Type_New'] == 'Non-Consumable'
                X.loc[mask, 'Item_Fat_Content'] = 'Non-Edible'
                
        return X

class DropFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, features_to_drop=['Item_Identifier', 'Outlet_Establishment_Year']):
        self.features_to_drop = features_to_drop
    def fit(self, X, y=None): return self
    def transform(self, X):
        return X.drop(columns=self.features_to_drop, errors='ignore')
    
if __name__ == "__main__":
    print("Loading and Preparing Data...")
    
    # Load Data
    train_df = pd.read_csv('train_v9rqX0R.csv')
    test_df = pd.read_csv('test_AbJTz2l.csv')

    # Separate Target
    X = train_df.drop(columns=['Item_Outlet_Sales'])
    y = train_df['Item_Outlet_Sales']
    
    # Validation Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    ordinal_cols = ['Item_Fat_Content', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type', 'Item_Type_New']
    ohe_cols = ['Item_Type', 'Outlet_Identifier'] 

    preprocessor = ColumnTransformer(transformers=[
        # Ordinal Encoding for the new groups
        ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan), ordinal_cols),
        # One Hot Encoding for nominal groups
        ('ohe', OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore'), ohe_cols)
    ], remainder='passthrough')


    print("Initializing Stacking Ensemble...")
    
    best_xgb = XGBRegressor(
        n_estimators=500, learning_rate=0.01, max_depth=3,
        subsample=0.9, colsample_bytree=0.8, n_jobs=-1, random_state=42
    )

    best_lgbm = LGBMRegressor(
        n_estimators=500, learning_rate=0.01, num_leaves=20,
        subsample=0.8, colsample_bytree=0.8, verbose=-1, n_jobs=-1, random_state=42
    )

    stacking_ensemble = StackingRegressor(
        estimators=[('xgb', best_xgb), ('lgbm', best_lgbm)],
        final_estimator=Ridge(alpha=1.0),
        passthrough=False
    )


    final_pipeline = Pipeline([
        ('feature_eng', AdvancedFeatureEngineer()),    
        ('drop_cols', DropFeatures()),                 
        ('preprocess', preprocessor),                  
        ('impute', KNNImputer(n_neighbors=10)),        
        ('power', PowerTransformer(method='yeo-johnson')), 
        ('stacking', stacking_ensemble)                
    ])

    # Add Target Transformation (Critical for Sales)
    final_model = TransformedTargetRegressor(
        regressor=final_pipeline,
        transformer=PowerTransformer(method='yeo-johnson')
    )

    # --- Training ---
    print("Fitting Model on Full Training Data...")
    final_model.fit(X_train, y_train)
    
    # Check Local Score
    local_pred = final_model.predict(X_test)
    local_rmse = root_mean_squared_error(y_test, local_pred)
    print(f"Local Test RMSE: {local_rmse:.4f} (Aiming for < 1080)")

    final_model.fit(X, y) # Fit on 100% of training rows

    
    # 1. Save IDs for file
    submission_ids = test_df[['Item_Identifier', 'Outlet_Identifier']]
    
    # 2. Predict (Pipeline handles Feature Engineering & Dropping IDs internally)
    final_preds = final_model.predict(test_df)
    
    # 3. Sanity Check (No negative sales)
    final_preds = np.maximum(final_preds, 0)
    
    # 4. Save
    submission = pd.DataFrame({
        'Item_Identifier': submission_ids['Item_Identifier'],
        'Outlet_Identifier': submission_ids['Outlet_Identifier'],
        'Item_Outlet_Sales': final_preds
    })
    
    submission.to_csv('submission.csv', index=False)
