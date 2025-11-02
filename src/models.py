"""
Model implementation for Seoul Bike Sharing Demand Prediction
Implements: CUBIST, RRF, CART, KNN, and CIT models as per the paper
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

class SeoulBikeModels:
    def __init__(self):
        """
        Initialize the models container
        """
        self.models = {}
        self.best_params = {}
        self.results = {}
        
    def evaluate_model(self, y_true, y_pred, model_name='Model'):
        """
        Calculate evaluation metrics as per paper
        """
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        
        # Coefficient of Variation (CV)
        cv = (rmse / np.mean(y_true)) * 100
        
        return {
            'Model': model_name,
            'R2': round(r2, 4),
            'RMSE': round(rmse, 2),
            'MAE': round(mae, 2),
            'CV': round(cv, 2)
        }
    
    def train_cart(self, X_train, y_train, X_test, y_test):
        """
        Train CART (Classification and Regression Tree) model
        """
        print("\nTraining CART model...")
        
        # Hyperparameter grid as per paper
        param_grid = {
            'max_depth': [None, 10, 20, 30, 40],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'ccp_alpha': [0.0001, 0.001, 0.01, 0.1]  # Complexity parameter
        }
        
        # Initialize model
        cart_model = DecisionTreeRegressor(random_state=42)
        
        # Grid search with 10-fold CV
        kfold = KFold(n_splits=10, shuffle=True, random_state=42)
        grid_search = GridSearchCV(
            cart_model, 
            param_grid, 
            cv=kfold,
            scoring='r2',
            n_jobs=-1,
            verbose=1
        )
        
        # Fit the model
        grid_search.fit(X_train, y_train)
        
        # Best parameters
        self.best_params['CART'] = grid_search.best_params_
        print(f"Best parameters: {grid_search.best_params_}")
        
        # Make predictions
        y_train_pred = grid_search.predict(X_train)
        y_test_pred = grid_search.predict(X_test)
        
        # Evaluate
        train_metrics = self.evaluate_model(y_train, y_train_pred, 'CART_train')
        test_metrics = self.evaluate_model(y_test, y_test_pred, 'CART_test')
        
        self.models['CART'] = grid_search.best_estimator_
        self.results['CART'] = {'train': train_metrics, 'test': test_metrics}
        
        return train_metrics, test_metrics
    
    def train_knn(self, X_train, y_train, X_test, y_test):
        """
        Train KNN (K-Nearest Neighbors) model
        """
        print("\nTraining KNN model...")
        
        # Hyperparameter grid as per paper
        param_grid = {
            'n_neighbors': list(range(1, 29)),  # 1 to 28 as per paper
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        }
        
        # Initialize model
        knn_model = KNeighborsRegressor()
        
        # Grid search with 10-fold CV
        kfold = KFold(n_splits=10, shuffle=True, random_state=42)
        grid_search = GridSearchCV(
            knn_model, 
            param_grid, 
            cv=kfold,
            scoring='r2',
            n_jobs=-1,
            verbose=1
        )
        
        # Fit the model
        grid_search.fit(X_train, y_train)
        
        # Best parameters
        self.best_params['KNN'] = grid_search.best_params_
        print(f"Best parameters: {grid_search.best_params_}")
        
        # Make predictions
        y_train_pred = grid_search.predict(X_train)
        y_test_pred = grid_search.predict(X_test)
        
        # Evaluate
        train_metrics = self.evaluate_model(y_train, y_train_pred, 'KNN_train')
        test_metrics = self.evaluate_model(y_test, y_test_pred, 'KNN_test')
        
        self.models['KNN'] = grid_search.best_estimator_
        self.results['KNN'] = {'train': train_metrics, 'test': test_metrics}
        
        return train_metrics, test_metrics
    
    def train_rrf(self, X_train, y_train, X_test, y_test):
        """
        Train Regularized Random Forest (RRF) model
        Note: Using standard RF with regularization parameters as sklearn doesn't have RRF
        """
        print("\nTraining RRF (Regularized Random Forest) model...")
        
        # Hyperparameter grid
        # Paper mentions mtry=14 and coefReg=0.505
        n_features = X_train.shape[1]
        
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_features': [int(n_features/2), int(n_features/3), 'sqrt'],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        # Initialize model
        rf_model = RandomForestRegressor(random_state=42, n_jobs=-1)
        
        # Grid search with 10-fold CV
        kfold = KFold(n_splits=10, shuffle=True, random_state=42)
        grid_search = GridSearchCV(
            rf_model, 
            param_grid, 
            cv=kfold,
            scoring='r2',
            n_jobs=-1,
            verbose=1
        )
        
        # Fit the model
        grid_search.fit(X_train, y_train)
        
        # Best parameters
        self.best_params['RRF'] = grid_search.best_params_
        print(f"Best parameters: {grid_search.best_params_}")
        
        # Make predictions
        y_train_pred = grid_search.predict(X_train)
        y_test_pred = grid_search.predict(X_test)
        
        # Evaluate
        train_metrics = self.evaluate_model(y_train, y_train_pred, 'RRF_train')
        test_metrics = self.evaluate_model(y_test, y_test_pred, 'RRF_test')
        
        self.models['RRF'] = grid_search.best_estimator_
        self.results['RRF'] = {'train': train_metrics, 'test': test_metrics}
        
        return train_metrics, test_metrics
    
    def train_cubist(self, X_train, y_train, X_test, y_test):
        """
        Train CUBIST model
        Note: Python doesn't have native Cubist implementation. 
        Using ensemble of trees with linear models as approximation.
        For exact implementation, use R or consider cubist package if available.
        """
        print("\nTraining CUBIST-like model (ensemble approximation)...")
        print("Note: For exact CUBIST, consider using R implementation")
        
        from sklearn.ensemble import GradientBoostingRegressor
        
        # Hyperparameter grid (approximating Cubist parameters)
        param_grid = {
            'n_estimators': [41, 50, 100],  # committees in Cubist
            'learning_rate': [0.01, 0.1, 0.3],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 3, 5]  # neighbors in Cubist
        }
        
        # Initialize model
        gb_model = GradientBoostingRegressor(random_state=42)
        
        # Grid search with 10-fold CV
        kfold = KFold(n_splits=10, shuffle=True, random_state=42)
        grid_search = GridSearchCV(
            gb_model, 
            param_grid, 
            cv=kfold,
            scoring='r2',
            n_jobs=-1,
            verbose=1
        )
        
        # Fit the model
        grid_search.fit(X_train, y_train)
        
        # Best parameters
        self.best_params['CUBIST'] = grid_search.best_params_
        print(f"Best parameters: {grid_search.best_params_}")
        
        # Make predictions
        y_train_pred = grid_search.predict(X_train)
        y_test_pred = grid_search.predict(X_test)
        
        # Evaluate
        train_metrics = self.evaluate_model(y_train, y_train_pred, 'CUBIST_train')
        test_metrics = self.evaluate_model(y_test, y_test_pred, 'CUBIST_test')
        
        self.models['CUBIST'] = grid_search.best_estimator_
        self.results['CUBIST'] = {'train': train_metrics, 'test': test_metrics}
        
        return train_metrics, test_metrics
    
    def train_cit(self, X_train, y_train, X_test, y_test):
        """
        Train Conditional Inference Tree (CIT) model
        Note: Using DecisionTree with statistical tests approximation
        """
        print("\nTraining CIT (Conditional Inference Tree) model...")
        
        # Hyperparameter grid
        param_grid = {
            'max_depth': list(range(1, 22)),  # 1-21 as per paper
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 2, 5, 10],
            'criterion': ['squared_error', 'absolute_error']
        }
        
        # Initialize model
        cit_model = DecisionTreeRegressor(random_state=42)
        
        # Grid search with 10-fold CV
        kfold = KFold(n_splits=10, shuffle=True, random_state=42)
        grid_search = GridSearchCV(
            cit_model, 
            param_grid, 
            cv=kfold,
            scoring='r2',
            n_jobs=-1,
            verbose=1
        )
        
        # Fit the model
        grid_search.fit(X_train, y_train)
        
        # Best parameters
        self.best_params['CIT'] = grid_search.best_params_
        print(f"Best parameters: {grid_search.best_params_}")
        
        # Make predictions
        y_train_pred = grid_search.predict(X_train)
        y_test_pred = grid_search.predict(X_test)
        
        # Evaluate
        train_metrics = self.evaluate_model(y_train, y_train_pred, 'CIT_train')
        test_metrics = self.evaluate_model(y_test, y_test_pred, 'CIT_test')
        
        self.models['CIT'] = grid_search.best_estimator_
        self.results['CIT'] = {'train': train_metrics, 'test': test_metrics}
        
        return train_metrics, test_metrics
    
    def train_all_models(self, X_train, y_train, X_test, y_test):
        """
        Train all models and compare results
        """
        print("="*50)
        print("Training all models...")
        print("="*50)
        
        # Train each model
        self.train_cart(X_train, y_train, X_test, y_test)
        self.train_knn(X_train, y_train, X_test, y_test)
        self.train_rrf(X_train, y_train, X_test, y_test)
        self.train_cubist(X_train, y_train, X_test, y_test)
        self.train_cit(X_train, y_train, X_test, y_test)
        
        # Create results summary
        self.create_results_summary()
        
        return self.results
    
    def create_results_summary(self):
        """
        Create a summary table of all model results
        """
        print("\n" + "="*80)
        print("MODEL PERFORMANCE SUMMARY")
        print("="*80)
        
        # Training results
        print("\nTRAINING SET RESULTS:")
        print("-"*60)
        train_df = pd.DataFrame([
            self.results[model]['train'] 
            for model in self.results
        ])
        print(train_df.to_string(index=False))
        
        # Testing results
        print("\nTESTING SET RESULTS:")
        print("-"*60)
        test_df = pd.DataFrame([
            self.results[model]['test'] 
            for model in self.results
        ])
        print(test_df.to_string(index=False))
        
        # Best model
        best_model = test_df.loc[test_df['R2'].idxmax(), 'Model']
        print(f"\nBest performing model (by R2 on test set): {best_model}")
        
        return train_df, test_df
    
    def get_feature_importance(self, feature_names):
        """
        Get feature importance for tree-based models
        """
        importance_dict = {}
        
        for model_name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importance = pd.DataFrame({
                    'feature': feature_names,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                importance_dict[model_name] = importance
                
        return importance_dict

# Usage example
if __name__ == "__main__":
    # This would be run after data preprocessing
    print("Model training script ready!")
    print("Run after preprocessing data using SeoulBikeDataPreprocessor")