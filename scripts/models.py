"""
Model Training and Forecasting Module with Full GPU Acceleration.
Leverages NVIDIA GeForce RTX 5050 (CUDA) for high-performance training:
1. Zero Baseline (CPU)
2. Historical Mean Baseline (CPU)
3. Momentum Baseline (CPU)
4. OLS Linear Regression (CPU)
5. Ridge Regression (L2 regularization, CPU closed-form)
6. Random Forest Regressor (GPU-accelerated via XGBoost parallel trees on CUDA)
7. Gradient Boosted Decision Trees (GPU-accelerated via XGBoost on CUDA with early stopping)

Strictly uses StandardScaler fit on TRAIN data only. Zero look-ahead leakage.
"""

import time
import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

def fit_predict_baselines(
    X_train: pd.DataFrame, y_train: pd.Series,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Produces predictions for heuristic baselines:
    - Zero Baseline
    - Historical Mean Baseline
    - Last-Return (Momentum) Baseline
    """
    mean_val = float(y_train.mean())
    
    # 1. Zero Baseline
    pred_zero_val = np.zeros(len(X_val), dtype=np.float32)
    pred_zero_test = np.zeros(len(X_test), dtype=np.float32)
    
    # 2. Historical Mean Baseline
    pred_mean_val = np.full(len(X_val), mean_val, dtype=np.float32)
    pred_mean_test = np.full(len(X_test), mean_val, dtype=np.float32)
    
    # 3. Last Return / Momentum Baseline (using ret_5d if present, else ret_1d)
    mom_col = "ret_5d" if "ret_5d" in X_val.columns else "ret_1d"
    pred_mom_val = X_val[mom_col].values.astype(np.float32)
    pred_mom_test = X_test[mom_col].values.astype(np.float32)
    
    return {
        "BASELINE_ZERO": (pred_zero_val, pred_zero_test),
        "BASELINE_HIST_MEAN": (pred_mean_val, pred_mean_test),
        "BASELINE_MOMENTUM": (pred_mom_val, pred_mom_test)
    }

class LinearModelWrapper:
    def __init__(self, model_type="ridge", alpha=100.0):
        self.model_type = model_type
        self.alpha = alpha
        self.scaler = StandardScaler()
        if model_type == "ols":
            self.model = LinearRegression()
        else:
            self.model = Ridge(alpha=alpha, random_state=42)
            
    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        start = time.time()
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, y_train)
        self.train_time = time.time() - start
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled).astype(np.float32)

class GPURandomForestWrapper:
    """
    GPU-Accelerated Random Forest using XGBoost's parallel tree ensemble on CUDA.
    Trains B parallel trees independently with subsampling and feature subsampling.
    """
    def __init__(self, n_estimators=50, max_depth=8, subsample=0.8, colsample_bynode=0.8, random_state=42):
        self.model = xgb.XGBRegressor(
            n_estimators=1,
            num_parallel_tree=n_estimators,
            subsample=subsample,
            colsample_bynode=colsample_bynode,
            learning_rate=1.0,
            max_depth=max_depth,
            tree_method="hist",
            device="cuda",
            random_state=random_state,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        
    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        start = time.time()
        X_scaled = self.scaler.fit_transform(X_train).astype(np.float32)
        y_f32 = y_train.astype(np.float32)
        self.model.fit(X_scaled, y_f32)
        self.train_time = time.time() - start
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X).astype(np.float32)
        return self.model.predict(X_scaled).astype(np.float32)

class GPUGradientBoostingWrapper:
    """
    GPU-Accelerated Gradient Boosted Decision Trees using XGBoost on CUDA with early stopping.
    """
    def __init__(self, n_estimators=500, learning_rate=0.03, max_depth=6, random_state=42):
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            tree_method="hist",
            device="cuda",
            early_stopping_rounds=30,
            random_state=random_state,
            eval_metric="rmse",
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None):
        start = time.time()
        X_scaled = self.scaler.fit_transform(X_train).astype(np.float32)
        y_f32 = y_train.astype(np.float32)
        
        eval_set = None
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val).astype(np.float32)
            eval_set = [(X_val_scaled, y_val.astype(np.float32))]
            
        self.model.fit(
            X_scaled, y_f32,
            eval_set=eval_set,
            verbose=False
        )
        self.train_time = time.time() - start
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X).astype(np.float32)
        return self.model.predict(X_scaled).astype(np.float32)
