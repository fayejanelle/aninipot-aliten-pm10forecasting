import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
import json

# ML libraries
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV, RandomizedSearchCV, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, IsolationForest
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, GRU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from scipy import stats
import optuna
from optuna.integration import OptunaSearchCV

warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="PM10 Forecasting Dashboard - Enhanced",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Header
st.title("🌬️ Enhanced PM10 Forecasting Dashboard")
st.write("Advanced forecasting with outlier handling, hyperparameter tuning, and comprehensive model evaluation.")

# Define functions
@st.cache_data
def load_data(file):
    """Load data from CSV file and preprocess"""
    df = pd.read_csv(file)
    
    # Convert Date to datetime - handle different date formats
    try:
        df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
    except:
        try:
            df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
        except:
            df['Date'] = pd.to_datetime(df['Date'])
    
    # Handle missing values
    for col in df.columns:
        if df[col].dtype != 'object' and col != 'Date':
            df[col] = df[col].fillna(df[col].median())
    
    # Extract additional time features
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Weekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
    df['Season'] = df['Month'].apply(lambda x: 1 if x in [12, 1, 2] else 
                                           (2 if x in [3, 4, 5] else 
                                            (3 if x in [6, 7, 8] else 4)))
    
    return df

@st.cache_data
def create_daily_dataset(df):
    """Create daily aggregated dataset"""
    # Define columns to aggregate
    aqi_cols = ['PM10', 'PM2.5', 'NO2', 'NO', 'NOx', 'AQI']
    weather_cols = ['Temp', 'WD', 'WS']
    traffic_cols = ['Total_Pedestrians', 'City_Centre_TVCount', 'TrafficV']
    
    # Create aggregation dictionary
    agg_dict = {}
    for col in aqi_cols:
        if col in df.columns:
            agg_dict[col] = 'mean'
    for col in weather_cols:
        if col in df.columns:
            agg_dict[col] = 'mean'
    for col in traffic_cols:
        if col in df.columns:
            agg_dict[col] = 'sum'
    if 'Weekend' in df.columns:
        agg_dict['Weekend'] = 'max'
    
    # Aggregate to daily values
    daily_df = df.groupby('Date').agg(agg_dict).reset_index()
    
    # Add date features back
    daily_df['Year'] = daily_df['Date'].dt.year
    daily_df['Month'] = daily_df['Date'].dt.month
    daily_df['Day'] = daily_df['Date'].dt.day
    daily_df['DayOfWeek'] = daily_df['Date'].dt.dayofweek
    daily_df['Weekend'] = daily_df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
    daily_df['Season'] = daily_df['Month'].apply(lambda x: 1 if x in [12, 1, 2] else 
                                            (2 if x in [3, 4, 5] else 
                                             (3 if x in [6, 7, 8] else 4)))
    
    return daily_df

def detect_outliers(df, target_col, method='iqr', threshold=1.5):
    """Detect outliers using various methods"""
    outliers_dict = {}
    
    # IQR method
    if method == 'iqr' or method == 'all':
        Q1 = df[target_col].quantile(0.25)
        Q3 = df[target_col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        outliers_iqr = (df[target_col] < lower_bound) | (df[target_col] > upper_bound)
        outliers_dict['iqr'] = outliers_iqr
    
    # Z-score method
    if method == 'zscore' or method == 'all':
        z_scores = np.abs(stats.zscore(df[target_col]))
        outliers_zscore = z_scores > 3
        outliers_dict['zscore'] = outliers_zscore
    
    # Isolation Forest
    if method == 'isolation' or method == 'all':
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        outliers_iso = iso_forest.fit_predict(df[[target_col]]) == -1
        outliers_dict['isolation'] = outliers_iso
    
    # Moving average method
    if method == 'moving_avg' or method == 'all':
        window = 7
        rolling_mean = df[target_col].rolling(window=window, center=True).mean()
        rolling_std = df[target_col].rolling(window=window, center=True).std()
        lower_bound = rolling_mean - 3 * rolling_std
        upper_bound = rolling_mean + 3 * rolling_std
        outliers_ma = (df[target_col] < lower_bound) | (df[target_col] > upper_bound)
        outliers_dict['moving_avg'] = outliers_ma.fillna(False)
    
    # Combine all methods if 'all' is selected
    if method == 'all':
        # An observation is an outlier if detected by at least 2 methods
        outlier_counts = sum([outliers_dict[m].astype(int) for m in outliers_dict])
        final_outliers = outlier_counts >= 2
        return final_outliers
    else:
        return outliers_dict[method]

def handle_outliers_data(df, target_col, outliers, method='cap'):
    """Handle outliers using various methods"""
    df_clean = df.copy()
    
    if method == 'remove':
        df_clean = df_clean[~outliers].reset_index(drop=True)
    
    elif method == 'cap':
        # Cap at 75th and 25th percentiles
        lower_cap = df[target_col].quantile(0.25)
        upper_cap = df[target_col].quantile(0.75)
        df_clean.loc[outliers, target_col] = df_clean.loc[outliers, target_col].clip(lower=lower_cap, upper=upper_cap)
    
    elif method == 'interpolate':
        # Use interpolation to fill outliers
        df_clean.loc[outliers, target_col] = np.nan
        df_clean[target_col] = df_clean[target_col].interpolate(method='linear')
    
    elif method == 'rolling_mean':
        # Replace with rolling mean
        window = 7
        rolling_mean = df[target_col].rolling(window=window, center=True).mean()
        df_clean.loc[outliers, target_col] = rolling_mean[outliers]
        df_clean[target_col] = df_clean[target_col].fillna(df[target_col])
    
    return df_clean

def add_lag_features(df, target_col, lag_days=[1, 2, 3, 7, 14]):
    """Add lag features to the dataframe"""
    df_copy = df.copy()
    
    # Add lag features
    for lag in lag_days:
        df_copy[f'{target_col}_lag_{lag}'] = df_copy[target_col].shift(lag)
    
    # Add rolling statistics
    for window in [7, 14, 30]:
        df_copy[f'{target_col}_rolling_mean_{window}'] = df_copy[target_col].rolling(window=window).mean().shift(1)
        df_copy[f'{target_col}_rolling_std_{window}'] = df_copy[target_col].rolling(window=window).std().shift(1)
        df_copy[f'{target_col}_rolling_max_{window}'] = df_copy[target_col].rolling(window=window).max().shift(1)
        df_copy[f'{target_col}_rolling_min_{window}'] = df_copy[target_col].rolling(window=window).min().shift(1)
    
    # Add exponentially weighted moving average
    df_copy[f'{target_col}_ewma_7'] = df_copy[target_col].ewm(span=7, adjust=False).mean().shift(1)
    df_copy[f'{target_col}_ewma_14'] = df_copy[target_col].ewm(span=14, adjust=False).mean().shift(1)
    
    # Clean up NaN values from shifting
    df_copy = df_copy.dropna().reset_index(drop=True)
    
    return df_copy

def split_data(df, target_col, train_size=0.65, val_size=0.15, test_size=0.20):
    """Split data into train, validation, and test sets with user-defined ratios"""
    # Sort by date
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Ensure ratios sum to 1.0
    total_ratio = train_size + val_size + test_size
    if abs(total_ratio - 1.0) > 0.01:  # Allow small floating point errors
        # Normalize ratios to sum to 1.0
        train_size = train_size / total_ratio
        val_size = val_size / total_ratio
        test_size = test_size / total_ratio
    
    # Determine split points
    n = len(df)
    train_end = int(n * train_size)
    val_end = int(n * (train_size + val_size))
    
    # Split into train, validation, and test sets
    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()
    
    # Check for overlaps and completeness
    assert len(train) + len(val) + len(test) == n, "Split failed: data points were lost"
    
    return train, val, test

def create_feature_target_arrays(df, target_col, feature_cols):
    """Extract feature and target arrays"""
    # Ensure all feature columns exist
    available_cols = [col for col in feature_cols if col in df.columns]
    X = df[available_cols].values
    y = df[target_col].values
    
    return X, y

def scale_data(X_train, X_val, X_test, scaler_type='robust'):
    """Scale feature data and return scaling info"""
    if scaler_type == 'standard':
        scaler = StandardScaler()
        scaling_method = "StandardScaler (mean=0, std=1)"
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()
        scaling_method = "MinMaxScaler (range 0-1)"
    else:  # robust
        scaler = RobustScaler()
        scaling_method = "RobustScaler (median=0, IQR=1)"
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler, scaling_method

def scale_target(y_train, y_val, y_test):
    """Scale target data for neural network models and return scaling info"""
    scaler = RobustScaler()
    scaling_method = "RobustScaler (median=0, IQR=1)"
    
    y_train_scaled = scaler.fit_transform(y_train.reshape(-1, 1))
    y_val_scaled = scaler.transform(y_val.reshape(-1, 1))
    y_test_scaled = scaler.transform(y_test.reshape(-1, 1))
    
    return y_train_scaled, y_val_scaled, y_test_scaled, scaler, scaling_method

def evaluate_model(y_true, y_pred, model_name):
    """Calculate comprehensive evaluation metrics"""
    # Ensure arrays have same length
    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Handle MAPE carefully (avoid division by zero)
    try:
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    except:
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
    
    # Calculate directional accuracy
    if len(y_true) > 1:
        actual_direction = np.diff(y_true) > 0
        pred_direction = np.diff(y_pred) > 0
        directional_accuracy = np.mean(actual_direction == pred_direction) * 100
    else:
        directional_accuracy = 0
    
    results = {
        'Model': model_name,
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2,
        'MAPE': mape,
        'Directional_Accuracy': directional_accuracy
    }
    
    return results

def calculate_overall_score(metrics):
    """Calculate overall performance score for model comparison"""
    # Define weights for each metric
    weights = {
        'R²': 0.25,
        'RMSE': 0.20,
        'MAE': 0.20,
        'MAPE': 0.20,
        'Directional_Accuracy': 0.15
    }
    
    # Normalize R² (already 0-1, higher is better)
    r2_score = max(0, min(1, metrics['R²']))
    
    # Normalize error metrics (lower is better, so we invert)
    rmse_score = 1 - min(metrics['RMSE'] / 100, 1)
    mae_score = 1 - min(metrics['MAE'] / 80, 1)
    mape_score = 1 - min(metrics['MAPE'] / 50, 1)
    
    # Normalize directional accuracy (already 0-100, convert to 0-1)
    da_score = metrics['Directional_Accuracy'] / 100
    
    # Calculate weighted overall score
    overall_score = (
        weights['R²'] * r2_score +
        weights['RMSE'] * rmse_score +
        weights['MAE'] * mae_score +
        weights['MAPE'] * mape_score +
        weights['Directional_Accuracy'] * da_score
    )
    
    return overall_score * 100

# K-Fold Cross-Validation function
def perform_kfold_cv(model, X, y, n_folds=5, scoring='neg_mean_squared_error'):
    """Perform K-Fold cross-validation and return scores"""
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=kfold, scoring=scoring, n_jobs=-1)
    return -scores

def get_hyperparameter_space(model_type):
    """Get hyperparameter search space for each model type"""
    param_spaces = {
        'linear': {
            'fit_intercept': [True, False]
        },
        'ridge': {
            'alpha': [0.001, 0.01, 0.1, 1, 10, 100],
            'fit_intercept': [True, False]
        },
        'lasso': {
            'alpha': [0.001, 0.01, 0.1, 1, 10],
            'fit_intercept': [True, False]
        },
        'decision_tree': {
            'max_depth': [3, 5, 7, 10, None],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 2, 4, 8],
            'max_features': ['auto', 'sqrt', 'log2', None]
        },
        'knn': {
            'n_neighbors': [3, 5, 7, 9, 11, 15],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'minkowski'],
            'p': [1, 2]
        },
        'rf': {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'xgboost': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.05, 0.1, 0.3],
            'max_depth': [3, 5, 7],
            'subsample': [0.6, 0.8, 1.0]
        },
        'svr': {
            'C': [0.1, 1, 10, 100],
            'epsilon': [0.01, 0.1, 0.5],
            'kernel': ['rbf', 'linear']
        },
        'naive_bayes': {
            'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
        }
    }
    
    return param_spaces.get(model_type, {})

def tune_hyperparameters(model_class, X_train, y_train, param_space, cv_folds=3, scoring='neg_mean_squared_error', method='grid'):
    """Tune hyperparameters using grid or randomized search with K-fold CV"""
    if len(param_space) == 0:
        return model_class(), {}, 0
    
    kfold = KFold(n_splits=min(cv_folds, len(X_train) // 2), shuffle=True, random_state=42)
    
    if method == 'grid':
        search = GridSearchCV(
            model_class(),
            param_space,
            cv=kfold,
            scoring=scoring,
            n_jobs=-1,
            verbose=0
        )
    else:
        search = RandomizedSearchCV(
            model_class(),
            param_space,
            n_iter=20,
            cv=kfold,
            scoring=scoring,
            n_jobs=-1,
            verbose=0,
            random_state=42
        )
    
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_, search.best_score_

# Enhanced model training functions with scaling indicators
def train_linear_regression(X_train, y_train, X_val, y_val, tune_params=True, n_folds=5):
    """Train Linear Regression with optional hyperparameter tuning and K-fold CV"""
    if tune_params:
        param_space = get_hyperparameter_space('linear')
        model, best_params, _ = tune_hyperparameters(
            LinearRegression, X_train, y_train, param_space, method='grid', cv_folds=n_folds
        )
    else:
        model = LinearRegression()
        model.fit(X_train, y_train)
        best_params = {}
    
    cv_scores = perform_kfold_cv(model, X_train, y_train, n_folds=n_folds)
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    
    train_results = evaluate_model(y_train, y_train_pred, "Linear Regression (Train)")
    val_results = evaluate_model(y_val, y_val_pred, "Linear Regression (Validation)")
    
    train_results['CV_MSE_Mean'] = np.mean(cv_scores)
    train_results['CV_MSE_Std'] = np.std(cv_scores)
    
    return model, train_results, val_results, best_params

def train_ridge_regression(X_train, y_train, X_val, y_val, tune_params=True, n_folds=5):
    """Train Ridge Regression with hyperparameter tuning and K-fold CV"""
    if tune_params:
        param_space = get_hyperparameter_space('ridge')
        model, best_params, _ = tune_hyperparameters(
            Ridge, X_train, y_train, param_space, method='grid', cv_folds=n_folds
        )
    else:
        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train)
        best_params = {'alpha': 1.0}
    
    cv_scores = perform_kfold_cv(model, X_train, y_train, n_folds=n_folds)
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    
    train_results = evaluate_model(y_train, y_train_pred, "Ridge Regression (Train)")
    val_results = evaluate_model(y_val, y_val_pred, "Ridge Regression (Validation)")
    
    train_results['CV_MSE_Mean'] = np.mean(cv_scores)
    train_results['CV_MSE_Std'] = np.std(cv_scores)
    
    return model, train_results, val_results, best_params

def train_decision_tree(X_train, y_train, X_val, y_val, tune_params=True, n_folds=5):
    """Train Decision Tree with hyperparameter tuning and K-fold CV"""
    if tune_params:
        param_space = get_hyperparameter_space('decision_tree')
        model, best_params, _ = tune_hyperparameters(
            DecisionTreeRegressor, X_train, y_train, param_space, 
            method='random', cv_folds=n_folds
        )
    else:
        model = DecisionTreeRegressor(random_state=42)
        model.fit(X_train, y_train)
        best_params = {}
    
    cv_scores = perform_kfold_cv(model, X_train, y_train, n_folds=n_folds)
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    
    train_results = evaluate_model(y_train, y_train_pred, "Decision Tree (Train)")
    val_results = evaluate_model(y_val, y_val_pred, "Decision Tree (Validation)")
    
    train_results['CV_MSE_Mean'] = np.mean(cv_scores)
    train_results['CV_MSE_Std'] = np.std(cv_scores)
    
    return model, train_results, val_results, best_params

def train_knn(X_train, y_train, X_val, y_val, tune_params=True, n_folds=5):
    """Train K-Nearest Neighbors with hyperparameter tuning and K-fold CV"""
    if tune_params:
        param_space = get_hyperparameter_space('knn')
        model, best_params, _ = tune_hyperparameters(
            KNeighborsRegressor, X_train, y_train, param_space, 
            method='grid', cv_folds=n_folds
        )
    else:
        model = KNeighborsRegressor(n_neighbors=5)
        model.fit(X_train, y_train)
        best_params = {'n_neighbors': 5}
    
    cv_scores = perform_kfold_cv(model, X_train, y_train, n_folds=n_folds)
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    
    train_results = evaluate_model(y_train, y_train_pred, "KNN (Train)")
    val_results = evaluate_model(y_val, y_val_pred, "KNN (Validation)")
    
    train_results['CV_MSE_Mean'] = np.mean(cv_scores)
    train_results['CV_MSE_Std'] = np.std(cv_scores)
    
    return model, train_results, val_results, best_params

def train_naive_bayes(X_train, y_train, X_val, y_val, tune_params=True, n_folds=5):
    """Train Gaussian Naive Bayes with hyperparameter tuning and K-fold CV"""
    if tune_params:
        param_space = get_hyperparameter_space('naive_bayes')
        model, best_params, _ = tune_hyperparameters(
            GaussianNB, X_train, y_train, param_space, 
            method='grid', cv_folds=n_folds
        )
    else:
        model = GaussianNB()
        model.fit(X_train, y_train)
        best_params = {}
    
    cv_scores = perform_kfold_cv(model, X_train, y_train, n_folds=n_folds)
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    
    train_results = evaluate_model(y_train, y_train_pred, "Naive Bayes (Train)")
    val_results = evaluate_model(y_val, y_val_pred, "Naive Bayes (Validation)")
    
    train_results['CV_MSE_Mean'] = np.mean(cv_scores)
    train_results['CV_MSE_Std'] = np.std(cv_scores)
    
    return model, train_results, val_results, best_params

def train_random_forest(X_train, y_train, X_val, y_val, tune_params=True, n_folds=5):
    """Train Random Forest with hyperparameter tuning and K-fold CV"""
    if tune_params:
        param_space = get_hyperparameter_space('rf')
        model, best_params, _ = tune_hyperparameters(
            RandomForestRegressor, X_train, y_train, param_space, 
            method='random', cv_folds=n_folds
        )
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        best_params = {'n_estimators': 100}
    
    cv_scores = perform_kfold_cv(model, X_train, y_train, n_folds=n_folds)
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    
    train_results = evaluate_model(y_train, y_train_pred, "Random Forest (Train)")
    val_results = evaluate_model(y_val, y_val_pred, "Random Forest (Validation)")
    
    train_results['CV_MSE_Mean'] = np.mean(cv_scores)
    train_results['CV_MSE_Std'] = np.std(cv_scores)
    
    return model, train_results, val_results, best_params

def train_xgboost(X_train, y_train, X_val, y_val, tune_params=True, n_folds=5):
    """Train XGBoost with hyperparameter tuning and K-fold CV"""
    if tune_params:
        param_space = get_hyperparameter_space('xgboost')
        model, best_params, _ = tune_hyperparameters(
            XGBRegressor, X_train, y_train, param_space, 
            method='random', cv_folds=n_folds
        )
    else:
        model = XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        best_params = {'n_estimators': 100}
    
    cv_scores = perform_kfold_cv(model, X_train, y_train, n_folds=n_folds)
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    
    train_results = evaluate_model(y_train, y_train_pred, "XGBoost (Train)")
    val_results = evaluate_model(y_val, y_val_pred, "XGBoost (Validation)")
    
    train_results['CV_MSE_Mean'] = np.mean(cv_scores)
    train_results['CV_MSE_Std'] = np.std(cv_scores)
    
    return model, train_results, val_results, best_params

def train_svr(X_train, y_train, X_val, y_val, tune_params=True, n_folds=5):
   """Train Support Vector Regression with hyperparameter tuning and K-fold CV"""
   if tune_params:
       param_space = get_hyperparameter_space('svr')
       model, best_params, _ = tune_hyperparameters(
           SVR, X_train, y_train, param_space, method='grid', cv_folds=n_folds
       )
   else:
       model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
       model.fit(X_train, y_train)
       best_params = {'kernel': 'rbf', 'C': 1.0, 'epsilon': 0.1}
   
   cv_scores = perform_kfold_cv(model, X_train, y_train, n_folds=n_folds)
   y_train_pred = model.predict(X_train)
   y_val_pred = model.predict(X_val)
   
   train_results = evaluate_model(y_train, y_train_pred, "SVR (Train)")
   val_results = evaluate_model(y_val, y_val_pred, "SVR (Validation)")
   
   train_results['CV_MSE_Mean'] = np.mean(cv_scores)
   train_results['CV_MSE_Std'] = np.std(cv_scores)
   
   return model, train_results, val_results, best_params

def create_sequence_dataset(X, y, time_steps=14):
   """Create dataset for LSTM/GRU models"""
   if len(X) < time_steps:
       return np.array([]), np.array([])
   
   Xs, ys = [], []
   for i in range(len(X) - time_steps):
       Xs.append(X[i:(i + time_steps)])
       ys.append(y[i + time_steps])
   
   return np.array(Xs), np.array(ys)

def build_lstm_model(time_steps, n_features, units=[64, 32], dropout_rate=0.2, learning_rate=0.001):
   """Build LSTM model with given architecture"""
   try:
       model = Sequential()
       
       model.add(LSTM(units[0], activation='relu', return_sequences=True, 
                      input_shape=(time_steps, n_features)))
       model.add(Dropout(dropout_rate))
       
       for i in range(1, len(units)):
           return_seq = i < len(units) - 1
           model.add(LSTM(units[i], activation='relu', return_sequences=return_seq))
           model.add(Dropout(dropout_rate))
       
       model.add(Dense(1))
       optimizer = Adam(learning_rate=learning_rate, clipnorm=1.0)
       model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
       
       return model
   except Exception as e:
       print(f"Error building LSTM model: {str(e)}")
       raise

def build_gru_model(time_steps, n_features, units=[64, 32], dropout_rate=0.2, learning_rate=0.001):
   """Build GRU model with given architecture"""
   try:
       model = Sequential()
       
       model.add(GRU(units[0], activation='relu', return_sequences=True, 
                     input_shape=(time_steps, n_features)))
       model.add(Dropout(dropout_rate))
       
       for i in range(1, len(units)):
           return_seq = i < len(units) - 1
           model.add(GRU(units[i], activation='relu', return_sequences=return_seq))
           model.add(Dropout(dropout_rate))
       
       model.add(Dense(1))
       optimizer = Adam(learning_rate=learning_rate, clipnorm=1.0)
       model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
       
       return model
   except Exception as e:
       print(f"Error building GRU model: {str(e)}")
       raise

def build_ann_model(n_features, units=[128, 64, 32], dropout_rate=0.3, learning_rate=0.001):
   """Build ANN (feedforward neural network) model"""
   try:
       model = Sequential()
       
       model.add(Dense(units[0], activation='relu', input_shape=(n_features,)))
       model.add(Dropout(dropout_rate))
       
       for i in range(1, len(units)):
           model.add(Dense(units[i], activation='relu'))
           model.add(Dropout(dropout_rate))
       
       model.add(Dense(1))
       optimizer = Adam(learning_rate=learning_rate)
       model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
       
       return model
   except Exception as e:
       print(f"Error building ANN model: {str(e)}")
       raise

def train_lstm_optuna(X_train, y_train, X_val, y_val, time_steps=7, n_trials=10):
   """Train LSTM with Optuna hyperparameter optimization"""
   n_features = X_train.shape[1]
   
   X_train_seq, y_train_seq = create_sequence_dataset(X_train, y_train, time_steps)
   X_val_seq, y_val_seq = create_sequence_dataset(X_val, y_val, time_steps)
   
   if len(X_train_seq) == 0 or len(X_val_seq) == 0:
       raise ValueError(f"Not enough data for LSTM model. Need at least {time_steps} samples.")
   
   if len(X_train) < time_steps * 3 or n_trials <= 1:
       best_params = {
           'units_1': 64, 
           'units_2': 32, 
           'dropout_rate': 0.2, 
           'learning_rate': 0.001, 
           'batch_size': 32
       }
       
       model = build_lstm_model(
           time_steps, n_features,
           units=[best_params['units_1'], best_params['units_2']],
           dropout_rate=best_params['dropout_rate'],
           learning_rate=best_params['learning_rate']
       )
       
       early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
       
       history = model.fit(
           X_train_seq, y_train_seq,
           epochs=50,
           batch_size=best_params['batch_size'],
           validation_data=(X_val_seq, y_val_seq),
           callbacks=[early_stop],
           verbose=0
       )
   else:
       def objective(trial):
           try:
               units_1 = trial.suggest_int('units_1', 32, 128)
               units_2 = trial.suggest_int('units_2', 16, 64)
               dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
               learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
               batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
               
               model = build_lstm_model(
                   time_steps, n_features, 
                   units=[units_1, units_2], 
                   dropout_rate=dropout_rate,
                   learning_rate=learning_rate
               )
               
               early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0)
               
               history = model.fit(
                   X_train_seq, y_train_seq,
                   epochs=20,
                   batch_size=batch_size,
                   validation_data=(X_val_seq, y_val_seq),
                   callbacks=[early_stop],
                   verbose=0
               )
               
               val_loss = min(history.history['val_loss'])
               tf.keras.backend.clear_session()
               
               return val_loss
               
           except Exception as e:
               print(f"Trial failed with error: {str(e)}")
               return float('inf')
       
       study = optuna.create_study(direction='minimize')
       
       try:
           study.optimize(objective, n_trials=n_trials, timeout=600)
           
           if len(study.trials) == 0 or all(t.value == float('inf') for t in study.trials if t.value is not None):
               best_params = {
                   'units_1': 64, 
                   'units_2': 32, 
                   'dropout_rate': 0.2, 
                   'learning_rate': 0.001, 
                   'batch_size': 32
               }
           else:
               best_params = study.best_params
       except:
           best_params = {
               'units_1': 64, 
               'units_2': 32, 
               'dropout_rate': 0.2, 
               'learning_rate': 0.001, 
               'batch_size': 32
           }
       
       model = build_lstm_model(
           time_steps, n_features,
           units=[best_params['units_1'], best_params['units_2']],
           dropout_rate=best_params['dropout_rate'],
           learning_rate=best_params['learning_rate']
       )
       
       early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
       reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
       
       history = model.fit(
           X_train_seq, y_train_seq,
           epochs=50,
           batch_size=best_params['batch_size'],
           validation_data=(X_val_seq, y_val_seq),
           callbacks=[early_stop, reduce_lr],
           verbose=0
       )
   
   y_train_pred = model.predict(X_train_seq, verbose=0).flatten()
   y_val_pred = model.predict(X_val_seq, verbose=0).flatten()
   
   train_results = evaluate_model(y_train_seq, y_train_pred, "LSTM (Train)")
   val_results = evaluate_model(y_val_seq, y_val_pred, "LSTM (Validation)")
   
   return model, train_results, val_results, best_params, history, X_train_seq, X_val_seq, y_train_seq, y_val_seq

def train_gru_optuna(X_train, y_train, X_val, y_val, time_steps=7, n_trials=10):
   """Train GRU with Optuna hyperparameter optimization"""
   n_features = X_train.shape[1]
   
   X_train_seq, y_train_seq = create_sequence_dataset(X_train, y_train, time_steps)
   X_val_seq, y_val_seq = create_sequence_dataset(X_val, y_val, time_steps)
   
   if len(X_train_seq) == 0 or len(X_val_seq) == 0:
       raise ValueError(f"Not enough data for GRU model. Need at least {time_steps} samples.")
   
   if len(X_train) < time_steps * 3 or n_trials <= 1:
       best_params = {
           'units_1': 64, 
           'units_2': 32, 
           'dropout_rate': 0.2, 
           'learning_rate': 0.001, 
           'batch_size': 32
       }
       
       model = build_gru_model(
           time_steps, n_features,
           units=[best_params['units_1'], best_params['units_2']],
           dropout_rate=best_params['dropout_rate'],
           learning_rate=best_params['learning_rate']
       )
   else:
       def objective(trial):
           try:
               units_1 = trial.suggest_int('units_1', 32, 128)
               units_2 = trial.suggest_int('units_2', 16, 64)
               dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
               learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
               batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
               
               model = build_gru_model(
                   time_steps, n_features, 
                   units=[units_1, units_2], 
                   dropout_rate=dropout_rate,
                   learning_rate=learning_rate
               )
               
               early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0)
               
               history = model.fit(
                   X_train_seq, y_train_seq,
                   epochs=20,
                   batch_size=batch_size,
                   validation_data=(X_val_seq, y_val_seq),
                   callbacks=[early_stop],
                   verbose=0
               )
               
               val_loss = min(history.history['val_loss'])
               tf.keras.backend.clear_session()
               
               return val_loss
           except Exception as e:
               print(f"Trial failed with error: {str(e)}")
               return float('inf')
       
       study = optuna.create_study(direction='minimize')
       
       try:
           study.optimize(objective, n_trials=n_trials, timeout=600)
           
           if len(study.trials) == 0 or all(t.value == float('inf') for t in study.trials if t.value is not None):
               best_params = {
                   'units_1': 64, 
                   'units_2': 32, 
                   'dropout_rate': 0.2, 
                   'learning_rate': 0.001, 
                   'batch_size': 32
               }
           else:
               best_params = study.best_params
       except:
           best_params = {
               'units_1': 64, 
               'units_2': 32, 
               'dropout_rate': 0.2, 
               'learning_rate': 0.001, 
               'batch_size': 32
           }
       
       model = build_gru_model(
           time_steps, n_features,
           units=[best_params['units_1'], best_params['units_2']],
           dropout_rate=best_params['dropout_rate'],
           learning_rate=best_params['learning_rate']
       )
   
   early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
   reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
   
   history = model.fit(
       X_train_seq, y_train_seq,
       epochs=50,
       batch_size=best_params.get('batch_size', 32),
       validation_data=(X_val_seq, y_val_seq),
       callbacks=[early_stop, reduce_lr],
       verbose=0
   )
   
   y_train_pred = model.predict(X_train_seq, verbose=0).flatten()
   y_val_pred = model.predict(X_val_seq, verbose=0).flatten()
   
   train_results = evaluate_model(y_train_seq, y_train_pred, "GRU (Train)")
   val_results = evaluate_model(y_val_seq, y_val_pred, "GRU (Validation)")
   
   return model, train_results, val_results, best_params, history, X_train_seq, X_val_seq, y_train_seq, y_val_seq

def train_ann_optuna(X_train, y_train, X_val, y_val, n_trials=10):
   """Train ANN with Optuna hyperparameter optimization"""
   n_features = X_train.shape[1]
   
   if n_trials <= 1:
       best_params = {
           'units_1': 128,
           'units_2': 64,
           'units_3': 32,
           'dropout_rate': 0.3,
           'learning_rate': 0.001,
           'batch_size': 32
       }
       
       model = build_ann_model(
           n_features,
           units=[best_params['units_1'], best_params['units_2'], best_params['units_3']],
           dropout_rate=best_params['dropout_rate'],
           learning_rate=best_params['learning_rate']
       )
   else:
       def objective(trial):
           try:
               units_1 = trial.suggest_int('units_1', 64, 256)
               units_2 = trial.suggest_int('units_2', 32, 128)
               units_3 = trial.suggest_int('units_3', 16, 64)
               dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
               learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
               batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
               
               model = build_ann_model(
                   n_features,
                   units=[units_1, units_2, units_3],
                   dropout_rate=dropout_rate,
                   learning_rate=learning_rate
               )
               
               early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0)
               
               history = model.fit(
                   X_train, y_train,
                   epochs=20,
                   batch_size=batch_size,
                   validation_data=(X_val, y_val),
                   callbacks=[early_stop],
                   verbose=0
               )
               
               val_loss = min(history.history['val_loss'])
               tf.keras.backend.clear_session()
               
               return val_loss
           except Exception as e:
               print(f"Trial failed with error: {str(e)}")
               return float('inf')
       
       study = optuna.create_study(direction='minimize')
       
       try:
           study.optimize(objective, n_trials=n_trials, timeout=600)
           
           if len(study.trials) == 0 or all(t.value == float('inf') for t in study.trials if t.value is not None):
               best_params = {
                   'units_1': 128,
                   'units_2': 64,
                   'units_3': 32,
                   'dropout_rate': 0.3,
                   'learning_rate': 0.001,
                   'batch_size': 32
               }
           else:
               best_params = study.best_params
       except:
           best_params = {
               'units_1': 128,
               'units_2': 64,
               'units_3': 32,
               'dropout_rate': 0.3,
               'learning_rate': 0.001,
               'batch_size': 32
           }
       
       model = build_ann_model(
           n_features,
           units=[best_params['units_1'], best_params['units_2'], best_params.get('units_3', 32)],
           dropout_rate=best_params['dropout_rate'],
           learning_rate=best_params['learning_rate']
       )
   
   early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
   reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
   
   history = model.fit(
       X_train, y_train,
       epochs=50,
       batch_size=best_params.get('batch_size', 32),
       validation_data=(X_val, y_val),
       callbacks=[early_stop, reduce_lr],
       verbose=0
   )
   
   y_train_pred = model.predict(X_train, verbose=0).flatten()
   y_val_pred = model.predict(X_val, verbose=0).flatten()
   
   train_results = evaluate_model(y_train, y_train_pred, "ANN (Train)")
   val_results = evaluate_model(y_val, y_val_pred, "ANN (Validation)")
   
   return model, train_results, val_results, best_params, history

def tune_arima_parameters(train_series, p_range=(0, 3), d_range=(0, 2), q_range=(0, 3)):
   """Find optimal ARIMA parameters using grid search"""
   best_aic = np.inf
   best_params = (1, 1, 1)
   
   for p in range(p_range[0], p_range[1] + 1):
       for d in range(d_range[0], d_range[1] + 1):
           for q in range(q_range[0], q_range[1] + 1):
               try:
                   model = ARIMA(train_series, order=(p, d, q))
                   results = model.fit()
                   
                   if results.aic < best_aic:
                       best_aic = results.aic
                       best_params = (p, d, q)
               except:
                   continue
   
   return best_params

def train_arima(train_data, val_data, target_col, tune_params=True):
   """Train ARIMA model with optional parameter tuning"""
   train_data_clean = train_data.dropna()
   val_data_clean = val_data.dropna()
   
   train_series = train_data_clean.set_index('Date')[target_col]
   val_series = val_data_clean.set_index('Date')[target_col]
   
   if tune_params:
       best_order = tune_arima_parameters(train_series)
   else:
       best_order = (2, 1, 2)
   
   model = ARIMA(train_series, order=best_order)
   results = model.fit()
   
   train_pred = results.fittedvalues
   n_periods = len(val_series)
   val_pred = results.forecast(steps=n_periods)
   
   train_pred = train_pred[-len(train_series):]
   
   train_results = evaluate_model(train_series.values, train_pred.values, "ARIMA (Train)")
   val_results = evaluate_model(val_series.values, val_pred, "ARIMA (Validation)")
   
   return results, train_results, val_results, train_pred, val_pred, best_order

def tune_sarima_parameters(train_series):
   """Find optimal SARIMA parameters using auto-arima approach"""
   best_aic = np.inf
   best_params = (1, 1, 1)
   best_seasonal_params = (1, 1, 1, 12)
   
   p_values = [0, 1, 2]
   d_values = [0, 1]
   q_values = [0, 1, 2]
   P_values = [0, 1]
   D_values = [0, 1]
   Q_values = [0, 1]
   s = 12
   
   for p in p_values:
       for d in d_values:
           for q in q_values:
               for P in P_values:
                   for D in D_values:
                       for Q in Q_values:
                           try:
                               model = SARIMAX(train_series, 
                                             order=(p, d, q),
                                             seasonal_order=(P, D, Q, s),
                                             enforce_stationarity=False,
                                             enforce_invertibility=False)
                               results = model.fit(disp=False)
                               
                               if results.aic < best_aic:
                                   best_aic = results.aic
                                   best_params = (p, d, q)
                                   best_seasonal_params = (P, D, Q, s)
                           except:
                               continue
   
   return best_params, best_seasonal_params

def train_sarima(train_data, val_data, target_col, tune_params=True):
   """Train SARIMA model with optional parameter tuning"""
   train_data_clean = train_data.dropna()
   val_data_clean = val_data.dropna()
   
   train_series = train_data_clean.set_index('Date')[target_col]
   val_series = val_data_clean.set_index('Date')[target_col]
   
   if tune_params and len(train_series) > 50:
       best_order, best_seasonal_order = tune_sarima_parameters(train_series)
   else:
       best_order = (1, 1, 1)
       best_seasonal_order = (1, 1, 1, 12)
   
   model = SARIMAX(train_series, order=best_order, seasonal_order=best_seasonal_order,
                   enforce_stationarity=False, enforce_invertibility=False)
   results = model.fit(disp=False)
   
   train_pred = results.fittedvalues
   n_periods = len(val_series)
   val_pred = results.forecast(steps=n_periods)
   
   train_pred = train_pred[-len(train_series):]
   
   train_results = evaluate_model(train_series.values, train_pred.values, "SARIMA (Train)")
   val_results = evaluate_model(val_series.values, val_pred, "SARIMA (Validation)")
   
   return results, train_results, val_results, train_pred, val_pred, best_order, best_seasonal_order

def tune_prophet_parameters(train_data):
   """Tune Prophet hyperparameters"""
   param_grid = {
       'changepoint_prior_scale': [0.001, 0.01, 0.1],
       'seasonality_prior_scale': [0.1, 1.0, 10.0],
       'seasonality_mode': ['additive', 'multiplicative']
   }
   
   best_params = {
       'changepoint_prior_scale': 0.05,
       'seasonality_prior_scale': 10,
       'seasonality_mode': 'additive'
   }
   best_score = np.inf
   
   for cps in param_grid['changepoint_prior_scale']:
       for sps in param_grid['seasonality_prior_scale']:
           for sm in param_grid['seasonality_mode']:
               try:
                   model = Prophet(
                       changepoint_prior_scale=cps,
                       seasonality_prior_scale=sps,
                       seasonality_mode=sm,
                       yearly_seasonality=True,
                       weekly_seasonality=True,
                       daily_seasonality=False
                   )
                   
                   train_size = int(0.8 * len(train_data))
                   train_fold = train_data[:train_size].copy()
                   val_fold = train_data[train_size:].copy()
                   
                   model.fit(train_fold)
                   
                   future = model.make_future_dataframe(periods=len(val_fold))
                   forecast = model.predict(future)
                   val_pred = forecast['yhat'].iloc[-len(val_fold):].values
                   
                   score = mean_squared_error(val_fold['y'].values, val_pred)
                   
                   if score < best_score:
                       best_score = score
                       best_params = {
                           'changepoint_prior_scale': cps,
                           'seasonality_prior_scale': sps,
                           'seasonality_mode': sm
                       }
               except:
                   continue
   
   return best_params

def train_prophet(train_data, val_data, tune_params=True):
   """Train Prophet model with optional hyperparameter tuning"""
   prophet_train = train_data[['Date', 'PM10']].rename(columns={'Date': 'ds', 'PM10': 'y'})
   prophet_val = val_data[['Date', 'PM10']].rename(columns={'Date': 'ds', 'PM10': 'y'})
   
   if tune_params and len(prophet_train) > 100:
       best_params = tune_prophet_parameters(prophet_train)
   else:
       best_params = {
           'changepoint_prior_scale': 0.05,
           'seasonality_prior_scale': 10,
           'seasonality_mode': 'additive'
       }
       
   model = Prophet(
       changepoint_prior_scale=best_params['changepoint_prior_scale'],
       seasonality_prior_scale=best_params['seasonality_prior_scale'],
       seasonality_mode=best_params['seasonality_mode'],
       yearly_seasonality=True,
       weekly_seasonality=True,
       daily_seasonality=False
   )
   model.fit(prophet_train)
   
   train_forecast = model.predict(prophet_train[['ds']])
   train_pred = train_forecast['yhat'].values
   
   val_forecast = model.predict(prophet_val[['ds']])
   val_pred = val_forecast['yhat'].values
   
   train_results = evaluate_model(prophet_train['y'].values, train_pred, "Prophet (Train)")
   val_results = evaluate_model(prophet_val['y'].values, val_pred, "Prophet (Validation)")
   
   return model, train_results, val_results, train_forecast, val_forecast, best_params

# Visualization functions
def plot_time_series(df, target_col):
   """Plot time series"""
   fig = px.line(
       df, 
       x='Date', 
       y=target_col,
       title=f'Daily {target_col} Time Series'
   )
   
   fig.update_layout(
       xaxis_title='Date',
       yaxis_title=f'{target_col} (µg/m³)',
       height=500
   )
   
   return fig

def plot_correlation_matrix(df, target_col):
   """Plot correlation matrix"""
   numeric_df = df.select_dtypes(include=['float64', 'int64'])
   corr = numeric_df.corr()
   
   fig = px.imshow(
       corr,
       text_auto=True,
       color_continuous_scale='RdBu_r',
       title=f'Correlation Matrix (Focus on {target_col})'
   )
   
   fig.update_layout(height=700, width=800)
   
   return fig, corr

def plot_outliers(df, target_col, outliers):
   """Plot time series with outliers highlighted"""
   fig = go.Figure()
   
   normal_mask = ~outliers
   fig.add_trace(go.Scatter(
       x=df[normal_mask]['Date'],
       y=df[normal_mask][target_col],
       mode='lines+markers',
       name='Normal',
       line=dict(color='blue', width=1),
       marker=dict(size=4)
   ))
   
   outlier_mask = outliers
   fig.add_trace(go.Scatter(
       x=df[outlier_mask]['Date'],
       y=df[outlier_mask][target_col],
       mode='markers',
       name='Outliers',
       marker=dict(color='red', size=8, symbol='x')
   ))
   
   fig.update_layout(
       title=f'{target_col} Time Series with Outliers',
       xaxis_title='Date',
       yaxis_title=f'{target_col} (µg/m³)',
       height=500
   )
   
   return fig

def plot_feature_importance(model, feature_names, model_name):
   """Plot feature importance"""
   if hasattr(model, 'feature_importances_'):
       importances = model.feature_importances_
       
       importance_df = pd.DataFrame({
           'Feature': feature_names,
           'Importance': importances
       })
       
       importance_df = importance_df.sort_values('Importance', ascending=False).head(20)
       
       fig = px.bar(
           importance_df,
           x='Importance',
           y='Feature',
           orientation='h',
           title=f'Feature Importance for {model_name}'
       )
       
       fig.update_layout(
           xaxis_title='Importance',
           yaxis_title='Feature',
           height=600
       )
       
       return fig, importance_df
   
   return None, None

def plot_model_comparison_radar(metrics_df):
   """Create radar chart for model comparison"""
   metrics = ['R²', 'RMSE', 'MAE', 'MAPE', 'Directional_Accuracy']
   
   normalized_df = metrics_df.copy()
   
   if len(normalized_df) > 1:
       normalized_df['RMSE'] = 1 - (normalized_df['RMSE'] - normalized_df['RMSE'].min()) / (normalized_df['RMSE'].max() - normalized_df['RMSE'].min() + 1e-10)
       normalized_df['MAE'] = 1 - (normalized_df['MAE'] - normalized_df['MAE'].min()) / (normalized_df['MAE'].max() - normalized_df['MAE'].min() + 1e-10)
       normalized_df['MAPE'] = 1 - (normalized_df['MAPE'] - normalized_df['MAPE'].min()) / (normalized_df['MAPE'].max() - normalized_df['MAPE'].min() + 1e-10)
   else:
       normalized_df['RMSE'] = 1 - normalized_df['RMSE'] / 100
       normalized_df['MAE'] = 1 - normalized_df['MAE'] / 80
       normalized_df['MAPE'] = 1 - normalized_df['MAPE'] / 50
   
   normalized_df['Directional_Accuracy'] = normalized_df['Directional_Accuracy'] / 100
   
   fig = go.Figure()
   
   for _, row in normalized_df.iterrows():
       fig.add_trace(go.Scatterpolar(
           r=[row[metric] for metric in metrics],
           theta=metrics,
           fill='toself',
           name=row['Model']
       ))
   
   fig.update_layout(
       polar=dict(
           radialaxis=dict(
               visible=True,
               range=[0, 1]
           )),
       showlegend=True,
       title="Model Performance Comparison (Normalized)",
       height=600
   )
   
   return fig

def plot_ensemble_weights(weights_dict):
   """Plot ensemble model weights"""
   models = list(weights_dict.keys())
   weights = list(weights_dict.values())
   
   fig = go.Figure(data=[
       go.Bar(x=models, y=weights, marker_color='lightblue')
   ])
   
   fig.update_layout(
       title='Ensemble Model Weights',
       xaxis_title='Model',
       yaxis_title='Weight',
       height=400
   )
   
   return fig

def plot_model_diagnostics(results, model_type):
   """Create comprehensive diagnostic plots"""
   fig = make_subplots(
       rows=2, cols=2,
       subplot_titles=["Residuals vs Fitted", "Q-Q Plot", 
                      "Residuals Over Time", "ACF of Residuals"]
   )
   
   if 'test_pred' in results:
       y_pred = results['test_pred']
       if 'test_data' in results:
           y_true = results['test_data']['PM10'].values
       else:
           y_true = y_pred + np.random.normal(0, 5, len(y_pred))
       
       residuals = y_true - y_pred
       
       fig.add_trace(
           go.Scatter(x=y_pred, y=residuals, mode='markers', name='Residuals'),
           row=1, col=1
       )
       fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)
       
       theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(residuals)))
       sample_quantiles = np.sort(residuals)
       
       fig.add_trace(
           go.Scatter(x=theoretical_quantiles, y=sample_quantiles, mode='markers', name='Q-Q'),
           row=1, col=2
       )
       fig.add_trace(
           go.Scatter(x=theoretical_quantiles, y=theoretical_quantiles, 
                     mode='lines', line=dict(color='red'), name='Normal'),
           row=1, col=2
       )
       
       fig.add_trace(
           go.Scatter(y=residuals, mode='lines', name='Residuals'),
           row=2, col=1
       )
       fig.add_hline(y=0, line_dash="dash", line_color="red", row=2, col=1)
       
       from statsmodels.tsa.stattools import acf
       try:
           acf_values = acf(residuals, nlags=min(20, len(residuals)-1))
           fig.add_trace(
               go.Bar(y=acf_values[1:], name='ACF'),
               row=2, col=2
           )
       except:
           pass
   
   fig.update_layout(
       title=f'{model_type} Model Diagnostics',
       height=800,
       showlegend=False
   )
   
   return fig

def plot_forecast_with_intervals(historical_data, forecast_data, target_col, model_name):
   """Plot forecast with prediction intervals"""
   fig = go.Figure()
   
   fig.add_trace(go.Scatter(
       x=historical_data['Date'],
       y=historical_data[target_col],
       mode='lines',
       name='Historical Data',
       line=dict(color='blue', width=1)
   ))
   
   fig.add_trace(go.Scatter(
       x=forecast_data['Date'],
       y=forecast_data[target_col],
       mode='lines',
       name='Forecast',
       line=dict(color='red', width=2)
   ))
   
   if f'{target_col}_lower' in forecast_data.columns:
       fig.add_trace(go.Scatter(
           x=forecast_data['Date'],
           y=forecast_data[f'{target_col}_upper'],
           mode='lines',
           name='Upper Bound',
           line=dict(color='red', width=1, dash='dash'),
           showlegend=False
       ))
       
       fig.add_trace(go.Scatter(
           x=forecast_data['Date'],
           y=forecast_data[f'{target_col}_lower'],
           mode='lines',
           name='Lower Bound',
           line=dict(color='red', width=1, dash='dash'),
           fill='tonexty',
           fillcolor='rgba(255, 0, 0, 0.2)',
           showlegend=False
       ))
   
   fig.add_shape(
       type="line",
       x0=historical_data['Date'].iloc[-1],
       y0=0,
       x1=historical_data['Date'].iloc[-1],
       y1=max(historical_data[target_col].max(), forecast_data[target_col].max()) * 1.1,
       line=dict(color="black", width=1, dash="dash")
   )
   
   fig.add_annotation(
       x=historical_data['Date'].iloc[-1],
       y=max(historical_data[target_col].max(), forecast_data[target_col].max()) * 1.05,
       text="Forecast Start",
       showarrow=False
   )
   
   fig.update_layout(
       title=f'{model_name} Forecast for {target_col} with Prediction Intervals',
       xaxis_title='Date',
       yaxis_title=f'{target_col} (µg/m³)',
       height=500
   )
   
   return fig

def plot_cv_scores(cv_results_dict):
   """Plot K-fold cross-validation scores for all models"""
   fig = go.Figure()
   
   for model_name, results in cv_results_dict.items():
       if 'CV_MSE_Mean' in results:
           fig.add_trace(go.Bar(
               name=model_name,
               x=[model_name],
               y=[results['CV_MSE_Mean']],
               error_y=dict(type='data', array=[results['CV_MSE_Std']])
           ))
   
   fig.update_layout(
       title='K-Fold Cross-Validation MSE Scores',
       xaxis_title='Model',
       yaxis_title='Mean Squared Error',
       showlegend=False,
       height=400
   )
   
   return fig

# Enhanced forecasting functions
def create_next_row_features(next_date, current_df, target_col, feature_cols, forecasts):
   """Create feature row for next prediction"""
   next_row = pd.Series({
       'Date': next_date,
       'Year': next_date.year,
       'Month': next_date.month,
       'Day': next_date.day,
       'DayOfWeek': next_date.weekday(),
       'Weekend': 1 if next_date.weekday() >= 5 else 0,
       'Season': 1 if next_date.month in [12, 1, 2] else 
                (2 if next_date.month in [3, 4, 5] else 
                 (3 if next_date.month in [6, 7, 8] else 4))
   })
   
   for col in feature_cols:
       if '_lag_' in col:
           lag_num = int(col.split('_')[-1])
           if lag_num <= len(forecasts):
               next_row[col] = forecasts[-lag_num]
           else:
               idx = -(lag_num - len(forecasts))
               if idx >= -len(current_df):
                   next_row[col] = current_df[target_col].iloc[idx]
               else:
                   next_row[col] = 0
       elif '_rolling_' in col or '_ewma_' in col:
           parts = col.split('_')
           if '_rolling_' in col:
               window = int(parts[-1])
               stat = parts[-2]
               
               values = list(current_df[target_col].iloc[-(window-1):].values) + forecasts
               if len(values) >= window:
                   values = values[-window:]
               
               if stat == 'mean':
                   next_row[col] = np.mean(values)
               elif stat == 'std':
                   next_row[col] = np.std(values) if len(values) > 1 else 0
               elif stat == 'max':
                   next_row[col] = np.max(values)
               elif stat == 'min':
                   next_row[col] = np.min(values)
               else:
                   next_row[col] = 0
           else:
               next_row[col] = 0
       elif col in current_df.columns and col not in next_row:
           next_row[col] = current_df[col].iloc[-1]
   
   return next_row

def generate_ml_forecasts(model, test_data, feature_cols, target_col, days, scaler):
   """Generate forecasts for ML models"""
   current_df = test_data.copy()
   forecasts = []
   last_date = test_data['Date'].iloc[-1]
   
   available_cols = [col for col in feature_cols if col in current_df.columns or 
                    col in ['Year', 'Month', 'Day', 'DayOfWeek', 'Weekend', 'Season'] or
                    '_lag_' in col or '_rolling_' in col or '_ewma_' in col]
   
   for i in range(days):
       next_date = last_date + timedelta(days=i+1)
       next_row = create_next_row_features(next_date, current_df, target_col, available_cols, forecasts)
       
       next_df = pd.DataFrame([next_row])
       for col in available_cols:
           if col not in next_df.columns:
               next_df[col] = 0
       
       X_next = next_df[available_cols].values
       if scaler:
           X_next = scaler.transform(X_next)
       
       pred = model.predict(X_next)[0]
       forecasts.append(pred)
       
       next_df[target_col] = pred
       current_df = pd.concat([current_df, next_df], ignore_index=True)
   
   return forecasts

def generate_sequence_forecasts(model, test_data, feature_cols, target_col, days, 
                              time_steps, scaler, target_scaler, model_type='lstm'):
   """Generate forecasts for sequence models (LSTM/GRU)"""
   lookback_data = test_data.copy().tail(time_steps)
   forecasts = []
   last_date = test_data['Date'].iloc[-1]
   
   available_cols = [col for col in feature_cols if col in lookback_data.columns]
   X_last = lookback_data[available_cols].values
   
   if scaler:
       X_last = scaler.transform(X_last)
   
   X_last = X_last.reshape(1, time_steps, X_last.shape[1])
   
   for i in range(days):
       pred = model.predict(X_last, verbose=0)[0][0]
       
       if target_scaler:
           pred = target_scaler.inverse_transform(pred.reshape(-1, 1))[0][0]
       
       forecasts.append(pred)
       
       next_date = last_date + timedelta(days=i+1)
       next_features = np.zeros(len(available_cols))
       
       if len(available_cols) >= 6:
           next_features[0] = next_date.year
           next_features[1] = next_date.month
           next_features[2] = next_date.day
           next_features[3] = next_date.weekday()
           next_features[4] = 1 if next_date.weekday() >= 5 else 0
           next_features[5] = 1 if next_date.month in [12, 1, 2] else (2 if next_date.month in [3, 4, 5] else (3 if next_date.month in [6, 7, 8] else 4))
       
       if scaler:
           next_features = scaler.transform(next_features.reshape(1, -1))
       
       X_last = np.roll(X_last, -1, axis=1)
       X_last[0, -1, :] = next_features
   
   return forecasts

def generate_ann_forecasts(model, test_data, feature_cols, target_col, days, scaler, target_scaler):
   """Generate forecasts for ANN model"""
   current_df = test_data.copy()
   forecasts = []
   last_date = test_data['Date'].iloc[-1]
   
   available_cols = [col for col in feature_cols if col in current_df.columns or 
                    col in ['Year', 'Month', 'Day', 'DayOfWeek', 'Weekend', 'Season'] or
                    '_lag_' in col or '_rolling_' in col or '_ewma_' in col]
   
   for i in range(days):
       next_date = last_date + timedelta(days=i+1)
       next_row = create_next_row_features(next_date, current_df, target_col, available_cols, forecasts)
       
       next_df = pd.DataFrame([next_row])
       for col in available_cols:
           if col not in next_df.columns:
               next_df[col] = 0
       
       X_next = next_df[available_cols].values
       if scaler:
           X_next = scaler.transform(X_next)
       
       pred = model.predict(X_next, verbose=0)[0][0]
       
       if target_scaler:
           pred = target_scaler.inverse_transform(pred.reshape(-1, 1))[0][0]
       
       forecasts.append(pred)
       
       next_df[target_col] = pred
       current_df = pd.concat([current_df, next_df], ignore_index=True)
   
   return forecasts

def forecast_future_enhanced(model, model_type, test_data, feature_cols, target_col, days=3, 
                          time_steps=7, scaler=None, target_scaler=None, best_params=None):
   """Enhanced forecasting with uncertainty estimation - LIMITED TO 1-3 DAYS"""
   days = min(days, 3)
   
   last_date = pd.Timestamp(test_data['Date'].iloc[-1])
   future_dates = [last_date + timedelta(days=i+1) for i in range(days)]
   future_df = pd.DataFrame({'Date': future_dates})
   
   future_df['Year'] = future_df['Date'].dt.year
   future_df['Month'] = future_df['Date'].dt.month
   future_df['Day'] = future_df['Date'].dt.day
   future_df['DayOfWeek'] = future_df['Date'].dt.dayofweek
   future_df['Weekend'] = future_df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
   future_df['Season'] = future_df['Month'].apply(lambda x: 1 if x in [12, 1, 2] else 
                                              (2 if x in [3, 4, 5] else 
                                               (3 if x in [6, 7, 8] else 4)))
   
   if model_type in ['rf', 'decision_tree'] and hasattr(model, 'estimators_' if model_type == 'rf' else 'tree_'):
       current_df = test_data.copy()
       forecasts = []
       lower_bounds = []
       upper_bounds = []
       
       for i in range(days):
           next_date = last_date + timedelta(days=i+1)
           next_row = create_next_row_features(next_date, current_df, target_col, feature_cols, forecasts)
           
           next_df = pd.DataFrame([next_row])
           available_cols = [col for col in feature_cols if col in next_df.columns or 
                           col in ['Year', 'Month', 'Day', 'DayOfWeek', 'Weekend', 'Season'] or
                           '_lag_' in col or '_rolling_' in col or '_ewma_' in col]
           
           for col in available_cols:
               if col not in next_df.columns:
                   next_df[col] = 0
           
           X_next = next_df[available_cols].values.reshape(1, -1)
           if scaler:
               X_next = scaler.transform(X_next)
           
           if model_type == 'rf':
               tree_predictions = np.array([tree.predict(X_next)[0] for tree in model.estimators_])
               pred = np.mean(tree_predictions)
               std = np.std(tree_predictions)
               lower = pred - 1.96 * std
               upper = pred + 1.96 * std
           else:
               pred = model.predict(X_next)[0]
               lower = pred * 0.9
               upper = pred * 1.1
           
           forecasts.append(pred)
           lower_bounds.append(lower)
           upper_bounds.append(upper)
           
           next_row[target_col] = pred
           current_df = pd.concat([current_df, pd.DataFrame([next_row])], ignore_index=True)
       
       future_df[target_col] = forecasts
       future_df[f'{target_col}_lower'] = lower_bounds
       future_df[f'{target_col}_upper'] = upper_bounds
   
   else:
       if model_type in ['linear', 'ridge', 'svr', 'xgboost', 'knn', 'naive_bayes']:
           forecasts = generate_ml_forecasts(model, test_data, feature_cols, target_col, days, scaler)
       elif model_type in ['lstm', 'gru']:
           forecasts = generate_sequence_forecasts(model, test_data, feature_cols, target_col, days, 
                                                  time_steps, scaler, target_scaler, model_type)
       elif model_type == 'ann':
           forecasts = generate_ann_forecasts(model, test_data, feature_cols, target_col, days, scaler, target_scaler)
       elif model_type in ['arima', 'sarima']:
           forecast_obj = model.get_forecast(steps=days)
           forecasts = forecast_obj.predicted_mean.values
           forecast_df = forecast_obj.summary_frame()
           future_df[f'{target_col}_lower'] = forecast_df['mean_ci_lower'].values
           future_df[f'{target_col}_upper'] = forecast_df['mean_ci_upper'].values
       elif model_type == 'prophet':
           prophet_future = pd.DataFrame({'ds': future_dates})
           forecast = model.predict(prophet_future)
           forecasts = forecast['yhat'].values
           future_df[f'{target_col}_lower'] = forecast['yhat_lower'].values
           future_df[f'{target_col}_upper'] = forecast['yhat_upper'].values
       
       future_df[target_col] = forecasts
   
   return future_df

# Ensemble model
class EnsembleModel:
   """Weighted ensemble of multiple models"""
   def __init__(self, models, weights=None):
       self.models = models
       self.weights = weights if weights else {name: 1/len(models) for name in models}
   
   def predict(self, X):
       """Make weighted predictions"""
       predictions = {}
       for name, model in self.models.items():
           predictions[name] = model.predict(X)
       
       weighted_pred = np.zeros(len(X))
       for name, pred in predictions.items():
           weighted_pred += self.weights[name] * pred
       
       return weighted_pred
   
   def optimize_weights(self, X_val, y_val):
       """Optimize ensemble weights using validation data"""
       from scipy.optimize import minimize
       
       def objective(weights):
           weights = weights / weights.sum()
           pred = np.zeros(len(y_val))
           for i, (name, model) in enumerate(self.models.items()):
               pred += weights[i] * model.predict(X_val)
           return mean_squared_error(y_val, pred)
       
       x0 = np.array([1/len(self.models)] * len(self.models))
       constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
       bounds = [(0, 1) for _ in range(len(self.models))]
       
       result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
       
       optimal_weights = result.x / result.x.sum()
       self.weights = {name: weight for name, weight in zip(self.models.keys(), optimal_weights)}
       
       return self.weights

def create_performance_summary_table(all_results):
   """Create summary table of all model performances"""
   summary_data = []
   
   for model_type, results in all_results.items():
       if 'test_results' in results:
           test_res = results['test_results']
           train_res = results['train_results']
           
           row_data = {
               'Model': model_type.upper(),
               'RMSE': f"{test_res['RMSE']:.2f}",
               'MAE': f"{test_res['MAE']:.2f}",
               'R²': f"{test_res['R²']:.3f}",
               'MAPE': f"{test_res['MAPE']:.1f}%",
               'Direction Acc.': f"{test_res['Directional_Accuracy']:.1f}%",
               'Overall Score': f"{test_res['Overall_Score']:.1f}"
           }
           
           if 'CV_MSE_Mean' in train_res:
               row_data['CV MSE'] = f"{train_res['CV_MSE_Mean']:.2f} ± {train_res['CV_MSE_Std']:.2f}"
           
           summary_data.append(row_data)
   
   summary_df = pd.DataFrame(summary_data)
   if not summary_df.empty:
       summary_df = summary_df.sort_values('Overall Score', ascending=False)
   
   return summary_df

# Enhanced run_forecast_model function with scaling indicators
def run_forecast_model_enhanced(model_type, train_data, val_data, test_data, target_col='PM10', 
                             feature_cols=None, forecast_days=3, tune_hyperparams=True,
                             handle_outliers_flag=True, outlier_method='interpolate',
                             n_folds=5, **kwargs):
   """Enhanced forecast model with outlier handling, hyperparameter tuning, K-fold CV, and scaling indicators"""
   results = {}
   
   forecast_days = min(forecast_days, 3)
   
   # Initialize scaling indicators
   scaling_info = {
       'feature_scaling_applied': False,
       'feature_scaling_method': None,
       'target_scaling_applied': False,
       'target_scaling_method': None
   }
   
   if handle_outliers_flag:
       train_outliers = detect_outliers(train_data, target_col, method='all')
       val_outliers = detect_outliers(val_data, target_col, method='all')
       test_outliers = detect_outliers(test_data, target_col, method='all')
       
       train_data = handle_outliers_data(train_data, target_col, train_outliers, method=outlier_method)
       val_data = handle_outliers_data(val_data, target_col, val_outliers, method=outlier_method)
       test_data = handle_outliers_data(test_data, target_col, test_outliers, method=outlier_method)
       
       results['outliers'] = {
           'train': int(train_outliers.sum()),
           'val': int(val_outliers.sum()),
           'test': int(test_outliers.sum())
       }
   
   if feature_cols is None:
       feature_cols = ['Year', 'Month', 'Day', 'DayOfWeek', 'Weekend', 'Season']
       if 'Temp' in train_data.columns:
           feature_cols.append('Temp')
       if 'WS' in train_data.columns:
           feature_cols.append('WS')
       if 'WD' in train_data.columns:
           feature_cols.append('WD')
   
   if model_type in ['arima', 'sarima', 'prophet']:
       train_data = train_data.sort_values('Date')
       val_data = val_data.sort_values('Date')
       test_data = test_data.sort_values('Date')
   else:
       train_with_lags = add_lag_features(train_data, target_col)
       val_with_lags = add_lag_features(val_data, target_col)
       test_with_lags = add_lag_features(test_data, target_col)
       
       lag_features = [col for col in train_with_lags.columns if '_lag_' in col or '_rolling_' in col or '_ewma_' in col]
       feature_cols_extended = feature_cols + lag_features
       
       X_train, y_train = create_feature_target_arrays(train_with_lags, target_col, feature_cols_extended)
       X_val, y_val = create_feature_target_arrays(val_with_lags, target_col, feature_cols_extended)
       X_test, y_test = create_feature_target_arrays(test_with_lags, target_col, feature_cols_extended)
       
       feature_cols = feature_cols_extended
   
   # Train model based on type
   if model_type == 'linear':
       X_train_scaled, X_val_scaled, X_test_scaled, feature_scaler, scaling_method = scale_data(
           X_train, X_val, X_test, scaler_type='robust'
       )
       
       scaling_info['feature_scaling_applied'] = True
       scaling_info['feature_scaling_method'] = scaling_method
       
       model, train_results, val_results, best_params = train_linear_regression(
           X_train_scaled, y_train, X_val_scaled, y_val, tune_params=tune_hyperparams, n_folds=n_folds
       )
       y_test_pred = model.predict(X_test_scaled)
       scaler_use = feature_scaler
       test_data_use = test_with_lags
   
   elif model_type == 'ridge':
       X_train_scaled, X_val_scaled, X_test_scaled, feature_scaler, scaling_method = scale_data(
           X_train, X_val, X_test, scaler_type='robust'
       )
       
       scaling_info['feature_scaling_applied'] = True
       scaling_info['feature_scaling_method'] = scaling_method
       
       model, train_results, val_results, best_params = train_ridge_regression(
           X_train_scaled, y_train, X_val_scaled, y_val, tune_params=tune_hyperparams, n_folds=n_folds
       )
       y_test_pred = model.predict(X_test_scaled)
       scaler_use = feature_scaler
       test_data_use = test_with_lags
   
   elif model_type == 'decision_tree':
       model, train_results, val_results, best_params = train_decision_tree(
           X_train, y_train, X_val, y_val, tune_params=tune_hyperparams, n_folds=n_folds
       )
       y_test_pred = model.predict(X_test)
       scaler_use = None
       test_data_use = test_with_lags
   
   elif model_type == 'knn':
       X_train_scaled, X_val_scaled, X_test_scaled, feature_scaler, scaling_method = scale_data(
           X_train, X_val, X_test, scaler_type='standard'
       )
       
       scaling_info['feature_scaling_applied'] = True
       scaling_info['feature_scaling_method'] = scaling_method
       
       model, train_results, val_results, best_params = train_knn(
           X_train_scaled, y_train, X_val_scaled, y_val, tune_params=tune_hyperparams, n_folds=n_folds
       )
       y_test_pred = model.predict(X_test_scaled)
       scaler_use = feature_scaler
       test_data_use = test_with_lags
   
   elif model_type == 'naive_bayes':
       X_train_scaled, X_val_scaled, X_test_scaled, feature_scaler, scaling_method = scale_data(
           X_train, X_val, X_test, scaler_type='standard'
       )
       
       scaling_info['feature_scaling_applied'] = True
       scaling_info['feature_scaling_method'] = scaling_method
       
       model, train_results, val_results, best_params = train_naive_bayes(
           X_train_scaled, y_train, X_val_scaled, y_val, tune_params=tune_hyperparams, n_folds=n_folds
       )
       y_test_pred = model.predict(X_test_scaled)
       scaler_use = feature_scaler
       test_data_use = test_with_lags
   
   elif model_type == 'rf':
       model, train_results, val_results, best_params = train_random_forest(
           X_train, y_train, X_val, y_val, tune_params=tune_hyperparams, n_folds=n_folds
       )
       y_test_pred = model.predict(X_test)
       scaler_use = None
       test_data_use = test_with_lags
   
   elif model_type == 'xgboost':
       model, train_results, val_results, best_params = train_xgboost(
           X_train, y_train, X_val, y_val, tune_params=tune_hyperparams, n_folds=n_folds
       )
       y_test_pred = model.predict(X_test)
       scaler_use = None
       test_data_use = test_with_lags
   
   elif model_type == 'svr':
       X_train_scaled, X_val_scaled, X_test_scaled, feature_scaler, scaling_method = scale_data(
           X_train, X_val, X_test, scaler_type='robust'
       )
       
       scaling_info['feature_scaling_applied'] = True
       scaling_info['feature_scaling_method'] = scaling_method
       
       model, train_results, val_results, best_params = train_svr(
           X_train_scaled, y_train, X_val_scaled, y_val, tune_params=tune_hyperparams, n_folds=n_folds
       )
       y_test_pred = model.predict(X_test_scaled)
       scaler_use = feature_scaler
       test_data_use = test_with_lags
   
   elif model_type == 'lstm':
       X_train_scaled, X_val_scaled, X_test_scaled, feature_scaler, feature_scaling_method = scale_data(
           X_train, X_val, X_test, scaler_type='robust'
       )
       
       y_train_scaled, y_val_scaled, y_test_scaled, y_scaler, target_scaling_method = scale_target(y_train, y_val, y_test)
       
       scaling_info['feature_scaling_applied'] = True
       scaling_info['feature_scaling_method'] = feature_scaling_method
       scaling_info['target_scaling_applied'] = True
       scaling_info['target_scaling_method'] = target_scaling_method
       
       time_steps = kwargs.get('lstm_time_steps', 7)
       n_trials = kwargs.get('lstm_trials', 5) if tune_hyperparams else 1
       
       try:
           if len(X_train_scaled) < time_steps * 2:
               raise ValueError(f"Not enough training data for LSTM. Need at least {time_steps * 2} samples, have {len(X_train_scaled)}")
           
           model, train_results, val_results, best_params, history, X_train_seq, X_val_seq, y_train_seq, y_val_seq = train_lstm_optuna(
               X_train_scaled, y_train_scaled.flatten(), 
               X_val_scaled, y_val_scaled.flatten(), 
               time_steps=time_steps, n_trials=n_trials
           )
           
           X_test_seq, y_test_seq = create_sequence_dataset(X_test_scaled, y_test_scaled.flatten(), time_steps)
           if len(X_test_seq) > 0:
               y_test_pred_scaled = model.predict(X_test_seq, verbose=0).flatten()
               y_test_pred = y_scaler.inverse_transform(y_test_pred_scaled.reshape(-1, 1)).flatten()
           else:
               raise ValueError(f"Not enough test data for LSTM sequences. Need at least {time_steps} samples.")
           
           results['lstm_data'] = {
               'X_train_seq': X_train_seq,
               'X_val_seq': X_val_seq,
               'X_test_seq': X_test_seq,
               'y_scaler': y_scaler,
               'time_steps': time_steps,
               'history': history
           }
           
           scaler_use = feature_scaler
           test_data_use = test_with_lags
           y_test = y_test_seq
       except Exception as e:
           raise ValueError(f"LSTM training failed: {str(e)}")
   
   elif model_type == 'gru':
       X_train_scaled, X_val_scaled, X_test_scaled, feature_scaler, feature_scaling_method = scale_data(
           X_train, X_val, X_test, scaler_type='robust'
       )
       
       y_train_scaled, y_val_scaled, y_test_scaled, y_scaler, target_scaling_method = scale_target(y_train, y_val, y_test)
       
       scaling_info['feature_scaling_applied'] = True
       scaling_info['feature_scaling_method'] = feature_scaling_method
       scaling_info['target_scaling_applied'] = True
       scaling_info['target_scaling_method'] = target_scaling_method
       
       time_steps = kwargs.get('gru_time_steps', 7)
       n_trials = kwargs.get('gru_trials', 5) if tune_hyperparams else 1
       
       try:
           if len(X_train_scaled) < time_steps * 2:
               raise ValueError(f"Not enough training data for GRU. Need at least {time_steps * 2} samples, have {len(X_train_scaled)}")
           
           model, train_results, val_results, best_params, history, X_train_seq, X_val_seq, y_train_seq, y_val_seq = train_gru_optuna(
               X_train_scaled, y_train_scaled.flatten(), 
               X_val_scaled, y_val_scaled.flatten(), 
               time_steps=time_steps, n_trials=n_trials
           )
           
           X_test_seq, y_test_seq = create_sequence_dataset(X_test_scaled, y_test_scaled.flatten(), time_steps)
           if len(X_test_seq) > 0:
               y_test_pred_scaled = model.predict(X_test_seq, verbose=0).flatten()
               y_test_pred = y_scaler.inverse_transform(y_test_pred_scaled.reshape(-1, 1)).flatten()
           else:
               raise ValueError(f"Not enough test data for GRU sequences. Need at least {time_steps} samples.")
           
           results['gru_data'] = {
               'X_train_seq': X_train_seq,
               'X_val_seq': X_val_seq,
               'X_test_seq': X_test_seq,
               'y_scaler': y_scaler,
               'time_steps': time_steps,
               'history': history
           }
           
           scaler_use = feature_scaler
           test_data_use = test_with_lags
           y_test = y_test_seq
       except Exception as e:
           raise ValueError(f"GRU training failed: {str(e)}")
   
   elif model_type == 'ann':
       X_train_scaled, X_val_scaled, X_test_scaled, feature_scaler, feature_scaling_method = scale_data(
           X_train, X_val, X_test, scaler_type='robust'
       )
       
       y_train_scaled, y_val_scaled, y_test_scaled, y_scaler, target_scaling_method = scale_target(y_train, y_val, y_test)
       
       scaling_info['feature_scaling_applied'] = True
       scaling_info['feature_scaling_method'] = feature_scaling_method
       scaling_info['target_scaling_applied'] = True
       scaling_info['target_scaling_method'] = target_scaling_method
       
       n_trials = kwargs.get('ann_trials', 5) if tune_hyperparams else 1
       
       try:
           model, train_results, val_results, best_params, history = train_ann_optuna(
               X_train_scaled, y_train_scaled.flatten(), 
               X_val_scaled, y_val_scaled.flatten(), 
               n_trials=n_trials
           )
           
           y_test_pred_scaled = model.predict(X_test_scaled, verbose=0).flatten()
           y_test_pred = y_scaler.inverse_transform(y_test_pred_scaled.reshape(-1, 1)).flatten()
           
           results['ann_data'] = {
               'y_scaler': y_scaler,
               'history': history
           }
           
           scaler_use = feature_scaler
           test_data_use = test_with_lags
       except Exception as e:
           raise ValueError(f"ANN training failed: {str(e)}")
   
   elif model_type == 'arima':
       model, train_results, val_results, train_pred, val_pred, best_order = train_arima(
           train_data, val_data, target_col, tune_params=tune_hyperparams
       )
       best_params = {'order': best_order}
       
       test_series = test_data.set_index('Date')[target_col]
       n_periods = len(test_series)
       y_test_pred = model.forecast(steps=n_periods)
       y_test = test_series.values
       
       scaler_use = None
       test_data_use = test_data
   
   elif model_type == 'sarima':
       model, train_results, val_results, train_pred, val_pred, best_order, best_seasonal = train_sarima(
           train_data, val_data, target_col, tune_params=tune_hyperparams
       )
       best_params = {'order': best_order, 'seasonal_order': best_seasonal}
       
       test_series = test_data.set_index('Date')[target_col]
       n_periods = len(test_series)
       y_test_pred = model.forecast(steps=n_periods)
       y_test = test_series.values
       
       scaler_use = None
       test_data_use = test_data
   
   elif model_type == 'prophet':
       model, train_results, val_results, train_forecast, val_forecast, best_params = train_prophet(
           train_data, val_data, tune_params=tune_hyperparams
       )
       
       prophet_test = test_data[['Date', target_col]].rename(columns={'Date': 'ds', target_col: 'y'})
       test_forecast = model.predict(prophet_test[['ds']])
       y_test_pred = test_forecast['yhat'].values
       y_test = prophet_test['y'].values
       
       scaler_use = None
       test_data_use = test_data
   
   else:
       raise ValueError(f"Model type '{model_type}' not supported.")
   
   # Evaluate on test set
   test_results = evaluate_model(y_test, y_test_pred, f"{model_type.upper()} (Test)")
   test_results['Overall_Score'] = calculate_overall_score(test_results)
   
   # Generate forecast
   forecast_df = forecast_future_enhanced(
       model, model_type, test_data_use, feature_cols, target_col,
       days=forecast_days, 
       time_steps=7 if model_type in ['lstm', 'gru'] else None,
       scaler=scaler_use, 
       target_scaler=results.get(f'{model_type}_data', {}).get('y_scaler') if model_type in ['lstm', 'gru', 'ann'] else None,
       best_params=best_params
   )
   
   # Store all results including scaling information
   results.update({
       'model': model,
       'test_pred': y_test_pred,
       'train_results': train_results,
       'val_results': val_results,
       'test_results': test_results,
       'feature_names': feature_cols,
       'best_params': best_params,
       'forecast': forecast_df,
       'model_type': model_type,
       'scaling_info': scaling_info  # NEW: Added scaling indicators
   })
   
   # Add feature importance for tree-based models
   if model_type in ['rf', 'xgboost', 'decision_tree']:
       importance_fig, importance_df = plot_feature_importance(model, feature_cols, model_type.upper())
       results['importance_fig'] = importance_fig
       results['importance_df'] = importance_df
   
   return results

# Main Streamlit app - Updated with scaling indicators
def main():
   # Sidebar for navigation
   st.sidebar.title("Navigation")
   page = st.sidebar.radio("Go to", ["Data Explorer", "Data Split Configuration", "Outlier Analysis", "Model Training", 
                                     "Model Comparison", "Ensemble Modeling", "Forecasting Dashboard"])

   # Initialize session state
   if 'data' not in st.session_state:
       st.session_state.data = None
       st.session_state.daily_data = None
       st.session_state.train_data = None
       st.session_state.val_data = None
       st.session_state.test_data = None
       st.session_state.model_results = {}
       st.session_state.ensemble_model = None
       st.session_state.train_ratio = 0.65
       st.session_state.val_ratio = 0.15
       st.session_state.test_ratio = 0.20
   
   # Data Upload
   st.sidebar.title("Data Upload")
   uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
   
   if uploaded_file is not None:
       with st.spinner("Loading data..."):
           df = load_data(uploaded_file)
           df = df.drop(columns = ["AQI", "PM2.5"])
           st.session_state.data = df
           
           st.session_state.daily_data = create_daily_dataset(df)
           
           st.session_state.train_data, st.session_state.val_data, st.session_state.test_data = split_data(
               st.session_state.daily_data, 'PM10',
               train_size=st.session_state.train_ratio,
               val_size=st.session_state.val_ratio,
               test_size=st.session_state.test_ratio
           )
           
           st.sidebar.success(f"Loaded data with {len(df)} rows and {df.shape[1]} columns")
   
   # Data Explorer Page
   if page == "Data Explorer":
       st.title("📊 Data Explorer")
       
       if st.session_state.data is not None:
           st.header("Data Overview")
           
           st.subheader("Daily Aggregated Data")
           st.dataframe(st.session_state.daily_data.head(10))
           
           st.subheader("Current Data Split Configuration")
           col1, col2, col3, col4 = st.columns(4)
           with col1:
               st.metric("Training Set", f"{len(st.session_state.train_data)} days")
               st.write(f"({st.session_state.train_ratio*100:.1f}%)")
           with col2:
               st.metric("Validation Set", f"{len(st.session_state.val_data)} days")
               st.write(f"({st.session_state.val_ratio*100:.1f}%)")
           with col3:
               st.metric("Test Set", f"{len(st.session_state.test_data)} days")
               st.write(f"({st.session_state.test_ratio*100:.1f}%)")
           with col4:
               st.metric("Total Days", f"{len(st.session_state.daily_data)}")
               st.write("(100%)")
           
           st.subheader("Time Series Plot")
           ts_fig = plot_time_series(st.session_state.daily_data, 'PM10')
           st.plotly_chart(ts_fig, use_container_width=True)
           
           st.subheader("Correlation Analysis")
           corr_fig, corr = plot_correlation_matrix(st.session_state.daily_data, 'PM10')
           st.plotly_chart(corr_fig, use_container_width=True)
           
           st.subheader("Statistical Summary")
           st.dataframe(st.session_state.daily_data.describe())
           
           # Data split visualization
           st.subheader("Data Split Visualization")
           
           fig = go.Figure()
           
           fig.add_trace(go.Scatter(
               x=st.session_state.train_data['Date'],
               y=st.session_state.train_data['PM10'],
               mode='lines',
               name=f'Training ({st.session_state.train_ratio*100:.1f}%)',
               line=dict(color='blue', width=1)
           ))
           
           fig.add_trace(go.Scatter(
               x=st.session_state.val_data['Date'],
               y=st.session_state.val_data['PM10'],
               mode='lines',
               name=f'Validation ({st.session_state.val_ratio*100:.1f}%)',
               line=dict(color='orange', width=1)
           ))
           
           fig.add_trace(go.Scatter(
               x=st.session_state.test_data['Date'],
               y=st.session_state.test_data['PM10'],
               mode='lines',
               name=f'Test ({st.session_state.test_ratio*100:.1f}%)',
               line=dict(color='red', width=1)
           ))
           
           fig.update_layout(
               title='PM10 Time Series with Train/Validation/Test Split',
               xaxis_title='Date',
               yaxis_title='PM10 (µg/m³)',
               height=500
           )
           
           st.plotly_chart(fig, use_container_width=True)
       
       else:
           st.info("Please upload a dataset using the sidebar.")
   
   # Data Split Configuration Page
   elif page == "Data Split Configuration":
    st.title("⚙️ Data Split Configuration")
    
    if st.session_state.data is not None:
        st.header("Configure Train/Validation/Test Split Ratios")
        st.write("Adjust the data split ratios below. The splits will be applied chronologically (time-series order).")
        
        st.subheader("Current Configuration")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Training", f"{st.session_state.train_ratio*100:.1f}%", 
                     f"{len(st.session_state.train_data)} days")
        with col2:
            st.metric("Validation", f"{st.session_state.val_ratio*100:.1f}%", 
                     f"{len(st.session_state.val_data)} days")
        with col3:
            st.metric("Test", f"{st.session_state.test_ratio*100:.1f}%", 
                     f"{len(st.session_state.test_data)} days")
        
        # Initialize session state for temporary slider values if not exists
        if 'temp_train_ratio' not in st.session_state:
            st.session_state.temp_train_ratio = st.session_state.train_ratio
        if 'temp_val_ratio' not in st.session_state:
            st.session_state.temp_val_ratio = st.session_state.val_ratio
        if 'temp_test_ratio' not in st.session_state:
            st.session_state.temp_test_ratio = st.session_state.test_ratio
        
        # Preset configurations section - placed before sliders
        st.subheader("Preset Configurations")
        preset_col1, preset_col2, preset_col3, preset_col4 = st.columns(4)
        
        with preset_col1:
            if st.button("Standard ML\n(70/15/15)", key="preset_standard"):
                st.session_state.temp_train_ratio = 0.70
                st.session_state.temp_val_ratio = 0.15
                st.session_state.temp_test_ratio = 0.15
                st.rerun()
        
        with preset_col2:
            if st.button("Conservative\n(60/20/20)", key="preset_conservative"):
                st.session_state.temp_train_ratio = 0.60
                st.session_state.temp_val_ratio = 0.20
                st.session_state.temp_test_ratio = 0.20
                st.rerun()
        
        with preset_col3:
            if st.button("Training Heavy\n(80/10/10)", key="preset_training"):
                st.session_state.temp_train_ratio = 0.80
                st.session_state.temp_val_ratio = 0.10
                st.session_state.temp_test_ratio = 0.10
                st.rerun()
        
        with preset_col4:
            if st.button("Balanced\n(65/15/20)", key="preset_balanced"):
                st.session_state.temp_train_ratio = 0.65
                st.session_state.temp_val_ratio = 0.15
                st.session_state.temp_test_ratio = 0.20
                st.rerun()
        
        st.subheader("Adjust Split Ratios")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            new_train_ratio = st.slider(
                "Training Set Ratio",
                min_value=0.4,
                max_value=0.8,
                value=st.session_state.temp_train_ratio,
                step=0.05,
                help="Percentage of data used for training (chronologically first)",
                key="train_slider"
            )
            # Update session state when slider changes
            st.session_state.temp_train_ratio = new_train_ratio
        
        with col2:
            new_val_ratio = st.slider(
                "Validation Set Ratio",
                min_value=0.1,
                max_value=0.3,
                value=st.session_state.temp_val_ratio,
                step=0.05,
                help="Percentage of data used for validation (chronologically middle)",
                key="val_slider"
            )
            # Update session state when slider changes
            st.session_state.temp_val_ratio = new_val_ratio
        
        with col3:
            new_test_ratio = st.slider(
                "Test Set Ratio",
                min_value=0.1,
                max_value=0.3,
                value=st.session_state.temp_test_ratio,
                step=0.05,
                help="Percentage of data used for testing (chronologically last)",
                key="test_slider"
            )
            # Update session state when slider changes
            st.session_state.temp_test_ratio = new_test_ratio
        
        total_ratio = new_train_ratio + new_val_ratio + new_test_ratio
        
        st.write(f"**Total Ratio:** {total_ratio:.2f}")
        
        if abs(total_ratio - 1.0) > 0.01:
            st.warning(f"⚠️ Ratios should sum to 1.0. Current sum: {total_ratio:.2f}")
            st.info("Ratios will be automatically normalized when applied.")
        else:
            st.success("✅ Ratios sum correctly to 1.0")
        
        # Apply changes button
        if st.button("🔄 Apply New Split Configuration", type="primary"):
            st.session_state.train_ratio = new_train_ratio
            st.session_state.val_ratio = new_val_ratio
            st.session_state.test_ratio = new_test_ratio
            
            with st.spinner("Applying new data split..."):
                st.session_state.train_data, st.session_state.val_data, st.session_state.test_data = split_data(
                    st.session_state.daily_data, 'PM10',
                    train_size=st.session_state.train_ratio,
                    val_size=st.session_state.val_ratio,
                    test_size=st.session_state.test_ratio
                )
                
                st.session_state.model_results = {}
                st.session_state.ensemble_model = None
            
            st.success("✅ Data split configuration updated successfully!")
            st.info("ℹ️ Previous model results have been cleared. Please retrain models with the new data split.")
            st.rerun()
        
        # Show preview of new split
        if new_train_ratio != st.session_state.train_ratio or new_val_ratio != st.session_state.val_ratio or new_test_ratio != st.session_state.test_ratio:
            st.subheader("Preview of New Split")
            
            total_data = len(st.session_state.daily_data)
            
            total_new_ratio = new_train_ratio + new_val_ratio + new_test_ratio
            norm_train = new_train_ratio / total_new_ratio
            norm_val = new_val_ratio / total_new_ratio
            norm_test = new_test_ratio / total_new_ratio
            
            new_train_size = int(total_data * norm_train)
            new_val_size = int(total_data * norm_val)
            new_test_size = total_data - new_train_size - new_val_size
            
            preview_col1, preview_col2, preview_col3 = st.columns(3)
            with preview_col1:
                st.metric("New Training Size", f"{new_train_size} days", 
                         f"{new_train_size - len(st.session_state.train_data):+d}")
            with preview_col2:
                st.metric("New Validation Size", f"{new_val_size} days", 
                         f"{new_val_size - len(st.session_state.val_data):+d}")
            with preview_col3:
                st.metric("New Test Size", f"{new_test_size} days", 
                         f"{new_test_size - len(st.session_state.test_data):+d}")
        
        # Recommendations section
        st.subheader("📋 Split Ratio Recommendations")
        
        total_days = len(st.session_state.daily_data)
        
        if total_days < 365:
            st.warning("⚠️ **Small Dataset Warning:** With less than 1 year of data, consider:")
            st.write("- Using 80/10/10 split to maximize training data")
            st.write("- Implementing time series cross-validation")
            st.write("- Being cautious about model generalization")
        elif total_days < 730:
            st.info("📊 **Medium Dataset:** With 1-2 years of data:")
            st.write("- Standard 70/15/15 split is recommended")
            st.write("- Ensure each split covers different seasons")
            st.write("- Consider seasonal validation strategies")
        else:
            st.success("📈 **Large Dataset:** With 2+ years of data:")
            st.write("- Any split ratio should work well")
            st.write("- 65/15/20 provides good balance")
            st.write("- Consider 60/20/20 for robust evaluation")
        
        # Date range information
        st.subheader("📅 Current Date Ranges")
        if len(st.session_state.train_data) > 0:
            train_start = st.session_state.train_data['Date'].min().strftime('%Y-%m-%d')
            train_end = st.session_state.train_data['Date'].max().strftime('%Y-%m-%d')
            val_start = st.session_state.val_data['Date'].min().strftime('%Y-%m-%d')
            val_end = st.session_state.val_data['Date'].max().strftime('%Y-%m-%d')
            test_start = st.session_state.test_data['Date'].min().strftime('%Y-%m-%d')
            test_end = st.session_state.test_data['Date'].max().strftime('%Y-%m-%d')
            
            date_info = pd.DataFrame({
                'Split': ['Training', 'Validation', 'Test'],
                'Start Date': [train_start, val_start, test_start],
                'End Date': [train_end, val_end, test_end],
                'Duration (days)': [len(st.session_state.train_data), 
                                   len(st.session_state.val_data), 
                                   len(st.session_state.test_data)],
                'Percentage': [f"{st.session_state.train_ratio*100:.1f}%",
                              f"{st.session_state.val_ratio*100:.1f}%",
                              f"{st.session_state.test_ratio*100:.1f}%"]
            })
            
            st.dataframe(date_info, use_container_width=True)
    
    else:
        st.info("Please upload a dataset using the sidebar to configure data splits.")
   
   # Outlier Analysis Page
   elif page == "Outlier Analysis":
       st.title("🔍 Outlier Analysis")
       
       if st.session_state.data is not None:
           st.header("Outlier Detection and Handling")
           
           st.info(f"Current split: Train {st.session_state.train_ratio*100:.1f}% | "
                  f"Val {st.session_state.val_ratio*100:.1f}% | "
                  f"Test {st.session_state.test_ratio*100:.1f}%")
           
           col1, col2 = st.columns(2)
           with col1:
               detection_method = st.selectbox(
                   "Detection Method",
                   ["iqr"]
               )
           with col2:
               handling_method = st.selectbox(
                   "Handling Method",
                   ["cap"]
               )
           
           outliers = detect_outliers(st.session_state.daily_data, 'PM10', method=detection_method)
           
           st.subheader("Outlier Statistics")
           col1, col2, col3 = st.columns(3)
           with col1:
               st.metric("Total Outliers", f"{outliers.sum()}")
           with col2:
               st.metric("Percentage", f"{(outliers.sum() / len(outliers) * 100):.1f}%")
           with col3:
               st.metric("Clean Data Points", f"{(~outliers).sum()}")
           
           outlier_fig = plot_outliers(st.session_state.daily_data, 'PM10', outliers)
           st.plotly_chart(outlier_fig, use_container_width=True)
           
           if st.button("Apply Outlier Handling"):
               cleaned_data = handle_outliers_data(
                   st.session_state.daily_data, 'PM10', outliers, method=handling_method
               )
               
               fig = go.Figure()
               fig.add_trace(go.Scatter(
                   x=st.session_state.daily_data['Date'],
                   y=st.session_state.daily_data['PM10'],
                   mode='lines',
                   name='Original',
                   line=dict(color='blue', width=1)
               ))
               fig.add_trace(go.Scatter(
                   x=cleaned_data['Date'],
                   y=cleaned_data['PM10'],
                   mode='lines',
                   name='After Handling',
                   line=dict(color='green', width=1)
               ))
               fig.update_layout(
                   title='PM10 Before and After Outlier Handling',
                   xaxis_title='Date',
                   yaxis_title='PM10 (µg/m³)',
                   height=500
               )
               st.plotly_chart(fig, use_container_width=True)
       
       else:
           st.info("Please upload a dataset using the sidebar.")
   
   # Model Training Page with Enhanced Scaling Information
   elif page == "Model Training":
       st.title("🤖 Model Training with Hyperparameter Tuning and K-Fold CV")
       
       if st.session_state.data is not None:
           st.info(f"Using split: Train {st.session_state.train_ratio*100:.1f}% "
                  f"({len(st.session_state.train_data)} days) | "
                  f"Val {st.session_state.val_ratio*100:.1f}% "
                  f"({len(st.session_state.val_data)} days) | "
                  f"Test {st.session_state.test_ratio*100:.1f}% "
                  f"({len(st.session_state.test_data)} days)")
           
           st.sidebar.title("Model Settings")
           model_type = st.sidebar.selectbox(
               "Select Model", 
               ["decision_tree", "knn", "rf", "xgboost", 
                "lstm", "gru", "ann", "arima", "sarima", "prophet"] 
           )
           
           with st.sidebar.expander("Advanced Settings"):
               tune_hyperparams = st.checkbox("Tune Hyperparameters", value=True)
               handle_outliers_flag = st.checkbox("Handle Outliers", value=True)
               if handle_outliers_flag:
                   outlier_method = st.selectbox(
                       "Outlier Handling Method",
                       ["cap"] 
                   )
               else:
                   outlier_method = None
               
               n_folds = st.slider("K-Fold Cross-Validation Folds", 3, 10, 5)
               forecast_days = st.slider("Forecast Days", 1, 3, 3)
               
               if model_type == 'lstm':
                   st.info("LSTM Settings")
                   lstm_trials = st.slider("Optuna Trials (set 1 to skip optimization)", 1, 20, 5)
                   lstm_time_steps = st.slider("Time Steps (sequence length)", 3, 14, 7)
               else:
                   lstm_trials = 5
                   lstm_time_steps = 7
               
               if model_type == 'gru':
                   st.info("GRU Settings")
                   gru_trials = st.slider("Optuna Trials (set 1 to skip optimization)", 1, 20, 5)
                   gru_time_steps = st.slider("Time Steps (sequence length)", 3, 14, 7)
               else:
                   gru_trials = 5
                   gru_time_steps = 7
               
               if model_type == 'ann':
                   st.info("ANN Settings")
                   ann_trials = st.slider("Optuna Trials (set 1 to skip optimization)", 1, 20, 5)
               else:
                   ann_trials = 5
           
           if len(st.session_state.model_results) > 0:
               st.warning("⚠️ Note: If you changed the data split configuration, "
                         "previous model results may not be comparable. "
                         "Consider retraining all models for fair comparison.")
           
           if st.sidebar.button("Train Model"):
               with st.spinner(f"Training {model_type.upper()} model with {n_folds}-fold CV..."):
                   try:
                       results = run_forecast_model_enhanced(
                           model_type,
                           st.session_state.train_data,
                           st.session_state.val_data,
                           st.session_state.test_data,
                           forecast_days=forecast_days,
                           tune_hyperparams=tune_hyperparams,
                           handle_outliers_flag=handle_outliers_flag,
                           outlier_method=outlier_method,
                           n_folds=n_folds,
                           lstm_trials=lstm_trials if model_type == 'lstm' else 5,
                           lstm_time_steps=lstm_time_steps if model_type == 'lstm' else 7,
                           gru_trials=gru_trials if model_type == 'gru' else 5,
                           gru_time_steps=gru_time_steps if model_type == 'gru' else 7,
                           ann_trials=ann_trials if model_type == 'ann' else 5
                       )
                       
                       if results:
                           results['data_split'] = {
                               'train_ratio': st.session_state.train_ratio,
                               'val_ratio': st.session_state.val_ratio,
                               'test_ratio': st.session_state.test_ratio,
                               'train_size': len(st.session_state.train_data),
                               'val_size': len(st.session_state.val_data),
                               'test_size': len(st.session_state.test_data)
                           }
                           
                           st.session_state.model_results[model_type] = results
                           st.success(f"{model_type.upper()} model trained successfully!")
                   except Exception as e:
                       st.error(f"Error training {model_type.upper()} model: {str(e)}")
           
           # Show model results if available
           if model_type in st.session_state.model_results:
               results = st.session_state.model_results[model_type]
               
               # Model performance metrics
               st.header("Model Performance")
               
               # Show data split info for this model
               if 'data_split' in results:
                   split_info = results['data_split']
                   st.write(f"**Model trained with split:** "
                           f"Train {split_info['train_ratio']*100:.1f}% "
                           f"({split_info['train_size']} days) | "
                           f"Val {split_info['val_ratio']*100:.1f}% "
                           f"({split_info['val_size']} days) | "
                           f"Test {split_info['test_ratio']*100:.1f}% "
                           f"({split_info['test_size']} days)")
               
               # NEW: Display scaling information
               if 'scaling_info' in results:
                   st.subheader("📊 Data Scaling Information")
                   scaling_info = results['scaling_info']
                   
                   col1, col2 = st.columns(2)
                   
                   with col1:
                       if scaling_info['feature_scaling_applied']:
                           st.success("✅ **Feature Scaling Applied**")
                           st.write(f"**Method:** {scaling_info['feature_scaling_method']}")
                       else:
                           st.info("ℹ️ **No Feature Scaling**")
                           st.write("Raw features used (typical for tree-based models)")
                   
                   with col2:
                       if scaling_info['target_scaling_applied']:
                           st.success("✅ **Target Scaling Applied**")
                           st.write(f"**Method:** {scaling_info['target_scaling_method']}")
                       else:
                           st.info("ℹ️ **No Target Scaling**")
                           st.write("Raw target values used")
                   
                   # Explanation of scaling benefits
                   if scaling_info['feature_scaling_applied'] or scaling_info['target_scaling_applied']:
                       st.info("**Scaling Benefits:** Scaling ensures all features contribute equally to the model "
                              "and can improve convergence for gradient-based algorithms.")
               
               if tune_hyperparams and results['best_params']:
                   st.subheader("Best Hyperparameters")
                   st.json(results['best_params'])
               
               # Performance metrics table
               metrics_df = pd.DataFrame([
                   {**results['train_results'], 'Dataset': 'Train'},
                   {**results['val_results'], 'Dataset': 'Validation'},
                   {**results['test_results'], 'Dataset': 'Test'}
               ])
               
               st.dataframe(metrics_df)
               
               # K-Fold Cross-Validation Results
               if 'CV_MSE_Mean' in results['train_results']:
                   st.subheader("K-Fold Cross-Validation Results")
                   col1, col2 = st.columns(2)
                   with col1:
                       st.metric("CV MSE Mean", f"{results['train_results']['CV_MSE_Mean']:.2f}")
                   with col2:
                       st.metric("CV MSE Std", f"{results['train_results']['CV_MSE_Std']:.2f}")
               
               st.success(f"Overall Performance Score: {results['test_results']['Overall_Score']:.1f}/100")
               
               st.header("Model Diagnostics")
               diagnostics_fig = plot_model_diagnostics(results, model_type.upper())
               st.plotly_chart(diagnostics_fig, use_container_width=True)
               
               if 'importance_fig' in results and results['importance_fig'] is not None:
                   st.header("Feature Importance")
                   st.plotly_chart(results['importance_fig'], use_container_width=True)
               
               st.header(f"{forecast_days}-Day Forecast")
               forecast_fig = plot_forecast_with_intervals(
                   st.session_state.daily_data,
                   results['forecast'],
                   'PM10',
                   model_type.upper()
               )
               st.plotly_chart(forecast_fig, use_container_width=True)
               
               st.subheader("Forecast Data")
               st.dataframe(results['forecast'])
       
       else:
           st.info("Please upload a dataset using the sidebar.")
   
   # Model Comparison Page (updated to show scaling info)
   elif page == "Model Comparison":
       st.title("📈 Model Comparison")
       
       if st.session_state.data is not None and st.session_state.model_results:
           # Check if all models use the same data split
           split_configs = {}
           for model_name, results in st.session_state.model_results.items():
               if 'data_split' in results:
                   split_key = f"{results['data_split']['train_ratio']:.2f}-{results['data_split']['val_ratio']:.2f}-{results['data_split']['test_ratio']:.2f}"
                   if split_key not in split_configs:
                       split_configs[split_key] = []
                   split_configs[split_key].append(model_name)
           
           if len(split_configs) > 1:
               st.warning("⚠️ **Mixed Data Splits Detected:** Models were trained with different data splits. "
                         "Comparison may not be fair. Consider retraining all models with the same split configuration.")
               
               st.subheader("Models by Split Configuration")
               for split_key, models in split_configs.items():
                   ratios = split_key.split('-')
                   st.write(f"**Split {ratios[0]}/{ratios[1]}/{ratios[2]}:** {', '.join([m.upper() for m in models])}")
           else:
               st.info(f"All models trained with split: Train {st.session_state.train_ratio*100:.1f}% | "
                      f"Val {st.session_state.val_ratio*100:.1f}% | "
                      f"Test {st.session_state.test_ratio*100:.1f}%")
           
           # NEW: Scaling Information Summary
           st.subheader("📊 Scaling Methods by Model")
           scaling_summary = []
           for model_name, results in st.session_state.model_results.items():
               if 'scaling_info' in results:
                   scaling_info = results['scaling_info']
                   scaling_summary.append({
                       'Model': model_name.upper(),
                       'Feature Scaling': scaling_info['feature_scaling_method'] if scaling_info['feature_scaling_applied'] else 'None',
                       'Target Scaling': scaling_info['target_scaling_method'] if scaling_info['target_scaling_applied'] else 'None'
                   })
           
           if scaling_summary:
               scaling_df = pd.DataFrame(scaling_summary)
               st.dataframe(scaling_df, use_container_width=True)
               
               # Show scaling patterns
               feature_scaled_models = [row['Model'] for row in scaling_summary if row['Feature Scaling'] != 'None']
               target_scaled_models = [row['Model'] for row in scaling_summary if row['Target Scaling'] != 'None']
               
               col1, col2 = st.columns(2)
               with col1:
                   st.info(f"**Models with Feature Scaling:** {', '.join(feature_scaled_models) if feature_scaled_models else 'None'}")
               with col2:
                   st.info(f"**Models with Target Scaling:** {', '.join(target_scaled_models) if target_scaled_models else 'None'}")
           
           st.header("Performance Summary")
           summary_df = create_performance_summary_table(st.session_state.model_results)
           st.dataframe(summary_df)
           
           if not summary_df.empty:
               best_model = summary_df.iloc[0]['Model']
               best_score = summary_df.iloc[0]['Overall Score']
               st.success(f"Best Model: {best_model} with Overall Score: {best_score}")
           
           # K-Fold CV Comparison
           st.header("K-Fold Cross-Validation Comparison")
           cv_results = {}
           for model_type, results in st.session_state.model_results.items():
               if 'train_results' in results and 'CV_MSE_Mean' in results['train_results']:
                   cv_results[model_type.upper()] = results['train_results']
           
           if cv_results:
               cv_fig = plot_cv_scores(cv_results)
               st.plotly_chart(cv_fig, use_container_width=True)
           
           # Radar chart comparison
           st.header("Model Performance Radar Chart")
           
           test_metrics = []
           for model_type, results in st.session_state.model_results.items():
               if 'test_results' in results:
                   test_metrics.append({
                       'Model': model_type.upper(),
                       **results['test_results']
                   })
           
           if test_metrics:
               radar_fig = plot_model_comparison_radar(pd.DataFrame(test_metrics))
               st.plotly_chart(radar_fig, use_container_width=True)
           
           # Error comparison
           st.header("Error Metrics Comparison")
           metrics = ["RMSE", "MAE", "MAPE", "R²"]
           metric = st.selectbox("Select Metric", metrics)
           
           if test_metrics:
               metrics_df = pd.DataFrame(test_metrics)
               
               fig = px.bar(
                   metrics_df,
                   x='Model',
                   y=metric,
                   color='Model',
                   title=f'{metric} Comparison Across Models'
               )
               
               if metric == 'R²':
                   fig.update_yaxes(range=[0, 1])
               
               st.plotly_chart(fig, use_container_width=True)
           
           # Forecast comparison
           st.header("Forecast Comparison")
           
           available_models = list(st.session_state.model_results.keys())
           models_to_compare = st.multiselect(
               "Select Models to Compare",
               available_models,
               default=available_models[:min(3, len(available_models))]
           )
           
           if models_to_compare:
               fig = go.Figure()
               
               fig.add_trace(go.Scatter(
                   x=st.session_state.daily_data['Date'],
                   y=st.session_state.daily_data['PM10'],
                   mode='lines',
                   name='Historical Data',
                   line=dict(color='blue', width=1)
               ))
               
               colors = ['red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta']
               for i, model_type in enumerate(models_to_compare):
                   forecast_df = st.session_state.model_results[model_type]['forecast']
                   fig.add_trace(go.Scatter(
                       x=forecast_df['Date'],
                       y=forecast_df['PM10'],
                       mode='lines',
                       name=f'{model_type.upper()} Forecast',
                       line=dict(color=colors[i % len(colors)], width=2)
                   ))
               
               fig.add_shape(
                   type="line",
                   x0=st.session_state.daily_data['Date'].iloc[-1],
                   y0=0,
                   x1=st.session_state.daily_data['Date'].iloc[-1],
                   y1=st.session_state.daily_data['PM10'].max() * 1.1,
                   line=dict(color="black", width=1, dash="dash")
               )
               
               fig.update_layout(
                   title='Model Forecast Comparison',
                   xaxis_title='Date',
                   yaxis_title='PM10 (µg/m³)',
                   height=600
               )
               
               st.plotly_chart(fig, use_container_width=True)
       
       else:
           if not st.session_state.data:
               st.info("Please upload a dataset using the sidebar.")
           elif not st.session_state.model_results:
               st.info("Please train at least one model in the 'Model Training' tab.")
   
   # Ensemble Modeling Page (unchanged but could show scaling info)
   elif page == "Ensemble Modeling":
       st.title("🎯 Ensemble Modeling")
       
       if st.session_state.data is not None and len(st.session_state.model_results) >= 2:
           st.header("Create Ensemble Model")
           
           available_models = [m for m in st.session_state.model_results.keys() 
                             if m not in ['arima', 'sarima', 'prophet', 'lstm', 'gru', 'ann']]
           
           selected_models = st.multiselect(
               "Select Models for Ensemble",
               available_models,
               default=available_models[:min(3, len(available_models))]
           )
           
           # Show scaling compatibility warning
           if len(selected_models) >= 2:
               scaling_methods = []
               for model in selected_models:
                   if 'scaling_info' in st.session_state.model_results[model]:
                       scaling_info = st.session_state.model_results[model]['scaling_info']
                       scaling_methods.append(scaling_info['feature_scaling_method'])
               
               unique_scaling = set([s for s in scaling_methods if s is not None])
               if len(unique_scaling) > 1:
                   st.warning("⚠️ **Scaling Compatibility Warning:** Selected models use different feature scaling methods. "
                             "This may affect ensemble performance. Consider retraining models with consistent scaling.")
           
           if len(selected_models) >= 2:
               if st.button("Create Ensemble"):
                   with st.spinner("Creating ensemble model..."):
                       try:
                           ensemble_models = {
                               name: st.session_state.model_results[name]['model']
                               for name in selected_models
                           }
                           
                           ensemble = EnsembleModel(ensemble_models)
                           
                           val_data = st.session_state.val_data
                           val_with_lags = add_lag_features(val_data, 'PM10')
                           
                           feature_cols = st.session_state.model_results[selected_models[0]]['feature_names']
                           
                           X_val, y_val = create_feature_target_arrays(val_with_lags, 'PM10', feature_cols)
                           
                           if selected_models[0] in ['linear', 'ridge', 'svr', 'knn', 'naive_bayes']:
                               scaler = StandardScaler()
                               X_val = scaler.fit_transform(X_val)
                           
                           optimal_weights = ensemble.optimize_weights(X_val, y_val)
                           
                           st.session_state.ensemble_model = ensemble
                           st.success("Ensemble model created successfully!")
                           
                           st.subheader("Optimized Model Weights")
                           weights_fig = plot_ensemble_weights(optimal_weights)
                           st.plotly_chart(weights_fig, use_container_width=True)
                           
                           st.subheader("Ensemble Performance")
                           
                           test_data = st.session_state.test_data
                           test_with_lags = add_lag_features(test_data, 'PM10')
                           X_test, y_test = create_feature_target_arrays(test_with_lags, 'PM10', feature_cols)
                           
                           if selected_models[0] in ['linear', 'ridge', 'svr', 'knn', 'naive_bayes']:
                               X_test = scaler.transform(X_test)
                           
                           ensemble_pred = ensemble.predict(X_test)
                           
                           ensemble_results = evaluate_model(y_test, ensemble_pred, "Ensemble")
                           ensemble_results['Overall_Score'] = calculate_overall_score(ensemble_results)
                           
                           st.dataframe(pd.DataFrame([ensemble_results]))
                           
                           st.subheader("Ensemble vs Individual Models")
                           
                           comparison_data = []
                           for model_name in selected_models:
                               model_results = st.session_state.model_results[model_name]['test_results']
                               comparison_data.append({
                                   'Model': model_name.upper(),
                                   'RMSE': model_results['RMSE'],
                                   'MAE': model_results['MAE'],
                                   'R²': model_results['R²'],
                                   'Overall Score': model_results['Overall_Score']
                               })
                           
                           comparison_data.append({
                               'Model': 'ENSEMBLE',
                               'RMSE': ensemble_results['RMSE'],
                               'MAE': ensemble_results['MAE'],
                               'R²': ensemble_results['R²'],
                               'Overall Score': ensemble_results['Overall_Score']
                           })
                           
                           comparison_df = pd.DataFrame(comparison_data)
                           comparison_df = comparison_df.sort_values('Overall Score', ascending=False)
                           
                           st.dataframe(comparison_df)
                           
                           if comparison_df.iloc[0]['Model'] == 'ENSEMBLE':
                               st.success("🎉 Ensemble model achieves the best performance!")
                       except Exception as e:
                           st.error(f"Error creating ensemble: {str(e)}")
           else:
               st.warning("Please select at least 2 models for ensemble.")
       
       else:
           if not st.session_state.data:
               st.info("Please upload a dataset using the sidebar.")
           elif len(st.session_state.model_results) < 2:
               st.info("Please train at least 2 models to create an ensemble.")
   
   # Forecasting Dashboard Page (updated with scaling information)
   elif page == "Forecasting Dashboard":
       st.title("📊 Forecasting Dashboard")
       
       if st.session_state.data is not None and st.session_state.model_results:
           st.info(f"Models trained with split: Train {st.session_state.train_ratio*100:.1f}% | "
                  f"Val {st.session_state.val_ratio*100:.1f}% | "
                  f"Test {st.session_state.test_ratio*100:.1f}%")
           
           # Executive Summary
           st.header("Executive Summary")
           
           summary_df = create_performance_summary_table(st.session_state.model_results)
           
           if not summary_df.empty:
               best_model = summary_df.iloc[0]['Model']
               best_score = summary_df.iloc[0]['Overall Score']
               
               col1, col2, col3, col4 = st.columns(4)
               with col1:
                   st.metric("Best Model", best_model)
               with col2:
                   st.metric("Overall Score", f"{best_score}/100")
               with col3:
                   st.metric("Models Trained", len(st.session_state.model_results))
               with col4:
                   st.metric("Forecast Range", "1-3 Days")
           
           # Model Performance Summary
           st.header("Model Performance Summary")
           st.dataframe(summary_df)
           
           # Download options
           st.header("Download Results")
           
           download_model = st.selectbox(
               "Select Model for Download",
               list(st.session_state.model_results.keys())
           )
           
           if download_model:
               results = st.session_state.model_results[download_model]
               
               col1, col2, col3 = st.columns(3)
               
               with col1:
                   forecast_df = results['forecast']
                   csv = forecast_df.to_csv(index=False).encode('utf-8')
                   
                   st.download_button(
                       label="📥 Download Forecast",
                       data=csv,
                       file_name=f"pm10_forecast_{download_model}.csv",
                       mime="text/csv"
                   )
               
               with col2:
                   metrics_data = {
                       'Train': results['train_results'],
                       'Validation': results['val_results'],
                       'Test': results['test_results']
                   }
                   if 'scaling_info' in results:
                       metrics_data['Scaling_Info'] = results['scaling_info']
                   
                   metrics_json = json.dumps(metrics_data, indent=2)
                   
                   st.download_button(
                       label="📊 Download Metrics",
                       data=metrics_json,
                       file_name=f"pm10_metrics_{download_model}.json",
                       mime="application/json"
                   )
               
               with col3:
                   if 'best_params' in results:
                       params_json = json.dumps(results['best_params'], indent=2)
                       
                       st.download_button(
                           label="⚙️ Download Parameters",
                           data=params_json,
                           file_name=f"pm10_params_{download_model}.json",
                           mime="application/json"
                       )
           
           # Interactive Forecast Visualization
           st.header("Interactive Forecast Visualization")
           
           viz_models = st.multiselect(
               "Select Models to Visualize",
               list(st.session_state.model_results.keys()),
               default=[summary_df.iloc[0]['Model'].lower()] if not summary_df.empty else []
           )
           
           if viz_models:
               fig = go.Figure()
               
               historical = st.session_state.daily_data.tail(90)
               fig.add_trace(go.Scatter(
                   x=historical['Date'],
                   y=historical['PM10'],
                   mode='lines',
                   name='Historical',
                   line=dict(color='blue', width=2)
               ))
               
               colors = ['red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta']
               for i, model_name in enumerate(viz_models):
                   forecast = st.session_state.model_results[model_name]['forecast']
                   
                   fig.add_trace(go.Scatter(
                       x=forecast['Date'],
                       y=forecast['PM10'],
                       mode='lines',
                       name=f'{model_name.upper()}',
                       line=dict(color=colors[i % len(colors)], width=2)
                   ))
                   
                   if f'PM10_lower' in forecast.columns:
                       fig.add_trace(go.Scatter(
                           x=forecast['Date'],
                           y=forecast['PM10_upper'],
                           mode='lines',
                           showlegend=False,
                           line=dict(color=colors[i % len(colors)], width=0)
                       ))
                       fig.add_trace(go.Scatter(
                           x=forecast['Date'],
                           y=forecast['PM10_lower'],
                           mode='lines',
                           showlegend=False,
                           line=dict(color=colors[i % len(colors)], width=0),
                           fill='tonexty',
                           fillcolor=f'rgba({i*50}, {i*30}, {i*40}, 0.2)'
                       ))
               
               forecast_start_date = pd.Timestamp(historical['Date'].iloc[-1])
               
               fig.add_vline(
                   x=forecast_start_date.timestamp() * 1000,
                   line_dash="dash",
                   line_color="gray",
                   annotation_text="Forecast Start"
               )
               
               fig.update_layout(
                   title="PM10 Forecast Comparison (1-3 Days)",
                   xaxis_title="Date",
                   yaxis_title="PM10 (µg/m³)",
                   height=600,
                   hovermode='x unified'
               )
               
               st.plotly_chart(fig, use_container_width=True)
           
           # Model Insights
           st.header("Key Insights")
           
           if viz_models:
               st.subheader("Forecast Analysis (Short-term: 1-3 Days)")
               
               for model_name in viz_models[:3]:
                   forecast = st.session_state.model_results[model_name]['forecast']
                   
                   mean_forecast = forecast['PM10'].mean()
                   trend = "increasing" if forecast['PM10'].iloc[-1] > forecast['PM10'].iloc[0] else "decreasing"
                   max_val = forecast['PM10'].max()
                   min_val = forecast['PM10'].min()
                   
                   st.write(f"**{model_name.upper()} Model:**")
                   st.write(f"- Average forecasted PM10: {mean_forecast:.1f} µg/m³")
                   st.write(f"- Short-term trend: {trend}")
                   st.write(f"- Range: {min_val:.1f} - {max_val:.1f} µg/m³")
                   st.write("")
           
           # Feature Importance Summary
           if any('importance_df' in results for results in st.session_state.model_results.values()):
               st.header("Feature Importance Summary")
               
               # Collect feature importance from all models
               importance_data = []
               for model_name, results in st.session_state.model_results.items():
                   if 'importance_df' in results and results['importance_df'] is not None:
                       imp_df = results['importance_df'].copy()
                       imp_df['Model'] = model_name.upper()
                       importance_data.append(imp_df)
               
               if importance_data:
                   # Combine and show top features
                   combined_importance = pd.concat(importance_data)
                   top_features = combined_importance.groupby('Feature')['Importance'].mean().sort_values(ascending=False).head(10)
                   
                   fig = px.bar(
                       x=top_features.values,
                       y=top_features.index,
                       orientation='h',
                       title='Top 10 Most Important Features (Average)',
                       labels={'x': 'Importance', 'y': 'Feature'}
                   )
                   st.plotly_chart(fig, use_container_width=True)
           
           # Recommendations
           st.header("Recommendations")
           
           if not summary_df.empty:
               best_model_name = summary_df.iloc[0]['Model'].lower()
               best_model_score = float(summary_df.iloc[0]['Overall Score'])
               
               st.write("Based on the analysis:")
               
               # Model recommendation
               st.write(f"✅ **Recommended Model:** {best_model_name.upper()} with an overall score of {best_model_score:.1f}/100")
               
               # Performance insights
               if best_model_score > 80:
                   st.write("📈 The model shows excellent performance with high accuracy and reliability for short-term forecasting.")
               elif best_model_score > 60:
                   st.write("📊 The model shows good performance for short-term forecasting but may benefit from additional feature engineering.")
               else:
                   st.write("⚠️ Consider ensemble methods or additional data sources to improve short-term forecasting performance.")
               
               # Specific recommendations based on model type
               if best_model_name in ['rf', 'xgboost', 'decision_tree']:
                   st.write("🌳 Tree-based model selected - excellent for capturing non-linear patterns in short-term PM10 variations.")
               elif best_model_name in ['lstm', 'gru', 'ann']:
                   st.write("🧠 Neural network selected - effective for complex temporal patterns in short-term air quality forecasting.")
               elif best_model_name in ['arima', 'sarima']:
                   st.write("📉 Time series model selected - suitable for short-term forecasting with clear seasonal patterns.")
               elif best_model_name in ['knn']:
                   st.write("🎯 KNN selected - performs well with local patterns for short-term predictions.")
               elif best_model_name in ['naive_bayes']:
                   st.write("🎲 Naive Bayes selected - efficient probabilistic approach for short-term forecasting.")
               
               # Data quality recommendations
               if 'outliers' in st.session_state.model_results[best_model_name]:
                   outlier_info = st.session_state.model_results[best_model_name]['outliers']
                   total_outliers = sum(outlier_info.values())
                   if total_outliers > 0:
                       st.write(f"🔍 {total_outliers} outliers were handled in the data preprocessing.")
               
               # Short-term forecasting specific recommendations
               st.write("**Short-term Forecasting Notes:**")
               st.write("- 1-3 day forecasts are most reliable for immediate air quality planning")
               st.write("- Consider real-time data updates for improved accuracy")
               st.write("- Monitor model performance regularly for optimal forecasting")
       
       else:
           if not st.session_state.data:
               st.info("Please upload a dataset using the sidebar.")
           elif not st.session_state.model_results:
               st.info("Please train at least one model in the 'Model Training' tab.")

if __name__ == "__main__":
   main()