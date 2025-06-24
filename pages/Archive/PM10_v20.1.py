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
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV, RandomizedSearchCV, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, IsolationForest, VotingRegressor
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
    page_title="PM10 Forecasting Dashboard - Fixed",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Header
st.title("🌬️ Fixed PM10 Forecasting Dashboard with Optuna")
st.write("Enhanced forecasting with categorical date features, one-hot encoding, outlier handling, and comprehensive model evaluation including neural networks with Optuna optimization.")

# Define functions
def check_missing_values(df, model_types=None):
    """Check for missing values in the dataframe and return detailed information"""
    missing_info = {}
    
    # Check each column for missing values
    for col in df.columns:
        if df[col].dtype != 'object' and col != 'Date':
            missing_count = df[col].isnull().sum()
            missing_percentage = (missing_count / len(df)) * 100
            if missing_count > 0:
                missing_info[col] = {
                    'count': missing_count,
                    'percentage': missing_percentage,
                    'indices': df[df[col].isnull()].index.tolist()
                }
    
    return missing_info

def get_models_requiring_complete_data():
    """Get list of models that require complete data (no missing values)"""
    return ['knn', 'arima', 'sarima', 'prophet', 'linear', 'ridge', 'lasso', 'svr', 'naive_bayes', 'lstm', 'ann', 'gru']

def get_models_handling_missing_data():
    """Get list of models that can handle missing data natively"""
    return ['decision_tree', 'rf', 'xgboost']

def get_neural_network_models():
    """Get list of neural network models that require special handling"""
    return ['lstm', 'ann', 'gru']

def get_ensemble_models():
    """Get list of available models for ensemble"""
    return ['decision_tree', 'knn', 'rf', 'xgboost', 'svr', 'lstm', 'ann', 'gru']

def check_model_missing_value_compatibility(model_type, missing_info):
    """Check if a specific model can handle missing values"""
    models_requiring_complete = get_models_requiring_complete_data()
    models_handling_missing = get_models_handling_missing_data()
    
    if not missing_info:
        return True, "No missing values detected - all models can proceed"
    
    if model_type in models_handling_missing:
        return True, f"{model_type.upper()} can handle missing values natively"
    elif model_type in models_requiring_complete:
        return False, f"{model_type.upper()} requires complete data (no missing values)"
    else:
        return False, f"{model_type.upper()} requires complete data (no missing values)"

def display_missing_value_error(missing_info, model_types=None):
    """Display comprehensive error message about missing values"""
    if not missing_info:
        return False
    
    st.error("🚨 **Missing Values Detected - Cannot Proceed with Selected Model**")
    
    st.write("**Missing Value Summary:**")
    
    # Create summary table
    summary_data = []
    for col, info in missing_info.items():
        summary_data.append({
            'Column': col,
            'Missing Count': info['count'],
            'Missing Percentage': f"{info['percentage']:.2f}%",
            'Total Values': len(missing_info) if isinstance(missing_info, dict) else 0
        })
    
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True)
    
    return True

def display_missing_value_warning(missing_info, model_type):
    """Display warning for models that can handle missing values"""
    if not missing_info:
        return False
    
    st.warning(f"⚠️ **Missing Values Detected - {model_type.upper()} Will Handle Them Automatically**")
    
    # Create summary table
    summary_data = []
    for col, info in missing_info.items():
        summary_data.append({
            'Column': col,
            'Missing Count': info['count'],
            'Missing Percentage': f"{info['percentage']:.2f}%"
        })
    
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True)
    
    return True

@st.cache_data
def load_data(file):
    """Load data from CSV file and preprocess with categorical date features"""
    df = pd.read_csv(file)
    
    # Convert Date to datetime - handle different date formats
    try:
        df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
    except:
        try:
            df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
        except:
            df['Date'] = pd.to_datetime(df['Date'])
    
    # Extract categorical date features
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month_name()
    df['DayOfWeek'] = df['Date'].dt.day_name()
    df['Season'] = df['Date'].dt.month.map({
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Autumn', 10: 'Autumn', 11: 'Autumn'
    })
    
    # Additional useful categorical features
    df['Weekend'] = df['Date'].dt.dayofweek.apply(lambda x: 'Weekend' if x >= 5 else 'Weekday')
    df['Quarter'] = df['Date'].dt.quarter.map({1: 'Q1', 2: 'Q2', 3: 'Q3', 4: 'Q4'})
    df['MonthPart'] = df['Date'].dt.day.apply(lambda x: 'Early' if x <= 10 else ('Mid' if x <= 20 else 'Late'))
    
    return df

@st.cache_data
def create_daily_dataset(df):
    """Create daily aggregated dataset with categorical features"""
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
    
    # Aggregate to daily values
    daily_df = df.groupby('Date').agg(agg_dict).reset_index()
    
    # Add categorical date features back
    daily_df['Year'] = daily_df['Date'].dt.year
    daily_df['Month'] = daily_df['Date'].dt.month_name()
    daily_df['DayOfWeek'] = daily_df['Date'].dt.day_name()
    daily_df['Season'] = daily_df['Date'].dt.month.map({
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Autumn', 10: 'Autumn', 11: 'Autumn'
    })
    daily_df['Weekend'] = daily_df['Date'].dt.dayofweek.apply(lambda x: 'Weekend' if x >= 5 else 'Weekday')
    daily_df['Quarter'] = daily_df['Date'].dt.quarter.map({1: 'Q1', 2: 'Q2', 3: 'Q3', 4: 'Q4'})
    daily_df['MonthPart'] = daily_df['Date'].dt.day.apply(lambda x: 'Early' if x <= 10 else ('Mid' if x <= 20 else 'Late'))
    
    return daily_df

def create_one_hot_features(df, categorical_cols=['Month', 'DayOfWeek', 'Season', 'Weekend', 'Quarter', 'MonthPart']):
    """Create one-hot encoded features for categorical variables"""
    df_encoded = df.copy()
    encoders = {}
    
    for col in categorical_cols:
        if col in df.columns:
            encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
            encoded_features = encoder.fit_transform(df[[col]])
            feature_names = [f"{col}_{category}" for category in encoder.categories_[0][1:]]
            encoded_df = pd.DataFrame(encoded_features, columns=feature_names, index=df.index)
            df_encoded = pd.concat([df_encoded, encoded_df], axis=1)
            encoders[col] = encoder
    
    return df_encoded, encoders

def apply_one_hot_encoding(df, encoders, categorical_cols=['Month', 'DayOfWeek', 'Season', 'Weekend', 'Quarter', 'MonthPart']):
    """Apply existing one-hot encoders to new data"""
    df_encoded = df.copy()
    
    for col in categorical_cols:
        if col in df.columns and col in encoders:
            encoded_features = encoders[col].transform(df[[col]])
            feature_names = [f"{col}_{category}" for category in encoders[col].categories_[0][1:]]
            encoded_df = pd.DataFrame(encoded_features, columns=feature_names, index=df.index)
            df_encoded = pd.concat([df_encoded, encoded_df], axis=1)
    
    return df_encoded

def get_categorical_features():
    """Get list of categorical date features"""
    return ['Month', 'DayOfWeek', 'Season', 'Weekend', 'Quarter', 'MonthPart']

def get_base_features():
    """Get list of base features (non-categorical)"""
    return ['Year', 'Temp', 'WS', 'WD']

def get_one_hot_feature_names(encoders):
    """Get all one-hot encoded feature names"""
    feature_names = []
    for col, encoder in encoders.items():
        feature_names.extend([f"{col}_{category}" for category in encoder.categories_[0][1:]])
    return feature_names

def detect_outliers(df, target_col, method='iqr', threshold=1.5):
    """Detect outliers using various methods"""
    outliers_dict = {}
    
    if method == 'iqr' or method == 'all':
        Q1 = df[target_col].quantile(0.25)
        Q3 = df[target_col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        outliers_iqr = (df[target_col] < lower_bound) | (df[target_col] > upper_bound)
        outliers_dict['iqr'] = outliers_iqr
    
    if method == 'zscore' or method == 'all':
        z_scores = np.abs(stats.zscore(df[target_col]))
        outliers_zscore = z_scores > 3
        outliers_dict['zscore'] = outliers_zscore
    
    if method == 'isolation' or method == 'all':
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        outliers_iso = iso_forest.fit_predict(df[[target_col]]) == -1
        outliers_dict['isolation'] = outliers_iso
    
    if method == 'moving_avg' or method == 'all':
        window = 7
        rolling_mean = df[target_col].rolling(window=window, center=True).mean()
        rolling_std = df[target_col].rolling(window=window, center=True).std()
        lower_bound = rolling_mean - 3 * rolling_std
        upper_bound = rolling_mean + 3 * rolling_std
        outliers_ma = (df[target_col] < lower_bound) | (df[target_col] > upper_bound)
        outliers_dict['moving_avg'] = outliers_ma.fillna(False)
    
    if method == 'all':
        outlier_counts = sum([outliers_dict[m].astype(int) for m in outliers_dict])
        final_outliers = outlier_counts >= 2
        return final_outliers
    else:
        return outliers_dict[method]

def handle_outliers_data(df, target_col, outliers, method='cap'):
    """Handle outliers using various methods with enhanced reporting"""
    df_clean = df.copy()
    outliers_before = int(outliers.sum())
    
    if method == 'remove':
        df_clean = df_clean[~outliers].reset_index(drop=True)
        outliers_handled = outliers_before
        outliers_remaining = 0
        values_modified = outliers_before
        
    elif method == 'cap':
        Q1 = df[target_col].quantile(0.25)
        Q3 = df[target_col].quantile(0.75)
        IQR = Q3 - Q1
        lower_cap = Q1 - 1.5 * IQR
        upper_cap = Q3 + 1.5 * IQR
        
        original_values = df_clean.loc[outliers, target_col].copy()
        df_clean.loc[outliers, target_col] = df_clean.loc[outliers, target_col].clip(
            lower=lower_cap, upper=upper_cap
        )
        
        modified_values = df_clean.loc[outliers, target_col]
        values_modified = int((original_values != modified_values).sum())
        
        outliers_after = detect_outliers(df_clean, target_col, method='iqr')
        outliers_remaining = int(outliers_after.sum())
        outliers_handled = outliers_before - outliers_remaining
    
    elif method == 'interpolate':
        values_with_outliers = df_clean.loc[outliers, target_col].notna().sum()
        df_clean.loc[outliers, target_col] = np.nan
        df_clean[target_col] = df_clean[target_col].interpolate(method='linear')
        
        outliers_after = detect_outliers(df_clean, target_col, method='iqr')
        outliers_remaining = int(outliers_after.sum())
        outliers_handled = outliers_before - outliers_remaining
        values_modified = values_with_outliers
    
    outlier_info = {
        'detected': outliers_before,
        'handled': outliers_handled,
        'remaining': outliers_remaining,
        'modified': values_modified,
        'method': method,
        'effectiveness': f"{(outliers_handled/outliers_before*100):.1f}%" if outliers_before > 0 else "0%"
    }
    
    return df_clean, outlier_info

def add_enhanced_features(df, target_col='PM10'):
    """
    Adds domain-specific time-series features like change, volatility, and trend.
    This function should be called after lag features have been created and handled.
    """
    df_out = df.copy()

    # Add PM10 change features
    df_out[f'{target_col}_daily_change'] = df_out[target_col].diff()
    df_out[f'{target_col}_weekly_change'] = df_out[target_col].diff(7)

    # Add volatility features
    # df_out[f'{target_col}_volatility_7d'] = df_out[target_col].rolling(window=7).std()
    # df_out[f'{target_col}_volatility_30d'] = df_out[target_col].rolling(window=30).std()

    # Add trend feature (slope of the last 7 days)
    # def get_trend(x):
    #     if len(x.dropna()) < 2:
    #         return 0
    #     return np.polyfit(range(len(x.dropna())), x.dropna(), 1)[0]

    # df_out[f'{target_col}_trend_7d'] = df_out[target_col].rolling(window=7).apply(get_trend, raw=False)
    
    # Fill NaNs created by these new features
    df_out = df_out.fillna(method='bfill').fillna(method='ffill')
    
    # If any NaNs still exist, fill with 0
    df_out = df_out.fillna(0)

    return df_out

def prepare_data_for_distance_models(X_train, X_val, X_test, y_train):
    """
    Ensure absolutely no NaN values and apply robust scaling.
    """
    # Use SimpleImputer to fill any potential remaining NaN values with the median
    imputer = SimpleImputer(strategy='median')
    X_train_clean = imputer.fit_transform(X_train)
    X_val_clean = imputer.transform(X_val)
    X_test_clean = imputer.transform(X_test)
    
    # Clean target variable as well, just in case
    y_train_clean = np.nan_to_num(y_train, nan=np.nanmedian(y_train))

    # Apply RobustScaler, which is less sensitive to outliers
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train_clean)
    X_val_scaled = scaler.transform(X_val_clean)
    X_test_scaled = scaler.transform(X_test_clean)

    # Final verification to ensure no invalid values exist
    assert not np.any(np.isnan(X_train_scaled))
    assert not np.any(np.isinf(X_train_scaled))
    assert not np.any(np.isnan(X_val_scaled))
    assert not np.any(np.isinf(X_val_scaled))
    assert not np.any(np.isnan(X_test_scaled))
    assert not np.any(np.isinf(X_test_scaled))

    return X_train_scaled, X_val_scaled, X_test_scaled, y_train_clean, scaler

def add_lag_features(df, target_col, lag_days=[1, 2, 3, 7, 14], show_message=True):
    """Add lag features with ULTRA-ROBUST NaN prevention"""
    df_copy = df.copy()
    n_rows = len(df_copy)
    
    # Calculate minimum data requirements
    max_lag = max(lag_days) if lag_days else 0
    max_window = 30
    min_required = max_lag + max_window + 5
    
    # Adapt feature creation based on available data
    if n_rows < min_required:
        if show_message:
            st.info(f"📊 **Adapting features for dataset size ({n_rows} days)**")
        
        if n_rows < 50:
            lag_days = [1, 2, 3]
            window_sizes = [min(7, n_rows // 3)] if n_rows >= 15 else []
        elif n_rows < 100:
            lag_days = [1, 2, 3, 7]
            window_sizes = [7, min(14, n_rows // 3)]
        else:
            window_sizes = [7, 14, min(30, n_rows // 3)]
    else:
        window_sizes = [7, 14, 30]
    
    # Add lag features
    features_added = []
    for lag in lag_days:
        if lag < n_rows - 5:
            df_copy[f'{target_col}_lag_{lag}'] = df_copy[target_col].shift(lag)
            features_added.append(f'lag_{lag}')
    
    # Add rolling statistics
    for window in window_sizes:
        if window <= n_rows // 2:
            min_periods = max(1, window // 2)
            
            df_copy[f'{target_col}_rolling_mean_{window}'] = df_copy[target_col].rolling(
                window=window, min_periods=min_periods
            ).mean().shift(1)
            
            if window >= 3:
                df_copy[f'{target_col}_rolling_std_{window}'] = df_copy[target_col].rolling(
                    window=window, min_periods=max(2, min_periods)
                ).std().shift(1)
            
            df_copy[f'{target_col}_rolling_max_{window}'] = df_copy[target_col].rolling(
                window=window, min_periods=min_periods
            ).max().shift(1)
            df_copy[f'{target_col}_rolling_min_{window}'] = df_copy[target_col].rolling(
                window=window, min_periods=min_periods
            ).min().shift(1)
            
            features_added.append(f'rolling_{window}')
    
    # Add exponentially weighted moving average
    if n_rows >= 14:
        for span in [7, 14]:
            if span <= n_rows // 2:
                df_copy[f'{target_col}_ewma_{span}'] = df_copy[target_col].ewm(
                    span=span, adjust=False
                ).mean().shift(1)
                features_added.append(f'ewma_{span}')
    
    # COMPREHENSIVE NaN handling
    initial_length = len(df_copy)
    
    # Drop rows where target is NaN
    target_nan_count = df_copy[target_col].isnull().sum()
    if target_nan_count > 0:
        if show_message:
            st.info(f"Removing {target_nan_count} rows with missing target values")
        df_copy = df_copy.dropna(subset=[target_col])
    
    # Handle feature NaN values systematically
    feature_cols = [col for col in df_copy.columns if target_col in col and col != target_col]
    
    for col in feature_cols:
        if df_copy[col].isnull().any():
            if 'lag_' in col:
                df_copy[col] = df_copy[col].fillna(method='bfill', limit=5)
                df_copy[col] = df_copy[col].fillna(method='ffill', limit=5)
                if df_copy[col].isnull().any():
                    df_copy[col] = df_copy[col].fillna(df_copy[target_col].median())
            
            elif 'rolling_mean' in col:
                expanding_mean = df_copy[target_col].expanding().mean().shift(1)
                df_copy[col] = df_copy[col].fillna(expanding_mean)
                if df_copy[col].isnull().any():
                    df_copy[col] = df_copy[col].fillna(df_copy[target_col].median())
            
            elif 'rolling_std' in col:
                expanding_std = df_copy[target_col].expanding().std().shift(1)
                df_copy[col] = df_copy[col].fillna(expanding_std)
                df_copy[col] = df_copy[col].fillna(0.1)
            
            elif 'rolling_max' in col:
                expanding_max = df_copy[target_col].expanding().max().shift(1)
                df_copy[col] = df_copy[col].fillna(expanding_max)
                if df_copy[col].isnull().any():
                    df_copy[col] = df_copy[col].fillna(df_copy[target_col].median())
            
            elif 'rolling_min' in col:
                expanding_min = df_copy[target_col].expanding().min().shift(1)
                df_copy[col] = df_copy[col].fillna(expanding_min)
                if df_copy[col].isnull().any():
                    df_copy[col] = df_copy[col].fillna(df_copy[target_col].median())
            
            elif 'ewma_' in col:
                df_copy[col] = df_copy[col].fillna(method='ffill')
                if df_copy[col].isnull().any():
                    df_copy[col] = df_copy[col].fillna(df_copy[target_col].median())
            
            # ULTIMATE FALLBACK
            if df_copy[col].isnull().any():
                col_median = df_copy[col].median()
                if pd.isna(col_median):
                    col_median = df_copy[target_col].median()
                    if pd.isna(col_median):
                        col_median = 0.0
                df_copy[col] = df_copy[col].fillna(col_median)
    
    # Final cleanup - remove any rows that still have NaN
    before_final_dropna = len(df_copy)
    remaining_nan_cols = df_copy.columns[df_copy.isnull().any()].tolist()
    
    if remaining_nan_cols:
        for col in remaining_nan_cols:
            if col == target_col:
                continue
            else:
                global_median = df_copy[col].median()
                if pd.isna(global_median):
                    fill_value = 0.0
                else:
                    fill_value = global_median
                df_copy[col] = df_copy[col].fillna(fill_value)
        
        df_copy = df_copy.dropna()
    
    # Handle infinite values
    inf_cols = []
    for col in df_copy.columns:
        if df_copy[col].dtype in ['float64', 'int64']:
            if np.isinf(df_copy[col]).any():
                inf_cols.append(col)
                df_copy[col] = df_copy[col].replace([np.inf, -np.inf], 
                    [df_copy[col][np.isfinite(df_copy[col])].max(), 
                     df_copy[col][np.isfinite(df_copy[col])].min()])
    
    df_copy = df_copy.reset_index(drop=True)
    final_length = len(df_copy)
    
    # Final validation
    if df_copy.isnull().any().any():
        df_copy = df_copy.fillna(0.0)
    
    # Assertions
    assert not df_copy.isnull().any().any(), f"NaN values still present after cleaning!"
    assert not np.isinf(df_copy.select_dtypes(include=[np.number]).values).any(), "Infinite values still present!"
    
    if show_message:
        rows_lost = initial_length - final_length
        if rows_lost > 0:
            st.info(f"📉 Feature engineering removed {rows_lost} rows - {final_length} rows remaining")
        st.success("✅ **ULTRA-ROBUST data cleaning completed**")
    
    return df_copy

def split_data(df, target_col, train_size=0.65, val_size=0.15, test_size=0.20):
    """Split data into train, validation, and test sets with user-defined ratios"""
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Ensure ratios sum to 1.0
    total_ratio = train_size + val_size + test_size
    if abs(total_ratio - 1.0) > 0.01:
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
    
    assert len(train) + len(val) + len(test) == n, "Split failed: data points were lost"
    
    return train, val, test

def create_feature_target_arrays_with_encoding(df, target_col, categorical_cols, base_features, lag_features, encoders=None):
    """Extract feature and target arrays with one-hot encoding"""
    
    # Apply one-hot encoding
    if encoders is None:
        df_encoded, encoders = create_one_hot_features(df, categorical_cols)
    else:
        df_encoded = apply_one_hot_encoding(df, encoders, categorical_cols)
    
    # Get all feature columns
    one_hot_features = get_one_hot_feature_names(encoders)
    all_features = base_features + lag_features + one_hot_features
    
    # Ensure all feature columns exist
    available_features = [col for col in all_features if col in df_encoded.columns]
    
    X = df_encoded[available_features].values
    y = df_encoded[target_col].values

    return X, y, available_features, encoders

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
   """Scale target data for neural network models"""
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
   
   # Handle MAPE carefully
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

def perform_kfold_cv(model, X, y, n_folds=5, scoring='neg_mean_squared_error'):
  """Perform cross-validation using TimeSeriesSplit"""
  # For time series data, TimeSeriesSplit is the correct validation strategy
  cv = TimeSeriesSplit(n_splits=n_folds)
  
  scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
  return -scores

def get_hyperparameter_space(model_type):
   """
   Get hyperparameter search space for each model type.
   ADJUSTED FOR REDUCING OVERFITTING by promoting simpler models and stronger regularization.
   """
   param_spaces = {
       'linear': {
           'fit_intercept': [True, False]
       },
       'ridge': {
           # Increased alpha range for stronger regularization
           'alpha': [0.1, 1, 10, 50, 100],
           'fit_intercept': [True, False]
       },
       'lasso': {
           # Increased alpha range for stronger regularization
           'alpha': [0.01, 0.1, 1, 5, 10],
           'fit_intercept': [True, False]
       },
       'decision_tree': {
           # Reduced max_depth to prevent overly complex trees
           'max_depth': [4, 6, 8, 10],
           # Increased min_samples to ensure nodes are more general
           'min_samples_split': [40, 60, 80],
           'min_samples_leaf': [20, 30, 40],
           # Increased ccp_alpha for stronger post-pruning regularization
           'ccp_alpha': [0.0, 0.005, 0.01, 0.02]
       },
       'knn': {
           # Using a higher and narrower range of neighbors to encourage smoother decision boundaries
           'n_neighbors': [20, 30, 40, 50],
           'weights': ['distance'],
           'metric': ['euclidean', 'manhattan']
       },
       'rf': {
           'n_estimators': [100, 200],
           # Reduced max_depth for simpler individual trees
           'max_depth': [6, 8, 10, 12],
           # Increased min_samples for more robust splits and leaves
           'min_samples_split': [30, 50, 70],
           'min_samples_leaf': [15, 25, 35],
           # Kept feature/sample subsampling as they are excellent regularizers
           'max_features': ['sqrt', 'log2'],
           'max_samples': [0.7, 0.8]
       },
       'xgboost': {
           'n_estimators': [100, 200],
           'learning_rate': [0.01, 0.05, 0.1],
           # Strictly limited max_depth, the most common cause of overfitting in XGBoost
           'max_depth': [3, 4, 5],
           'subsample': [0.7, 0.8],
           'colsample_bytree': [0.7, 0.8],
           # Increased L1 and L2 regularization penalties
           'reg_alpha': [0.1, 0.5, 1],
           'reg_lambda': [1, 2, 5],
           # Increased min_child_weight to create more general nodes
           'min_child_weight': [3, 5, 7]
       },
       'svr': {
           # Reduced C values; smaller C means stronger regularization for SVR
           'C': [0.01, 0.1, 1.0],
           # Increased epsilon to allow a wider margin of error, reducing overfitting to noise
           'epsilon': [0.2, 0.5, 1.0],
           'kernel': ['linear', 'rbf'],
           'gamma': ['scale']
       },
       'naive_bayes': {
           'var_smoothing': [1e-9, 1e-8, 1e-7]
       }
   }
   
   return param_spaces.get(model_type, {})

def tune_hyperparameters(model_class, X_train, y_train, param_space, cv_folds=3, scoring='neg_mean_squared_error', method='grid'):
   """Tune hyperparameters using grid or randomized search with TimeSeriesSplit"""
   if len(param_space) == 0:
       return model_class(), {}, 0
   
   # Always use TimeSeriesSplit for this application
   cv = TimeSeriesSplit(n_splits=cv_folds)
       
   if method == 'grid':
       search = GridSearchCV(
           model_class(),
           param_space,
           cv=cv,  # Now correctly uses TimeSeriesSplit
           scoring=scoring,
           n_jobs=-1,
           verbose=0
       )
   else:
       search = RandomizedSearchCV(
           model_class(),
           param_space,
           n_iter=20,
           cv=cv,  # Now correctly uses TimeSeriesSplit
           scoring=scoring,
           n_jobs=-1,
           verbose=0,
           random_state=42
       )
   
   search.fit(X_train, y_train)
   return search.best_estimator_, search.best_params_, search.best_score_

# ===== OPTUNA OPTIMIZATION FOR NEURAL NETWORKS =====
def optimize_lstm_with_optuna(X_train, y_train, X_val, y_val, time_steps=7, n_trials=20):
  """
  Optimize LSTM hyperparameters using Optuna.
  ADJUSTED to search for simpler models and higher dropout to prevent overfitting.
  """
  
  def objective(trial):
      # Suggest hyperparameters
      # Reduced number of units to prevent overly complex layers
      units = trial.suggest_categorical('units', [16, 32, 50, 64])
      # Increased dropout range for stronger regularization
      dropout = trial.suggest_float('dropout', 0.2, 0.6)
      batch_size = trial.suggest_categorical('batch_size', [32, 64])
      learning_rate = trial.suggest_float('learning_rate', 1e-4, 5e-3, log=True)
      
      try:
          # Create sequences
          X_train_seq, y_train_seq = create_sequences(X_train, y_train, time_steps)
          X_val_seq, y_val_seq = create_sequences(X_val, y_val, time_steps)
          
          if len(X_train_seq) == 0 or len(X_val_seq) == 0:
              return float('inf')
          
          # Build model
          model = Sequential([
              LSTM(units, return_sequences=True, input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])),
              Dropout(dropout),
              LSTM(units // 2, return_sequences=False),
              Dropout(dropout),
              Dense(units // 2),
              Dense(1)
          ])
          
          model.compile(
              optimizer=Adam(learning_rate=learning_rate),
              loss='mse',
              metrics=['mae']
          )
          
          # Train model with early stopping
          early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
          
          history = model.fit(
              X_train_seq, y_train_seq,
              epochs=100,
              batch_size=batch_size,
              validation_data=(X_val_seq, y_val_seq),
              callbacks=[early_stopping],
              verbose=0
          )
          
          # Return validation loss
          val_loss = min(history.history['val_loss'])
          return val_loss
          
      except Exception as e:
          return float('inf')
  
  # Create Optuna study
  study = optuna.create_study(direction='minimize')
  study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
  
  return study.best_params, study.best_value

def optimize_ann_with_optuna(X_train, y_train, X_val, y_val, n_trials=20):
  """
  Optimize ANN hyperparameters using Optuna.
  ADJUSTED to search for simpler architectures and higher dropout.
  """
  
  def objective(trial):
      # Suggest hyperparameters
      # Limited to fewer layers
      hidden_layers = trial.suggest_int('hidden_layers', 1, 2)
      # Reduced number of neurons
      neurons = trial.suggest_categorical('neurons', [32, 50, 64])
      # Increased dropout range
      dropout = trial.suggest_float('dropout', 0.2, 0.6)
      learning_rate = trial.suggest_float('learning_rate', 1e-4, 5e-3, log=True)
      batch_size = trial.suggest_categorical('batch_size', [32, 64])
      
      try:
          # Build model
          model = Sequential()
          model.add(Dense(neurons, activation='relu', input_dim=X_train.shape[1]))
          model.add(Dropout(dropout))
          
          for _ in range(hidden_layers - 1):
              model.add(Dense(neurons // 2, activation='relu'))
              model.add(Dropout(dropout))
          
          model.add(Dense(1))
          
          model.compile(
              optimizer=Adam(learning_rate=learning_rate),
              loss='mse',
              metrics=['mae']
          )
          
          # Train model with early stopping
          early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
          
          history = model.fit(
              X_train, y_train,
              epochs=100,
              batch_size=batch_size,
              validation_data=(X_val, y_val),
              callbacks=[early_stopping],
              verbose=0
          )
          
          # Return validation loss
          val_loss = min(history.history['val_loss'])
          return val_loss
          
      except Exception as e:
          return float('inf')
  
  # Create Optuna study
  study = optuna.create_study(direction='minimize')
  study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
  
  return study.best_params, study.best_value
  
def optimize_gru_with_optuna(X_train, y_train, X_val, y_val, time_steps=7, n_trials=20):
  """Optimize GRU hyperparameters using Optuna"""
  
  def objective(trial):
      # Suggest hyperparameters
      units = trial.suggest_categorical('units', [32, 64, 128])
      dropout = trial.suggest_float('dropout', 0.1, 0.5)
      batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
      learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
      
      try:
          # Create sequences
          X_train_seq, y_train_seq = create_sequences(X_train, y_train, time_steps)
          X_val_seq, y_val_seq = create_sequences(X_val, y_val, time_steps)
          
          if len(X_train_seq) == 0 or len(X_val_seq) == 0:
              return float('inf')
          
          # Build model
          model = Sequential([
              GRU(units, return_sequences=True, input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])),
              Dropout(dropout),
              GRU(units // 2, return_sequences=False),
              Dropout(dropout),
              Dense(25),
              Dense(1)
          ])
          
          model.compile(
              optimizer=Adam(learning_rate=learning_rate),
              loss='mse',
              metrics=['mae']
          )
          
          # Train model
          early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
          
          history = model.fit(
              X_train_seq, y_train_seq,
              epochs=100,
              batch_size=batch_size,
              validation_data=(X_val_seq, y_val_seq),
              callbacks=[early_stopping],
              verbose=0
          )
          
          # Return validation loss
          val_loss = min(history.history['val_loss'])
          return val_loss
          
      except Exception as e:
          return float('inf')
  
  # Create Optuna study
  study = optuna.create_study(direction='minimize')
  study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
  
  return study.best_params, study.best_value

# ===== MODEL TRAINING FUNCTIONS =====
def train_decision_tree(X_train, y_train, X_val, y_val, tune_params=True, n_folds=5):
   """Train Decision Tree with hyperparameter tuning and TimeSeriesSplit CV"""
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
   """Train K-Nearest Neighbors with ROBUST NaN handling and scaling."""
   st.info("🔍 **KNN Data Prep**: Applying robust imputation and scaling...")
   
   # Use the new, robust preparation function
   X_train_scaled, X_val_scaled, _, y_train_clean, scaler = prepare_data_for_distance_models(
       X_train, X_val, X_train, y_train # Pass X_train for test set as it's not used here
   )
   
   # Note: Validation y is also cleaned implicitly for evaluation if needed
   y_val_clean = np.nan_to_num(y_val, nan=np.nanmedian(y_train))

   st.success("✅ **KNN Data Prep Complete** - Data is cleaned and scaled.")
   
   if tune_params:
       param_space = get_hyperparameter_space('knn')
       model, best_params, _ = tune_hyperparameters(
           KNeighborsRegressor, X_train_scaled, y_train_clean, param_space, 
           method='random', cv_folds=n_folds
       )
   else:
       model = KNeighborsRegressor(n_neighbors=5, weights='distance')
       model.fit(X_train_scaled, y_train_clean)
       best_params = {'n_neighbors': 5}
   
   cv_scores = perform_kfold_cv(model, X_train_scaled, y_train_clean, n_folds=n_folds)
   y_train_pred = model.predict(X_train_scaled)
   y_val_pred = model.predict(X_val_scaled)
   
   train_results = evaluate_model(y_train_clean, y_train_pred, "KNN (Train)")
   val_results = evaluate_model(y_val_clean, y_val_pred, "KNN (Validation)")
   
   train_results['CV_MSE_Mean'] = np.mean(cv_scores)
   train_results['CV_MSE_Std'] = np.std(cv_scores)
   
   st.success("✅ **KNN Training Successful!**")
   
   # Return the scaler along with the model
   return model, train_results, val_results, best_params, scaler

def train_random_forest(X_train, y_train, X_val, y_val, tune_params=True, n_folds=5):
   """Train Random Forest with hyperparameter tuning and TimeSeriesSplit CV"""
   if tune_params:
       param_space = get_hyperparameter_space('rf')

       # Ensure bootstrap is included in param_space if not already
       if 'bootstrap' not in param_space:
           param_space['bootstrap'] = [True]

       model, best_params, _ = tune_hyperparameters(
           RandomForestRegressor, X_train, y_train, param_space, 
           method='random', cv_folds=n_folds
       )
   else:
       model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, bootstrap = True)
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
   """Train XGBoost with hyperparameter tuning and TimeSeriesSplit CV"""
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
       """Train SVR with ROBUST NaN handling and scaling."""
       
       st.info("🔍 **SVR Data Prep**: Applying robust imputation and scaling...")

       # Use the new, robust preparation function
       X_train_scaled, X_val_scaled, _, y_train_clean, scaler = prepare_data_for_distance_models(
           X_train, X_val, X_train, y_train
       )

       # Note: Validation y is also cleaned implicitly for evaluation
       y_val_clean = np.nan_to_num(y_val, nan=np.nanmedian(y_train))

       st.success("✅ **SVR Data Prep Complete** - Data is cleaned and scaled.")
       
       if tune_params:
           param_space = get_hyperparameter_space('svr')
           model, best_params, _ = tune_hyperparameters(
               SVR, X_train_scaled, y_train_clean, param_space, 
               method='random', cv_folds=n_folds
           )
       else:
           model = SVR(C=1.0, epsilon=0.1)
           model.fit(X_train_scaled, y_train_clean)
           best_params = {'C': 1.0, 'epsilon': 0.1}

       cv_scores = perform_kfold_cv(model, X_train_scaled, y_train_clean, n_folds=n_folds)
       y_train_pred = model.predict(X_train_scaled)
       y_val_pred = model.predict(X_val_scaled)
       
       train_results = evaluate_model(y_train_clean, y_train_pred, "SVR (Train)")
       val_results = evaluate_model(y_val_clean, y_val_pred, "SVR (Validation)")
       
       train_results['CV_MSE_Mean'] = np.mean(cv_scores)
       train_results['CV_MSE_Std'] = np.std(cv_scores)
       
       st.success("✅ **SVR Training Successful!**")
       
       # Return the scaler along with the model
       return model, train_results, val_results, best_params, scaler

def train_linear_regression(X_train, y_train, X_val, y_val, tune_params=True, n_folds=5):
  """Train Linear Regression with K-fold CV"""
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

def train_ridge(X_train, y_train, X_val, y_val, tune_params=True, n_folds=5):
  """Train Ridge Regression with hyperparameter tuning and K-fold CV"""
  if tune_params:
      param_space = get_hyperparameter_space('ridge')
      model, best_params, _ = tune_hyperparameters(
          Ridge, X_train, y_train, param_space, 
          method='grid', cv_folds=n_folds
      )
  else:
      model = Ridge(alpha=1.0)
      model.fit(X_train, y_train)
      best_params = {'alpha': 1.0}

  cv_scores = perform_kfold_cv(model, X_train, y_train, n_folds=n_folds)
  y_train_pred = model.predict(X_train)
  y_val_pred = model.predict(X_val)
  
  train_results = evaluate_model(y_train, y_train_pred, "Ridge (Train)")
  val_results = evaluate_model(y_val, y_val_pred, "Ridge (Validation)")
  
  train_results['CV_MSE_Mean'] = np.mean(cv_scores)
  train_results['CV_MSE_Std'] = np.std(cv_scores)
  
  return model, train_results, val_results, best_params

def train_lasso(X_train, y_train, X_val, y_val, tune_params=True, n_folds=5):
  """Train Lasso Regression with hyperparameter tuning and K-fold CV"""
  if tune_params:
      param_space = get_hyperparameter_space('lasso')
      model, best_params, _ = tune_hyperparameters(
          Lasso, X_train, y_train, param_space, 
          method='grid', cv_folds=n_folds
      )
  else:
      model = Lasso(alpha=1.0)
      model.fit(X_train, y_train)
      best_params = {'alpha': 1.0}

  cv_scores = perform_kfold_cv(model, X_train, y_train, n_folds=n_folds)
  y_train_pred = model.predict(X_train)
  y_val_pred = model.predict(X_val)
  
  train_results = evaluate_model(y_train, y_train_pred, "Lasso (Train)")
  val_results = evaluate_model(y_val, y_val_pred, "Lasso (Validation)")
  
  train_results['CV_MSE_Mean'] = np.mean(cv_scores)
  train_results['CV_MSE_Std'] = np.std(cv_scores)
  
  return model, train_results, val_results, best_params

def create_sequences(X, y, time_steps=7):
  """Create sequences for LSTM/GRU models - FIXED"""
  if len(X) < time_steps + 1:
      return np.array([]), np.array([])
  
  Xs, ys = [], []
  for i in range(len(X) - time_steps):
      Xs.append(X[i:(i + time_steps)])
      ys.append(y[i + time_steps])
  return np.array(Xs), np.array(ys)

def build_lstm_model(input_shape, units=64, dropout=0.2):
  """Build LSTM model"""
  model = Sequential([
      LSTM(units, return_sequences=True, input_shape=input_shape),
      Dropout(dropout),
      LSTM(units // 2, return_sequences=False),
      Dropout(dropout),
      Dense(25),
      Dense(1)
  ])
  
  model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
  return model

def build_gru_model(input_shape, units=64, dropout=0.2):
  """Build GRU model"""
  model = Sequential([
      GRU(units, return_sequences=True, input_shape=input_shape),
      Dropout(dropout),
      GRU(units // 2, return_sequences=False),
      Dropout(dropout),
      Dense(25),
      Dense(1)
  ])
  
  model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
  return model

def build_ann_model(input_shape, hidden_layers=2, neurons=64, dropout=0.2):
  """Build ANN model"""
  model = Sequential()
  model.add(Dense(neurons, activation='relu', input_dim=input_shape))
  model.add(Dropout(dropout))
  
  for _ in range(hidden_layers - 1):
       model.add(Dense(neurons // 2, activation='relu'))
       model.add(Dropout(dropout))
   
  model.add(Dense(1))
  
  model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
  return model

def train_lstm(X_train, y_train, X_val, y_val, tune_params=True, n_folds=5, time_steps=7, use_optuna=True):
  """Train LSTM with Optuna optimization - FIXED"""
  
  st.info("🧠 **Training LSTM Model** - Creating sequences and optimizing...")
  
  # Create sequences
  X_train_seq, y_train_seq = create_sequences(X_train, y_train, time_steps)
  X_val_seq, y_val_seq = create_sequences(X_val, y_val, time_steps)
  
  if len(X_train_seq) == 0 or len(X_val_seq) == 0:
      raise ValueError("Not enough data to create sequences for LSTM")
  
  if tune_params and use_optuna:
      st.info("🔧 **Optimizing LSTM with Optuna** - This may take a few minutes...")
      best_params, best_score = optimize_lstm_with_optuna(
          X_train, y_train, X_val, y_val, time_steps=time_steps, n_trials=10
      )
      st.success(f"✅ **Optuna optimization complete** - Best validation loss: {best_score:.4f}")
  else:
      best_params = {'units': 64, 'dropout': 0.2, 'batch_size': 32, 'learning_rate': 0.001}
  
  # Build final model with best parameters
  model = Sequential([
      LSTM(best_params['units'], return_sequences=True, input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])),
      Dropout(best_params['dropout']),
      LSTM(best_params['units'] // 2, return_sequences=False),
      Dropout(best_params['dropout']),
      Dense(25),
      Dense(1)
  ])
  
  model.compile(
      optimizer=Adam(learning_rate=best_params.get('learning_rate', 0.001)),
      loss='mse',
      metrics=['mae']
  )
  
  # Train final model
  early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
  
  model.fit(
      X_train_seq, y_train_seq,
      epochs=150,
      batch_size=best_params.get('batch_size', 32),
      validation_data=(X_val_seq, y_val_seq),
      callbacks=[early_stopping],
      verbose=0
  )
  
  # Predictions - FIXED
  y_train_pred = model.predict(X_train_seq).flatten()
  y_val_pred = model.predict(X_val_seq).flatten()
  
  # Adjust actual arrays to match prediction length
  y_train_actual = y_train_seq
  y_val_actual = y_val_seq
  
  train_results = evaluate_model(y_train_actual, y_train_pred, "LSTM (Train)")
  val_results = evaluate_model(y_val_actual, y_val_pred, "LSTM (Validation)")
  
  train_results['CV_MSE_Mean'] = train_results['MSE']
  train_results['CV_MSE_Std'] = 0.0
  
  st.success("✅ **LSTM Training Complete!**")
  
  return model, train_results, val_results, best_params

def train_gru(X_train, y_train, X_val, y_val, tune_params=True, n_folds=5, time_steps=7, use_optuna=True):
  """Train GRU with Optuna optimization - FIXED"""
  
  st.info("🧠 **Training GRU Model** - Creating sequences and optimizing...")
  
  # Create sequences
  X_train_seq, y_train_seq = create_sequences(X_train, y_train, time_steps)
  X_val_seq, y_val_seq = create_sequences(X_val, y_val, time_steps)
  
  if len(X_train_seq) == 0 or len(X_val_seq) == 0:
      raise ValueError("Not enough data to create sequences for GRU")
  
  if tune_params and use_optuna:
      st.info("🔧 **Optimizing GRU with Optuna** - This may take a few minutes...")
      best_params, best_score = optimize_gru_with_optuna(
          X_train, y_train, X_val, y_val, time_steps=time_steps, n_trials=10
      )
      st.success(f"✅ **Optuna optimization complete** - Best validation loss: {best_score:.4f}")
  else:
      best_params = {'units': 64, 'dropout': 0.2, 'batch_size': 32, 'learning_rate': 0.001}
  
  # Build final model with best parameters
  model = Sequential([
      GRU(best_params['units'], return_sequences=True, input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])),
      Dropout(best_params['dropout']),
      GRU(best_params['units'] // 2, return_sequences=False),
      Dropout(best_params['dropout']),
      Dense(25),
      Dense(1)
  ])
  
  model.compile(
      optimizer=Adam(learning_rate=best_params.get('learning_rate', 0.001)),
      loss='mse',
      metrics=['mae']
  )
  
  # Train final model
  early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
  
  model.fit(
      X_train_seq, y_train_seq,
      epochs=150,
      batch_size=best_params.get('batch_size', 32),
      validation_data=(X_val_seq, y_val_seq),
      callbacks=[early_stopping],
      verbose=0
  )
  
  # Predictions - FIXED
  y_train_pred = model.predict(X_train_seq).flatten()
  y_val_pred = model.predict(X_val_seq).flatten()
  
  # Adjust actual arrays to match prediction length
  y_train_actual = y_train_seq
  y_val_actual = y_val_seq
  
  train_results = evaluate_model(y_train_actual, y_train_pred, "GRU (Train)")
  val_results = evaluate_model(y_val_actual, y_val_pred, "GRU (Validation)")
  
  train_results['CV_MSE_Mean'] = train_results['MSE']
  train_results['CV_MSE_Std'] = 0.0
  
  st.success("✅ **GRU Training Complete!**")
  
  return model, train_results, val_results, best_params

def train_ann(X_train, y_train, X_val, y_val, tune_params=True, n_folds=5, use_optuna=True):
  """Train ANN with Optuna optimization - FIXED"""
  
  st.info("🧠 **Training ANN Model** - Optimizing architecture...")
  
  # Ensure proper shapes - FIXED
  if y_train.ndim > 1:
      y_train = y_train.ravel()
  if y_val.ndim > 1:
      y_val = y_val.ravel()
  
  if tune_params and use_optuna:
      st.info("🔧 **Optimizing ANN with Optuna** - This may take a few minutes...")
      best_params, best_score = optimize_ann_with_optuna(
          X_train, y_train, X_val, y_val, n_trials=10
      )
      st.success(f"✅ **Optuna optimization complete** - Best validation loss: {best_score:.4f}")
  else:
      best_params = {'hidden_layers': 2, 'neurons': 64, 'dropout': 0.2, 'learning_rate': 0.001, 'batch_size': 32}
  
  # Build final model with best parameters
  model = Sequential()
  model.add(Dense(best_params['neurons'], activation='relu', input_dim=X_train.shape[1]))
  model.add(Dropout(best_params['dropout']))
  
  for _ in range(best_params['hidden_layers'] - 1):
      model.add(Dense(best_params['neurons'] // 2, activation='relu'))
      model.add(Dropout(best_params['dropout']))
  
  model.add(Dense(1))
  
  model.compile(
      optimizer=Adam(learning_rate=best_params.get('learning_rate', 0.001)),
      loss='mse',
      metrics=['mae']
  )
  
  # Train final model
  early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
  
  model.fit(
      X_train, y_train,
      epochs=150,
      batch_size=best_params.get('batch_size', 32),
      validation_data=(X_val, y_val),
      callbacks=[early_stopping],
      verbose=0
  )
  
  # Predictions - FIXED
  y_train_pred = model.predict(X_train).flatten()
  y_val_pred = model.predict(X_val).flatten()
  
  train_results = evaluate_model(y_train, y_train_pred, "ANN (Train)")
  val_results = evaluate_model(y_val, y_val_pred, "ANN (Validation)")
  
  train_results['CV_MSE_Mean'] = train_results['MSE']
  train_results['CV_MSE_Std'] = 0.0
  
  st.success("✅ **ANN Training Complete!**")
  
  return model, train_results, val_results, best_params

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

def train_sarima(train_data, val_data, target_col, tune_params=True, seasonal_period=12):
  """Train SARIMA model with optional parameter tuning"""
  train_data_clean = train_data.dropna()
  val_data_clean = val_data.dropna()
  
  train_series = train_data_clean.set_index('Date')[target_col]
  val_series = val_data_clean.set_index('Date')[target_col]
  
  if tune_params:
      best_order, best_seasonal_order = tune_sarima_parameters(train_series, seasonal_period)
  else:
      best_order = (2, 1, 2)
      best_seasonal_order = (1, 1, 1, seasonal_period)
  
  model = SARIMAX(train_series, order=best_order, seasonal_order=best_seasonal_order)
  results = model.fit(disp=False)
  
  train_pred = results.fittedvalues
  n_periods = len(val_series)
  val_pred = results.forecast(steps=n_periods)
  
  train_pred = train_pred[-len(train_series):]
  
  train_results = evaluate_model(train_series.values, train_pred.values, "SARIMA (Train)")
  val_results = evaluate_model(val_series.values, val_pred, "SARIMA (Validation)")
  
  best_params = {'order': best_order, 'seasonal_order': best_seasonal_order}
  
  return results, train_results, val_results, train_pred, val_pred, best_params

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

def tune_sarima_parameters(train_series, seasonal_period, p_range=(0, 2), d_range=(0, 2), q_range=(0, 2)):
  """Find optimal SARIMA parameters using grid search"""
  best_aic = np.inf
  best_order = (1, 1, 1)
  best_seasonal_order = (1, 1, 1, seasonal_period)
  
  for p in range(p_range[0], p_range[1] + 1):
      for d in range(d_range[0], d_range[1] + 1):
          for q in range(q_range[0], q_range[1] + 1):
              for P in range(0, 2):
                  for D in range(0, 2):
                      for Q in range(0, 2):
                          try:
                              model = SARIMAX(train_series, 
                                            order=(p, d, q), 
                                            seasonal_order=(P, D, Q, seasonal_period))
                              results = model.fit(disp=False)
                              
                              if results.aic < best_aic:
                                  best_aic = results.aic
                                  best_order = (p, d, q)
                                  best_seasonal_order = (P, D, Q, seasonal_period)
                          except:
                              continue
  
  return best_order, best_seasonal_order

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

def create_next_row_features_categorical(next_date, current_df, target_col, feature_cols, forecasts, encoders):
   """Create feature row for next prediction with categorical features - FIXED FOR NaN"""
   # Create basic date features
   next_row = pd.Series({
       'Date': next_date,
       'Year': next_date.year,
       'Month': next_date.month_name(),
       'DayOfWeek': next_date.day_name(),
       'Season': {12: 'Winter', 1: 'Winter', 2: 'Winter',
                 3: 'Spring', 4: 'Spring', 5: 'Spring',
                 6: 'Summer', 7: 'Summer', 8: 'Summer',
                 9: 'Autumn', 10: 'Autumn', 11: 'Autumn'}[next_date.month],
       'Weekend': 'Weekend' if next_date.weekday() >= 5 else 'Weekday',
       'Quarter': {1: 'Q1', 2: 'Q2', 3: 'Q3', 4: 'Q4'}[(next_date.month-1)//3 + 1],
       'MonthPart': 'Early' if next_date.day <= 10 else ('Mid' if next_date.day <= 20 else 'Late')
   })
   
   # Create dataframe for encoding
   next_df = pd.DataFrame([next_row])
   
   # Apply one-hot encoding
   next_df_encoded = apply_one_hot_encoding(next_df, encoders, get_categorical_features())
   
   # Add lag and rolling features
   for col in feature_cols:
       if '_lag_' in col:
           lag_num = int(col.split('_')[-1])
           if lag_num <= len(forecasts):
               next_df_encoded[col] = forecasts[-lag_num]
           else:
               idx = -(lag_num - len(forecasts))
               if idx >= -len(current_df):
                   next_df_encoded[col] = current_df[target_col].iloc[idx]
               else:
                   next_df_encoded[col] = current_df[target_col].median()  # Use median instead of 0
                   
       elif '_rolling_' in col or '_ewma_' in col:
           parts = col.split('_')
           if '_rolling_' in col:
               window = int(parts[-1])
               stat = parts[-2]
               
               # Get values for rolling calculation
               values = list(current_df[target_col].iloc[-(window-1):].values) + forecasts
               if len(values) >= window:
                   values = values[-window:]
               
               # Calculate statistics with NaN handling
               if stat == 'mean':
                   next_df_encoded[col] = np.nanmean(values) if len(values) > 0 else current_df[target_col].mean()
               elif stat == 'std':
                   next_df_encoded[col] = np.nanstd(values) if len(values) > 1 else 0
               elif stat == 'max':
                   next_df_encoded[col] = np.nanmax(values) if len(values) > 0 else current_df[target_col].max()
               elif stat == 'min':
                   next_df_encoded[col] = np.nanmin(values) if len(values) > 0 else current_df[target_col].min()
               else:
                   next_df_encoded[col] = current_df[target_col].median()
           else:
               # For EWMA, use last available value
               next_df_encoded[col] = current_df[target_col].iloc[-1]
               
       # Handle enhanced features (change, volatility, trend, percentile)
       elif '_daily_change' in col:
           if len(forecasts) > 0:
               next_df_encoded[col] = forecasts[-1] - current_df[target_col].iloc[-1]
           else:
               next_df_encoded[col] = 0
               
       elif '_weekly_change' in col:
           if len(current_df) >= 7:
               next_df_encoded[col] = forecasts[-1] if forecasts else current_df[target_col].iloc[-1] - current_df[target_col].iloc[-7]
           else:
               next_df_encoded[col] = 0
               
       elif '_volatility_' in col:
           window = int(col.split('_')[-1].replace('d', ''))
           if len(current_df) >= window:
               next_df_encoded[col] = current_df[target_col].iloc[-window:].std()
           else:
               next_df_encoded[col] = current_df[target_col].std()
               
       elif '_trend_' in col:
           window = int(col.split('_')[-1].replace('d', ''))
           if len(current_df) >= window:
               y_values = current_df[target_col].iloc[-window:].values
               x_values = np.arange(len(y_values))
               if len(y_values) > 1:
                   trend = np.polyfit(x_values, y_values, 1)[0]
                   next_df_encoded[col] = trend
               else:
                   next_df_encoded[col] = 0
           else:
               next_df_encoded[col] = 0
               
       elif '_percentile_' in col:
           window = int(col.split('_')[-1].replace('d', ''))
           if len(current_df) >= window:
               recent_values = current_df[target_col].iloc[-window:].values
               current_value = forecasts[-1] if forecasts else current_df[target_col].iloc[-1]
               percentile = stats.percentileofscore(recent_values, current_value)
               next_df_encoded[col] = percentile
           else:
               next_df_encoded[col] = 50.0  # Default to median percentile
               
       elif col in current_df.columns and col not in next_df_encoded.columns:
           next_df_encoded[col] = current_df[col].iloc[-1]
   
   # CRITICAL: Final NaN cleanup
   for col in feature_cols:
       if col in next_df_encoded.columns:
           if pd.isna(next_df_encoded[col].iloc[0]):
               # Use appropriate default based on feature type
               if 'std' in col or 'volatility' in col:
                   next_df_encoded[col] = 0
               elif 'percentile' in col:
                   next_df_encoded[col] = 50.0
               else:
                   # Use median of the feature from training data if available
                   next_df_encoded[col] = 0
   
   return next_df_encoded

def generate_ml_forecasts_categorical(model, test_data, feature_cols, target_col, days, scaler, encoders):
   """Generate forecasts for ML models with categorical features - FIXED"""
   current_df = test_data.copy()
   forecasts = []
   last_date = test_data['Date'].iloc[-1]
   
   for i in range(days):
       next_date = last_date + timedelta(days=i+1)
       next_df_encoded = create_next_row_features_categorical(
           next_date, current_df, target_col, feature_cols, forecasts, encoders
       )
       
       # Ensure all feature columns exist and have no NaN
       for col in feature_cols:
           if col not in next_df_encoded.columns:
               next_df_encoded[col] = 0
           # Extra NaN check
           if pd.isna(next_df_encoded[col].iloc[0]):
               next_df_encoded[col] = 0
       
       X_next = next_df_encoded[feature_cols].values.reshape(1, -1)
       
       # Final NaN check on the feature array
       if np.any(np.isnan(X_next)):
           # Replace any remaining NaN with 0
           X_next = np.nan_to_num(X_next, nan=0.0)
       
       if scaler:
           X_next = scaler.transform(X_next)
       
       pred = model.predict(X_next)[0]
       forecasts.append(pred)
       
       # Add prediction to current data for next iteration
       next_row_basic = pd.Series({
           'Date': next_date,
           target_col: pred,
           'Year': next_date.year,
           'Month': next_date.month_name(),
           'DayOfWeek': next_date.day_name(),
           'Season': {12: 'Winter', 1: 'Winter', 2: 'Winter',
                    3: 'Spring', 4: 'Spring', 5: 'Spring',
                    6: 'Summer', 7: 'Summer', 8: 'Summer',
                    9: 'Autumn', 10: 'Autumn', 11: 'Autumn'}[next_date.month],
           'Weekend': 'Weekend' if next_date.weekday() >= 5 else 'Weekday',
           'Quarter': {1: 'Q1', 2: 'Q2', 3: 'Q3', 4: 'Q4'}[(next_date.month-1)//3 + 1],
           'MonthPart': 'Early' if next_date.day <= 10 else ('Mid' if next_date.day <= 20 else 'Late')
       })
       
       current_df = pd.concat([current_df, pd.DataFrame([next_row_basic])], ignore_index=True)
   
   return forecasts

def forecast_future_enhanced_categorical(model, model_type, test_data, feature_cols, target_col, days=3, 
                                    scaler=None, encoders=None):
  """Enhanced forecasting with categorical features"""
  days = min(days, 3)
  
  last_date = pd.Timestamp(test_data['Date'].iloc[-1])
  future_dates = [last_date + timedelta(days=i+1) for i in range(days)]
  
  if model_type in ['rf', 'decision_tree'] and hasattr(model, 'estimators_' if model_type == 'rf' else 'tree_'):
      forecasts = []
      lower_bounds = []
      upper_bounds = []
      
      for i, next_date in enumerate(future_dates):
          next_df_encoded = create_next_row_features_categorical(
              next_date, test_data, target_col, feature_cols, forecasts, encoders
          )
          
          # Ensure all feature columns exist
          for col in feature_cols:
              if col not in next_df_encoded.columns:
                  next_df_encoded[col] = 0
          
          X_next = next_df_encoded[feature_cols].values.reshape(1, -1)
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
      
      future_df = pd.DataFrame({
          'Date': future_dates,
          target_col: forecasts,
          f'{target_col}_lower': lower_bounds,
          f'{target_col}_upper': upper_bounds
      })
  
  else:
      forecasts = generate_ml_forecasts_categorical(
          model, test_data, feature_cols, target_col, days, scaler, encoders
      )
      
      future_df = pd.DataFrame({
          'Date': future_dates,
          target_col: forecasts
      })
  
  return future_df

def forecast_future_enhanced(model, model_type, test_data, feature_cols, target_col, days=3, 
                       time_steps=7, scaler=None, target_scaler=None, best_params=None):
  """Enhanced forecasting for time series models"""
  days = min(days, 3)
  
  last_date = pd.Timestamp(test_data['Date'].iloc[-1])
  future_dates = [last_date + timedelta(days=i+1) for i in range(days)]
  future_df = pd.DataFrame({'Date': future_dates})
  
  if model_type in ['arima', 'sarima']:
      forecast_obj = model.get_forecast(steps=days)
      forecasts = forecast_obj.predicted_mean.values
      forecast_df = forecast_obj.summary_frame()
      future_df[target_col] = forecasts
      future_df[f'{target_col}_lower'] = forecast_df['mean_ci_lower'].values
      future_df[f'{target_col}_upper'] = forecast_df['mean_ci_upper'].values
  elif model_type == 'prophet':
      prophet_future = pd.DataFrame({'ds': future_dates})
      forecast = model.predict(prophet_future)
      forecasts = forecast['yhat'].values
      future_df[target_col] = forecasts
      future_df[f'{target_col}_lower'] = forecast['yhat_lower'].values
      future_df[f'{target_col}_upper'] = forecast['yhat_upper'].values
  
  return future_df

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

def plot_feature_importance_categorical(model, feature_names, model_name):
  """Plot feature importance with categorical feature grouping"""
  if hasattr(model, 'feature_importances_'):
      importances = model.feature_importances_
      
      importance_df = pd.DataFrame({
          'Feature': feature_names,
          'Importance': importances
      })
      
      # Group categorical features
      importance_df['Feature_Group'] = importance_df['Feature'].apply(
          lambda x: x.split('_')[0] if '_' in x and x.split('_')[0] in get_categorical_features()
          else 'Other'
      )
      
      # Show top individual features
      top_features = importance_df.sort_values('Importance', ascending=False).head(20)
      
      fig = px.bar(
          top_features,
          x='Importance',
          y='Feature',
          orientation='h',
          color='Feature_Group',
          title=f'Top 20 Feature Importances for {model_name}',
          color_discrete_sequence=px.colors.qualitative.Set3
      )
      
      fig.update_layout(height=700, width=800)
      
      # Group importance by category
      grouped_importance = importance_df.groupby('Feature_Group')['Importance'].sum().sort_values(ascending=False)
      
      fig_grouped = px.bar(
          x=grouped_importance.values,
          y=grouped_importance.index,
          orientation='h',
          title=f'Feature Group Importance for {model_name}',
          labels={'x': 'Total Importance', 'y': 'Feature Group'}
      )
      
      return fig, fig_grouped, importance_df
  
  return None, None, None

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

# Updated run_forecast_model_enhanced function with ALL fixes
def run_forecast_model_enhanced(model_type, train_data, val_data, test_data, target_col='PM10', 
                          feature_cols=None, forecast_days=3, tune_hyperparams=True,
                          handle_outliers_flag=True, outlier_method='cap',
                          n_folds=5, **kwargs):
  """Enhanced forecast model with ALL fixes and Optuna optimization"""
  results = {}
  
  forecast_days = min(forecast_days, 3)
  
  # ===== IMMEDIATE MISSING VALUE CHECK =====
  train_missing = check_missing_values(train_data, [model_type])
  val_missing = check_missing_values(val_data, [model_type])
  test_missing = check_missing_values(test_data, [model_type])
  
  all_missing = {**train_missing, **val_missing, **test_missing}
  
  can_proceed, compatibility_message = check_model_missing_value_compatibility(model_type, all_missing)
  
  if not can_proceed and all_missing:
      return {
          'error': 'missing_values_incompatible',
          'missing_info': { 'all': all_missing },
          'message': compatibility_message,
          'model_type': model_type
      }
  
  if all_missing:
      results['missing_values_info'] = {'detected': True, 'can_handle': can_proceed, 'message': compatibility_message, 'details': {'all': all_missing}}
  else:
      results['missing_values_info'] = {'detected': False, 'can_handle': True, 'message': "No missing values detected"}
  
  scaling_info = {
      'feature_scaling_applied': False,
      'feature_scaling_method': None,
      'target_scaling_applied': False,
      'target_scaling_method': None,
      'categorical_encoding': 'One-Hot Encoding'
  }
  
  if handle_outliers_flag:
      train_outliers = detect_outliers(train_data, target_col, method='all')
      val_outliers = detect_outliers(val_data, target_col, method='all')
      test_outliers = detect_outliers(test_data, target_col, method='all')
      
      train_data, train_outlier_info = handle_outliers_data(train_data, target_col, train_outliers, method=outlier_method)
      val_data, val_outlier_info = handle_outliers_data(val_data, target_col, val_outliers, method=outlier_method)
      test_data, test_outlier_info = handle_outliers_data(test_data, target_col, test_outliers, method=outlier_method)
      
      results['outliers'] = {
          'train': train_outlier_info, 'val': val_outlier_info, 'test': test_outlier_info,
          'total_detected': train_outlier_info['detected'] + val_outlier_info['detected'] + test_outlier_info['detected'],
          'total_handled': train_outlier_info['handled'] + val_outlier_info['handled'] + test_outlier_info['handled'],
          'total_remaining': train_outlier_info['remaining'] + val_outlier_info['remaining'] + test_outlier_info['remaining']
      }
  
  categorical_cols = get_categorical_features()
  base_features = [col for col in get_base_features() if col in train_data.columns]
  
  if model_type in ['arima', 'sarima', 'prophet']:
      if (train_data[target_col].isnull().any() or val_data[target_col].isnull().any() or test_data[target_col].isnull().any()):
          return {'error': 'missing_values_incompatible', 'missing_info': {'all': all_missing}, 'message': f"{model_type.upper()} requires no missing values in target column {target_col}", 'model_type': model_type}
          
      train_data, val_data, test_data = train_data.sort_values('Date'), val_data.sort_values('Date'), test_data.sort_values('Date')
      
      if model_type == 'arima':
          model, train_results, val_results, _, _, best_order = train_arima(train_data, val_data, target_col, tune_params=tune_hyperparams)
          best_params = {'order': best_order}
          test_series = test_data.set_index('Date')[target_col]
          y_test_pred = model.forecast(steps=len(test_series))
          y_test = test_series.values
          scaler_use = None
          test_data_use = test_data
      elif model_type == 'sarima':
          model, train_results, val_results, _, _, best_params = train_sarima(train_data, val_data, target_col, tune_params=tune_hyperparams, seasonal_period=kwargs.get('seasonal_period', 12))
          test_series = test_data.set_index('Date')[target_col]
          y_test_pred = model.forecast(steps=len(test_series))
          y_test = test_series.values
          scaler_use = None
          test_data_use = test_data
      elif model_type == 'prophet':
          model, train_results, val_results, _, _, best_params = train_prophet(train_data, val_data, tune_params=tune_hyperparams)
          prophet_test = test_data[['Date', target_col]].rename(columns={'Date': 'ds', target_col: 'y'})
          test_forecast = model.predict(prophet_test[['ds']])
          y_test_pred, y_test = test_forecast['yhat'].values, prophet_test['y'].values
          scaler_use = None
          test_data_use = test_data
  else:
      # Add lag features with ULTRA-ROBUST cleaning
      train_with_lags = add_lag_features(train_data, target_col, show_message=False)
      val_with_lags = add_lag_features(val_data, target_col, show_message=False)
      test_with_lags = add_lag_features(test_data, target_col, show_message=False)

      # Add enhanced domain features
      train_with_lags = add_enhanced_features(train_with_lags, target_col)
      val_with_lags = add_enhanced_features(val_with_lags, target_col)
      test_with_lags = add_enhanced_features(test_with_lags, target_col)

      lag_features = [col for col in train_with_lags.columns if any(x in col for x in ['_lag_', '_rolling_', '_ewma_', '_change', '_volatility', '_trend', '_percentile'])]
      
      # Create feature arrays with one-hot encoding
      X_train, y_train, feature_names, encoders = create_feature_target_arrays_with_encoding(train_with_lags, target_col, categorical_cols, base_features, lag_features)
      X_val, y_val, _, _ = create_feature_target_arrays_with_encoding(val_with_lags, target_col, categorical_cols, base_features, lag_features, encoders)
      X_test, y_test, _, _ = create_feature_target_arrays_with_encoding(test_with_lags, target_col, categorical_cols, base_features, lag_features, encoders)
      
      results['encoders'], results['feature_names'] = encoders, feature_names
      scaler_use = None # Default scaler to None
      
      # Training logic for different model types
      if model_type == 'decision_tree':
          model, train_results, val_results, best_params = train_decision_tree(X_train, y_train, X_val, y_val, tune_params=tune_hyperparams, n_folds=n_folds)
          y_test_pred = model.predict(X_test)
          test_data_use = test_with_lags
      elif model_type == 'rf':
          model, train_results, val_results, best_params = train_random_forest(X_train, y_train, X_val, y_val, tune_params=tune_hyperparams, n_folds=n_folds)
          y_test_pred = model.predict(X_test) 
          test_data_use = test_with_lags
      elif model_type == 'xgboost':
          model, train_results, val_results, best_params = train_xgboost(X_train, y_train, X_val, y_val, tune_params=tune_hyperparams, n_folds=n_folds)
          y_test_pred = model.predict(X_test)
          test_data_use = test_with_lags
      elif model_type == 'knn':
          model, train_results, val_results, best_params, scaler_use = train_knn(X_train, y_train, X_val, y_val, tune_params=tune_hyperparams, n_folds=n_folds)
          _, _, X_test_scaled, y_test, _ = prepare_data_for_distance_models(X_train, X_val, X_test, y_train)
          y_test_pred = model.predict(X_test_scaled)
          scaling_info['feature_scaling_applied'] = True
          scaling_info['feature_scaling_method'] = "RobustScaler (median=0, IQR=1)"
          test_data_use = test_with_lags
      elif model_type == 'svr':
          model, train_results, val_results, best_params, scaler_use = train_svr(X_train, y_train, X_val, y_val, tune_params=tune_hyperparams, n_folds=n_folds)
          _, _, X_test_scaled, y_test, _ = prepare_data_for_distance_models(X_train, X_val, X_test, y_train)
          y_test_pred = model.predict(X_test_scaled)
          scaling_info['feature_scaling_applied'] = True
          scaling_info['feature_scaling_method'] = "RobustScaler (median=0, IQR=1)"
          test_data_use = test_with_lags
      elif model_type == 'linear':
          model, train_results, val_results, best_params = train_linear_regression(X_train, y_train, X_val, y_val, tune_params=tune_hyperparams, n_folds=n_folds)
          y_test_pred = model.predict(X_test)
          test_data_use = test_with_lags
      elif model_type == 'ridge':
          model, train_results, val_results, best_params = train_ridge(X_train, y_train, X_val, y_val, tune_params=tune_hyperparams, n_folds=n_folds)
          y_test_pred = model.predict(X_test)
          test_data_use = test_with_lags
      elif model_type == 'lasso':
          model, train_results, val_results, best_params = train_lasso(X_train, y_train, X_val, y_val, tune_params=tune_hyperparams, n_folds=n_folds)
          y_test_pred = model.predict(X_test)
          test_data_use = test_with_lags
      elif model_type in ['lstm', 'gru', 'ann']:
           # Scaling for NN models
           X_train_scaled, X_val_scaled, X_test_scaled, feature_scaler, feature_scaling_method = scale_data(X_train, X_val, X_test, scaler_type='minmax')
           y_train_scaled, y_val_scaled, y_test_scaled, target_scaler, target_scaling_method = scale_target(y_train, y_val, y_test)
           scaling_info.update({'feature_scaling_applied': True, 'feature_scaling_method': feature_scaling_method, 'target_scaling_applied': True, 'target_scaling_method': target_scaling_method})
           use_optuna = kwargs.get('use_optuna', True)

           if model_type == 'lstm':
               time_steps = kwargs.get('time_steps', 7)
               model, train_results, val_results, best_params = train_lstm(X_train_scaled, y_train_scaled.ravel(), X_val_scaled, y_val_scaled.ravel(), tune_params=tune_hyperparams, n_folds=n_folds, time_steps=time_steps, use_optuna=use_optuna)
               X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled.ravel(), time_steps)
               y_test_pred = target_scaler.inverse_transform(model.predict(X_test_seq).reshape(-1, 1)).flatten() if len(X_test_seq) > 0 else np.array([])
               y_test = target_scaler.inverse_transform(y_test_seq.reshape(-1, 1)).flatten() if len(y_test_seq) > 0 else np.array([])
           elif model_type == 'gru':
               time_steps = kwargs.get('time_steps', 7)
               model, train_results, val_results, best_params = train_gru(X_train_scaled, y_train_scaled.ravel(), X_val_scaled, y_val_scaled.ravel(), tune_params=tune_hyperparams, n_folds=n_folds, time_steps=time_steps, use_optuna=use_optuna)
               X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled.ravel(), time_steps)
               y_test_pred = target_scaler.inverse_transform(model.predict(X_test_seq).reshape(-1, 1)).flatten() if len(X_test_seq) > 0 else np.array([])
               y_test = target_scaler.inverse_transform(y_test_seq.reshape(-1, 1)).flatten() if len(y_test_seq) > 0 else np.array([])
           elif model_type == 'ann':
               model, train_results, val_results, best_params = train_ann(X_train_scaled, y_train_scaled.ravel(), X_val_scaled, y_val_scaled.ravel(), tune_params=tune_hyperparams, n_folds=n_folds, use_optuna=use_optuna)
               y_test_pred = target_scaler.inverse_transform(model.predict(X_test_scaled).reshape(-1, 1)).flatten()
               y_test = target_scaler.inverse_transform(y_test_scaled.reshape(-1, 1)).flatten()
           
           scaler_use, test_data_use = feature_scaler, test_with_lags
      else:
          raise ValueError(f"Model type '{model_type}' not implemented")

  test_results = evaluate_model(y_test, y_test_pred, f"{model_type.upper()} (Test)")
  test_results['Overall_Score'] = calculate_overall_score(test_results)
  
  if model_type not in ['arima', 'sarima', 'prophet']:
      forecast_df = forecast_future_enhanced_categorical(model, model_type, test_data_use, feature_names, target_col, days=forecast_days, scaler=scaler_use, encoders=encoders)
  else:
      forecast_df = forecast_future_enhanced(model, model_type, test_data, feature_cols, target_col, days=forecast_days)
  
  results.update({'model': model, 'test_pred': y_test_pred, 'train_results': train_results, 'val_results': val_results, 'test_results': test_results, 'best_params': best_params, 'forecast': forecast_df, 'model_type': model_type, 'scaling_info': scaling_info})
  
  if model_type in ['rf', 'xgboost', 'decision_tree']:
      importance_fig, importance_fig_grouped, importance_df = plot_feature_importance_categorical(model, feature_names, model_type.upper())
      results['importance_fig'], results['importance_fig_grouped'], results['importance_df'] = importance_fig, importance_fig_grouped, importance_df
  
  return results

# Main Streamlit app
def main():
  # Sidebar for navigation
  st.sidebar.title("Navigation")
  page = st.sidebar.radio("Go to", ["Data Explorer", "Outlier Analysis", "Model Training", 
                                    "Model Comparison", "Forecasting Dashboard"])

  # Initialize session state
  if 'data' not in st.session_state:
      st.session_state.data = None
      st.session_state.daily_data = None
      st.session_state.train_data = None
      st.session_state.val_data = None
      st.session_state.test_data = None
      st.session_state.model_results = {}
      st.session_state.train_ratio = 0.65
      st.session_state.val_ratio = 0.15
      st.session_state.test_ratio = 0.20
      st.session_state.training_confirmed = False
  
  # Data Upload
  st.sidebar.title("Data Upload")
  uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
  
  if uploaded_file is not None:
      with st.spinner("Loading data..."):
          df = load_data(uploaded_file)
          if 'AQI' in df.columns:
              df = df.drop(columns = ["AQI"])
          if 'PM2.5' in df.columns:
              df = df.drop(columns = ["PM2.5"])
          st.session_state.data = df
          
          st.session_state.daily_data = create_daily_dataset(df)
          
          st.session_state.train_data, st.session_state.val_data, st.session_state.test_data = split_data(
              st.session_state.daily_data, 'PM10',
              train_size=st.session_state.train_ratio,
              val_size=st.session_state.val_ratio,
              test_size=st.session_state.test_ratio
          )
          
          # Check for missing values after loading
          missing_info = check_missing_values(st.session_state.daily_data)
          if missing_info:
              st.sidebar.warning(f"⚠️ Missing values detected in {len(missing_info)} columns")
          else:
              st.sidebar.success(f"✅ Loaded data with {len(df)} rows and {df.shape[1]} columns")
  
  # Data Explorer Page
  if page == "Data Explorer":
      st.title("📊 Data Explorer")
      
      if st.session_state.data is not None:
          # Missing values summary at top
          st.header("📋 Data Quality Overview")
          
          missing_info = check_missing_values(st.session_state.daily_data)
          if missing_info:
              st.warning("⚠️ **Missing values detected in your dataset!**")
              
              # Show model compatibility matrix
              st.subheader("🤖 Model Compatibility with Missing Values")
              
              models_handling_missing = get_models_handling_missing_data()
              models_requiring_complete = get_models_requiring_complete_data()
              
              col1, col2 = st.columns(2)
              
              with col1:
                  st.success("**✅ Models that CAN handle missing values:**")
                  for model in models_handling_missing:
                      st.write(f"- **{model.upper()}** (Tree-based)")
                  st.info("These models can be trained without data cleaning!")
              
              with col2:
                  st.error("**❌ Models that REQUIRE complete data:**")
                  for model in models_requiring_complete:
                      st.write(f"- **{model.upper()}**")
                  st.warning("These models need data cleaning first!")
              
              # Show missing value summary
              missing_summary = []
              for col, info in missing_info.items():
                  missing_summary.append({
                      'Column': col,
                      'Missing Count': info['count'],
                      'Missing %': f"{info['percentage']:.2f}%"
                  })
              
              missing_df = pd.DataFrame(missing_summary)
              st.dataframe(missing_df, use_container_width=True)
              
              with st.expander("💡 Data Cleaning Recommendations"):
                  st.write("**Option 1: Use tree-based models** (XGBoost, Random Forest, Decision Tree)")
                  st.write("- These models handle missing values automatically")
                  st.write("- No data preprocessing required")
                  st.write("- Often perform well even with missing data")
                  
                  st.write("**Option 2: Clean data for all models**")
                  st.write("**Remove rows with missing values:**")
                  st.code("df = df.dropna()")
                  
                  st.write("**Fill missing values with median:**")
                  st.code("df[column_name] = df[column_name].fillna(df[column_name].median())")
                  
                  st.write("**Forward fill for time series data:**")
                  st.code("df[column_name] = df[column_name].fillna(method='ffill')")
                  
                  st.write("**Interpolation:**")
                  st.code("df[column_name] = df[column_name].interpolate()")
          else:
              st.success("✅ **No missing values detected** - All models can be trained!")

          st.header("Data Overview")
          st.subheader("Daily Aggregated Data")
          st.dataframe(st.session_state.daily_data.head(10))

          # ===== DATA SPLIT CONFIGURATION =====
          st.header("📋 Data Split Configuration")
          st.write("Configure your train/validation/test split ratios. Splits are applied chronologically (time-series order).")
          
          # Initialize session state for temporary slider values if not exists
          if 'temp_train_ratio' not in st.session_state:
              st.session_state.temp_train_ratio = st.session_state.train_ratio
          if 'temp_val_ratio' not in st.session_state:
              st.session_state.temp_val_ratio = st.session_state.val_ratio
          if 'temp_test_ratio' not in st.session_state:
              st.session_state.temp_test_ratio = st.session_state.test_ratio
          
          # Current Configuration Display
          st.subheader("Current Split Configuration")
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
          
          # Preset configurations section
          st.subheader("Quick Preset Configurations")
          preset_col1, preset_col2, preset_col3, preset_col4 = st.columns(4)
          
          with preset_col1:
              if st.button("Standard ML\n(70/15/15)", key="explorer_preset_standard"):
                  st.session_state.temp_train_ratio = 0.70
                  st.session_state.temp_val_ratio = 0.15
                  st.session_state.temp_test_ratio = 0.15
                  st.rerun()
          
          with preset_col2:
              if st.button("Conservative\n(60/20/20)", key="explorer_preset_conservative"):
                  st.session_state.temp_train_ratio = 0.60
                  st.session_state.temp_val_ratio = 0.20
                  st.session_state.temp_test_ratio = 0.20
                  st.rerun()
          
          with preset_col3:
              if st.button("Training Heavy\n(80/10/10)", key="explorer_preset_training"):
                  st.session_state.temp_train_ratio = 0.80
                  st.session_state.temp_val_ratio = 0.10
                  st.session_state.temp_test_ratio = 0.10
                  st.rerun()
          
          with preset_col4:
              if st.button("Balanced\n(65/15/20)", key="explorer_preset_balanced"):
                  st.session_state.temp_train_ratio = 0.65
                  st.session_state.temp_val_ratio = 0.15
                  st.session_state.temp_test_ratio = 0.20
                  st.rerun()
          
          # Manual adjustment sliders
          with st.expander("🔧 Manual Split Adjustment", expanded=False):
              col1, col2, col3 = st.columns(3)
              
              with col1:
                  new_train_ratio = st.slider(
                      "Training Set Ratio",
                      min_value=0.4,
                      max_value=0.8,
                      value=st.session_state.temp_train_ratio,
                      step=0.05,
                      help="Percentage of data used for training (chronologically first)",
                      key="explorer_train_slider"
                  )
                  st.session_state.temp_train_ratio = new_train_ratio
              
              with col2:
                  new_val_ratio = st.slider(
                      "Validation Set Ratio",
                      min_value=0.1,
                      max_value=0.3,
                      value=st.session_state.temp_val_ratio,
                      step=0.05,
                      help="Percentage of data used for validation (chronologically middle)",
                      key="explorer_val_slider"
                  )
                  st.session_state.temp_val_ratio = new_val_ratio
              
              with col3:
                  new_test_ratio = st.slider(
                      "Test Set Ratio",
                      min_value=0.1,
                      max_value=0.3,
                      value=st.session_state.temp_test_ratio,
                      step=0.05,
                      help="Percentage of data used for testing (chronologically last)",
                      key="explorer_test_slider"
                  )
                  st.session_state.temp_test_ratio = new_test_ratio
              
              total_ratio = new_train_ratio + new_val_ratio + new_test_ratio
              
              st.write(f"**Total Ratio:** {total_ratio:.2f}")
              
              if abs(total_ratio - 1.0) > 0.01:
                  st.warning(f"⚠️ Ratios should sum to 1.0. Current sum: {total_ratio:.2f}")
                  st.info("Ratios will be automatically normalized when applied.")
              else:
                  st.success("✅ Ratios sum correctly to 1.0")
          
          # Apply changes section
          config_changed = (new_train_ratio != st.session_state.train_ratio or 
                           new_val_ratio != st.session_state.val_ratio or 
                           new_test_ratio != st.session_state.test_ratio)
          
          if config_changed:
              st.info("⚠️ **Split configuration changed.** Click 'Apply New Configuration' to update the data split.")
              
              # Preview section
              with st.expander("👁️ Preview New Split", expanded=True):
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
              
              # Apply button
              if st.button("🔄 Apply New Split Configuration", type="primary", key="explorer_apply_split"):
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
                      # Reset training confirmation when data split changes
                      st.session_state.training_confirmed = False
                  
                  st.success("✅ Data split configuration updated successfully!")
                  st.info("ℹ️ Previous model results have been cleared. Please retrain models with the new data split.")
                  st.rerun()
          
          # Quick recommendations based on dataset size
          total_days = len(st.session_state.daily_data)
          if total_days < 365:
              st.info("💡 **Recommendation for small dataset (<1 year):** Consider using 'Training Heavy' preset to maximize training data.")
          elif total_days < 730:
              st.info("💡 **Recommendation for medium dataset (1-2 years):** 'Standard ML' or 'Balanced' presets work well.")
          else:
              st.info("💡 **Recommendation for large dataset (2+ years):** Any preset should work well. 'Conservative' provides robust evaluation.")
          
          # Categorical Features Section
          st.header("🏷️ Categorical Date Features")
          categorical_features = get_categorical_features()
          sample_data = st.session_state.daily_data[categorical_features].head(10)
          st.dataframe(sample_data)
          
          st.info("**Note:** These categorical features will be one-hot encoded for machine learning models, " +
                 "providing better representation of temporal patterns compared to numerical encoding.")
          
          # Show unique values for each categorical feature
          with st.expander("View Unique Values in Categorical Features"):
              for feature in categorical_features:
                  if feature in st.session_state.daily_data.columns:
                      unique_vals = st.session_state.daily_data[feature].unique()
                      st.write(f"**{feature}:** {', '.join(map(str, unique_vals))}")
          
          st.header("📈 Data Visualization")
          
          st.subheader("Time Series Plot")
          ts_fig = plot_time_series(st.session_state.daily_data, 'PM10')
          st.plotly_chart(ts_fig, use_container_width=True)
          
          st.subheader("Correlation Analysis")
          corr_fig, corr = plot_correlation_matrix(st.session_state.daily_data, 'PM10')
          st.plotly_chart(corr_fig, use_container_width=True)
          
          st.subheader("Statistical Summary")
          st.dataframe(st.session_state.daily_data.describe())
          
          # Data split visualization
          st.subheader("📊 Current Data Split Visualization")
          
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
  
  # Outlier Analysis Page
  elif page == "Outlier Analysis":
      st.title("🔍 Outlier Analysis")
      
      if st.session_state.data is not None:
          # Missing values check
          missing_info = check_missing_values(st.session_state.daily_data)
          if missing_info:
              st.warning("⚠️ **Missing values detected!** Some models can still proceed, others cannot.")
              
              models_handling_missing = get_models_handling_missing_data()
              models_requiring_complete = get_models_requiring_complete_data()
              
              col1, col2 = st.columns(2)
              with col1:
                  st.success(f"**✅ Can proceed:** {', '.join([m.upper() for m in models_handling_missing])}")
              with col2:
                  st.error(f"**❌ Need data cleaning:** {', '.join([m.upper() for m in models_requiring_complete])}")
              
              display_missing_value_error(missing_info)
              return
          
          st.header("Outlier Detection and Handling")
          
          col1, col2 = st.columns(2)
          with col1:
              detection_method = st.selectbox("Detection Method", ["iqr"])
          with col2:
              handling_method = st.selectbox("Handling Method", ["cap"])
          
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
              cleaned_data, outlier_info = handle_outliers_data(
                  st.session_state.daily_data, 'PM10', outliers, method=handling_method
              )
              
              # Display detailed outlier handling results
              st.subheader("📊 Outlier Handling Results")
              
              col1, col2, col3, col4 = st.columns(4)
              with col1:
                  st.metric("Detected", outlier_info['detected'])
              with col2:
                  st.metric("Effectively Handled", outlier_info['handled'])
              with col3:
                  st.metric("Still Remaining", outlier_info['remaining'])
              with col4:
                  st.metric("Effectiveness", outlier_info['effectiveness'])
              
              # Show before/after comparison
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
                  title=f'PM10 Before and After Outlier Handling ({handling_method.title()})',
                  xaxis_title='Date',
                  yaxis_title='PM10 (µg/m³)',
                  height=500
              )
              st.plotly_chart(fig, use_container_width=True)
      
      else:
          st.info("Please upload a dataset using the sidebar.")
  
  # Model Training Page
  elif page == "Model Training":
      st.title("🤖 Model Training with Categorical Features & Optuna Optimization")
      
      if st.session_state.data is not None:
          # Missing values check
          missing_info = check_missing_values(st.session_state.daily_data)
          if missing_info:
              st.warning("🚨 **Missing Values Detected in Dataset**")
              
              models_handling_missing = get_models_handling_missing_data()
              models_requiring_complete = get_models_requiring_complete_data()
              
              col1, col2 = st.columns(2)
              with col1:
                  st.success("**✅ Available Models (handle missing values):**")
                  for model in models_handling_missing:
                      st.write(f"- {model.upper()}")
              
              with col2:
                  st.error("**❌ Unavailable Models (require complete data):**")
                  for model in models_requiring_complete:
                      st.write(f"- {model.upper()}")
              
              st.info("💡 **You can still train tree-based models, or clean your data to access all models.**")
          else:
              st.success("✅ **No missing values detected** - All models are available!")
          
          # Feature engineering info
          with st.expander("🏗️ Feature Engineering & Optuna Details", expanded=False):
              st.write("**Categorical Date Features (One-Hot Encoded):**")
              categorical_features = get_categorical_features()
              
              col1, col2 = st.columns(2)
              with col1:
                  for i, feature in enumerate(categorical_features[:3]):
                      unique_vals = st.session_state.daily_data[feature].unique()
                      st.write(f"**{feature}:** {len(unique_vals)} categories")
                      st.write(f"  → Creates {len(unique_vals)-1} binary features")
              
              with col2:
                  for i, feature in enumerate(categorical_features[3:]):
                      unique_vals = st.session_state.daily_data[feature].unique()
                      st.write(f"**{feature}:** {len(unique_vals)} categories")
                      st.write(f"  → Creates {len(unique_vals)-1} binary features")
              
              st.write("**Other Features:**")
              st.write("- **Base Features:** Year, Temperature, Wind Speed, Wind Direction")
              st.write("- **Lag Features:** PM10 values from 1, 2, 3, 7, 14 days ago")
              st.write("- **Rolling Features:** 7, 14, 30-day rolling statistics")
              
              st.write("**🔧 Optuna Optimization for Neural Networks:**")
              st.write("- **LSTM/GRU:** Optimizes units, dropout, batch size, learning rate")
              st.write("- **ANN:** Optimizes hidden layers, neurons, dropout, learning rate")
              st.write("- **Trials:** 10-20 optimization trials for best hyperparameters")
              st.write("- **Early Stopping:** Prevents overfitting during training")
          
          st.sidebar.title("Model Settings")
          model_type = st.sidebar.selectbox(
              "Select Model", 
              ["decision_tree", "knn", "rf", "xgboost", "svr", "lstm", "ann", "gru", 
               "arima", "sarima", "prophet", "linear", "ridge", "lasso"] 
          )
          
          # Show model-specific information
          missing_info = check_missing_values(st.session_state.daily_data)
          can_proceed, compatibility_message = check_model_missing_value_compatibility(model_type, missing_info)
          
          if model_type in ['decision_tree', 'rf', 'xgboost']:
              st.sidebar.success("🌳 **Tree-based model**: Excellent for handling categorical features and missing values!")
              if missing_info:
                  st.sidebar.info(f"✅ {compatibility_message}")
          elif model_type in ['knn', 'svr']:
              st.sidebar.info("📏 **Distance-based model**: Benefits from one-hot encoding and feature scaling!")
              if missing_info:
                  st.sidebar.warning(f"⚠️ {compatibility_message}")
          elif model_type in ['lstm', 'ann', 'gru']:
              st.sidebar.info("🧠 **Neural Network**: Requires feature scaling and uses Optuna optimization!")
              if missing_info:
                  st.sidebar.error(f"❌ {compatibility_message}")
          elif model_type in ['arima', 'sarima', 'prophet']:
              st.sidebar.info("📈 **Time series model**: Uses temporal patterns for forecasting!")
              if missing_info:
                  st.sidebar.error(f"❌ {compatibility_message}")
          else:
              if missing_info:
                  if can_proceed:
                      st.sidebar.success(f"✅ {compatibility_message}")
                  else:
                      st.sidebar.error(f"❌ {compatibility_message}")
          
          with st.sidebar.expander("Advanced Settings"):
              tune_hyperparams = st.checkbox("Tune Hyperparameters", value=True)
              handle_outliers_flag = st.checkbox("Handle Outliers", value=True)
              if handle_outliers_flag:
                  outlier_method = st.selectbox("Outlier Handling Method", ["cap"])
              else:
                  outlier_method = None
              
              n_folds = st.slider("K-Fold Cross-Validation Folds", 3, 10, 5)
              forecast_days = st.slider("Forecast Days", 1, 3, 3)
              
              # Special settings for neural networks
              if model_type in get_neural_network_models():
                  st.write("**🔧 Neural Network & Optuna Settings:**")
                  use_optuna = st.checkbox("Use Optuna Optimization", value=True, 
                                         help="Use Optuna for hyperparameter optimization (recommended)")
                  optuna_trials = st.slider("Optuna Trials", 5, 50, 10, 
                                          help="Number of optimization trials (more = better but slower)")
                  
                  if model_type in ['lstm', 'gru']:
                      time_steps = st.slider("Time Steps (Sequence Length)", 3, 14, 7)
                      st.info(f"Using {time_steps} previous days to predict next day")
                  
                  if use_optuna:
                      st.success(f"✅ Optuna will optimize {optuna_trials} trials")
                  else:
                      st.info("ℹ️ Using default hyperparameters (faster but may be suboptimal)")
              
              # Special settings for time series models
              if model_type in ['arima', 'sarima']:
                  st.write("**Time Series Settings:**")
                  max_order = st.slider("Maximum Order (p,d,q)", 1, 3, 2)
                  if model_type == 'sarima':
                      seasonal_period = st.selectbox("Seasonal Period", [7, 12, 24], index=1, 
                                                   help="7=Weekly, 12=Monthly, 24=Daily cycles")
          
          # Training button and logic
          if st.sidebar.button("🚀 Train Model"):
              # Check if model can proceed with current data state
              missing_info = check_missing_values(st.session_state.daily_data)
              can_proceed, compatibility_message = check_model_missing_value_compatibility(model_type, missing_info)
              
              if not can_proceed and missing_info:
                  st.error(f"❌ **Cannot train {model_type.upper()} model**")
                  st.write(compatibility_message)
                  display_missing_value_error(missing_info, [model_type])
                  st.stop()
              
              # Show warning for models that can handle missing values
              if can_proceed and missing_info:
                  display_missing_value_warning(missing_info, model_type)
              
              # Show additional info for neural networks
              if model_type in get_neural_network_models():
                  if use_optuna:
                      st.info(f"🔧 **Training {model_type.upper()} with Optuna Optimization** - This will take longer but provide better results...")
                  else:
                      st.info(f"🧠 **Training {model_type.upper()} Neural Network** - Using default hyperparameters...")
              
              # Proceed with training
              with st.spinner(f"Training {model_type.upper()} model with {n_folds}-fold CV..."):
                  try:
                      kwargs = {}
                      if model_type in ['lstm', 'gru']:
                          kwargs['time_steps'] = time_steps
                          kwargs['use_optuna'] = use_optuna
                      elif model_type == 'ann':
                          kwargs['use_optuna'] = use_optuna
                      elif model_type == 'sarima':
                          kwargs['seasonal_period'] = seasonal_period
                      
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
                          **kwargs
                      )
                      
                      # Handle missing value errors
                      if 'error' in results and results['error'] == 'missing_values_incompatible':
                          st.error(f"❌ **{model_type.upper()} Model Training Failed**")
                          st.write(results['message'])
                          missing_info = results['missing_info']['all']
                          display_missing_value_error(missing_info, [model_type])
                          st.stop()
                      
                      # Handle feature engineering errors
                      if 'error' in results and results['error'] == 'feature_engineering_failed':
                          st.error(f"❌ **{model_type.upper()} Model Training Failed During Feature Engineering**")
                          st.write(results['message'])
                          st.stop()
                      
                      # Normal success path
                      if results and 'error' not in results:
                          results['data_split'] = {
                              'train_ratio': st.session_state.train_ratio,
                              'val_ratio': st.session_state.val_ratio,
                              'test_ratio': st.session_state.test_ratio,
                              'train_size': len(st.session_state.train_data),
                              'val_size': len(st.session_state.val_data),
                              'test_size': len(st.session_state.test_data)
                          }
                          
                          st.session_state.model_results[model_type] = results
                          
                          # Success message
                          if model_type in get_neural_network_models() and use_optuna:
                              st.success(f"✅ {model_type.upper()} model trained successfully with Optuna optimization!")
                              st.info(f"🎯 **Best Parameters Found:** {results['best_params']}")
                          elif results.get('missing_values_info', {}).get('detected', False):
                              st.success(f"✅ {model_type.upper()} model trained successfully! Model handled missing values automatically.")
                          else:
                              st.success(f"✅ {model_type.upper()} model trained successfully!")
                      else:
                          st.error(f"❌ **{model_type.upper()} Model Training Failed**")
                          if 'message' in results:
                              st.write(results['message'])
                              
                  except Exception as e:
                      error_msg = str(e)
                      st.error(f"❌ **Unexpected Error in {model_type.upper()} Training:**")
                      st.code(str(e))
                      st.write("**Suggestions:**")
                      st.write("- Check data quality and format")
                      st.write("- Try a different model type")
                      st.write("- Reduce dataset size if memory issues")
                      
                      # Provide alternative recommendations
                      st.info("💡 **Alternative Models to Try:**")
                      if missing_info:
                          models_that_work = get_models_handling_missing_data()
                          st.write(f"**Recommended:** {', '.join([m.upper() for m in models_that_work])}")
                      else:
                          st.write("**All models should work with your clean dataset. Try a different model type.**")
          
          # Show model results if available
          if model_type in st.session_state.model_results:
              results = st.session_state.model_results[model_type]
              
              st.header("🎯 Model Performance")
              
              # Enhanced: Display scaling and encoding information
              if 'scaling_info' in results:
                  st.subheader("📊 Feature Engineering & Data Handling Information")
                  scaling_info = results['scaling_info']
                  
                  col1, col2, col3, col4 = st.columns(4)
                  
                  with col1:
                      st.success("✅ **Categorical Encoding Applied**")
                      st.write("**Method:** One-Hot Encoding")
                      st.write("**Features:** Month, DayOfWeek, Season, Weekend, Quarter, MonthPart")
                  
                  with col2:
                      if scaling_info['feature_scaling_applied']:
                          st.success("✅ **Feature Scaling Applied**")
                          st.write(f"**Method:** {scaling_info['feature_scaling_method']}")
                      else:
                          st.info("ℹ️ **No Feature Scaling**")
                          st.write("Raw features used")
                  
                  with col3:
                      if scaling_info['target_scaling_applied']:
                          st.success("✅ **Target Scaling Applied**")
                          st.write(f"**Method:** {scaling_info['target_scaling_method']}")
                      else:
                          st.info("ℹ️ **No Target Scaling**")
                          st.write("Raw target values used")
                  
                  with col4:
                      # Show missing value handling info
                      if 'missing_values_info' in results:
                          mv_info = results['missing_values_info']
                          if mv_info['detected']:
                              if mv_info['can_handle']:
                                  st.success("✅ **Missing Values Handled**")
                                  st.write("**Method:** Native algorithm handling")
                              else:
                                  st.error("❌ **Missing Values Issue**")
                          else:
                              st.success("✅ **No Missing Values**")
                              st.write("**Status:** Complete dataset")
              
              # Show Optuna optimization results for neural networks
              if model_type in get_neural_network_models() and 'best_params' in results:
                  st.subheader("🔧 Optuna Optimization Results")
                  st.write("**Best Hyperparameters Found:**")
                  
                  best_params = results['best_params']
                  param_cols = st.columns(len(best_params))
                  
                  for i, (param, value) in enumerate(best_params.items()):
                      with param_cols[i]:
                          st.metric(param.replace('_', ' ').title(), str(value))
                  
                  st.info("🎯 These parameters were selected after evaluating multiple combinations to minimize validation loss.")
              
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
              
              # Enhanced feature importance with categorical grouping
              if 'importance_fig' in results and results['importance_fig'] is not None:
                  st.header("🎯 Feature Importance Analysis")
                  
                  tab1, tab2 = st.tabs(["Individual Features", "Feature Groups"])
                  
                  with tab1:
                      st.plotly_chart(results['importance_fig'], use_container_width=True)
                  
                  with tab2:
                      if 'importance_fig_grouped' in results:
                          st.plotly_chart(results['importance_fig_grouped'], use_container_width=True)
                      
                      # Show insights about categorical features
                      st.subheader("📈 Categorical Feature Insights")
                      if 'importance_df' in results and results['importance_df'] is not None:
                          imp_df = results['importance_df']
                          
                          # Find most important day of week
                          dow_features = imp_df[imp_df['Feature'].str.startswith('DayOfWeek_')]
                          if not dow_features.empty:
                              best_dow = dow_features.loc[dow_features['Importance'].idxmax(), 'Feature']
                              st.write(f"**Most Important Day:** {best_dow.replace('DayOfWeek_', '')}")
                          
                          # Find most important month
                          month_features = imp_df[imp_df['Feature'].str.startswith('Month_')]
                          if not month_features.empty:
                              best_month = month_features.loc[month_features['Importance'].idxmax(), 'Feature']
                              st.write(f"**Most Important Month:** {best_month.replace('Month_', '')}")
                          
                          # Find most important season
                          season_features = imp_df[imp_df['Feature'].str.startswith('Season_')]
                          if not season_features.empty:
                              best_season = season_features.loc[season_features['Importance'].idxmax(), 'Feature']
                              st.write(f"**Most Important Season:** {best_season.replace('Season_', '')}")
              
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
  
  # Model Comparison Page
  elif page == "Model Comparison":
      st.title("📈 Model Comparison")
      
      if st.session_state.data is not None and len(st.session_state.model_results) > 0:
          # Check for missing values
          missing_info = check_missing_values(st.session_state.daily_data)
          if missing_info:
              st.warning("⚠️ **Missing values detected in current dataset.** Trained models may have used different data.")
          
          st.header("Performance Summary")
          summary_df = create_performance_summary_table(st.session_state.model_results)
          st.dataframe(summary_df)
          
          if not summary_df.empty:
              best_model = summary_df.iloc[0]['Model']
              best_score = summary_df.iloc[0]['Overall Score']
              st.success(f"Best Model: {best_model} with Overall Score: {best_score}")
          
          # Error comparison
          st.header("Error Metrics Comparison")
          metrics = ["RMSE", "MAE", "MAPE", "R²"]
          metric = st.selectbox("Select Metric", metrics)
          
          test_metrics = []
          for model_type, results in st.session_state.model_results.items():
              if 'test_results' in results:
                  test_metrics.append({
                      'Model': model_type.upper(),
                      **results['test_results']
                  })
          
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
          if st.session_state.data is None:
              st.info("Please upload a dataset using the sidebar.")
          else:
              st.info("Please train at least one model in the 'Model Training' tab.")
  
  # Forecasting Dashboard Page
  elif page == "Forecasting Dashboard":
      st.title("📊 Forecasting Dashboard")
      
      if st.session_state.data is not None and len(st.session_state.model_results) > 0:
          # Check for missing values
          missing_info = check_missing_values(st.session_state.daily_data)
          if missing_info:
              st.warning("⚠️ **Missing values detected in current dataset.** Dashboard results may not reflect current data state.")
          
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
              
              forecast_start_date = pd.Timestamp(historical['Date'].iloc[-1])
              
              fig.add_vline(
                  x=forecast_start_date.timestamp() * 1000,
                  line_dash="dash",
                  line_color="gray",
                  annotation_text="Forecast Start"
              )
              
              fig.update_layout(
                  title="PM10 Forecast Comparison (1-3 Days) with Optuna Optimization",
                  xaxis_title="Date",
                  yaxis_title="PM10 (µg/m³)",
                  height=600,
                  hovermode='x unified'
              )
              
              st.plotly_chart(fig, use_container_width=True)
          
          # Enhanced Categorical Feature Insights
          st.header("🏷️ Categorical Feature Insights")
          
          if viz_models:
              # Collect categorical insights from tree-based models
              categorical_insights = {}
              for model_name in viz_models:
                  if model_name in st.session_state.model_results:
                      results = st.session_state.model_results[model_name]
                      if 'importance_df' in results and results['importance_df'] is not None:
                          imp_df = results['importance_df']
                          
                          # Extract categorical insights
                          insights = {}
                          
                          # Day of week insights
                          dow_features = imp_df[imp_df['Feature'].str.startswith('DayOfWeek_')]
                          if not dow_features.empty:
                              best_dow = dow_features.loc[dow_features['Importance'].idxmax(), 'Feature']
                              insights['best_day'] = best_dow.replace('DayOfWeek_', '')
                          
                          # Month insights
                          month_features = imp_df[imp_df['Feature'].str.startswith('Month_')]
                          if not month_features.empty:
                              best_month = month_features.loc[month_features['Importance'].idxmax(), 'Feature']
                              insights['best_month'] = best_month.replace('Month_', '')
                          
                          # Season insights
                          season_features = imp_df[imp_df['Feature'].str.startswith('Season_')]
                          if not season_features.empty:
                              best_season = season_features.loc[season_features['Importance'].idxmax(), 'Feature']
                              insights['best_season'] = best_season.replace('Season_', '')
                          
                          # Weekend insights
                          weekend_features = imp_df[imp_df['Feature'].str.startswith('Weekend_')]
                          if not weekend_features.empty:
                              best_weekend = weekend_features.loc[weekend_features['Importance'].idxmax(), 'Feature']
                              insights['weekend_pattern'] = best_weekend.replace('Weekend_', '')
                          
                          categorical_insights[model_name] = insights
              
              if categorical_insights:
                  st.subheader("📈 Key Temporal Patterns Discovered")
                  
                  for model_name, insights in categorical_insights.items():
                      st.write(f"**{model_name.upper()} Model Findings:**")
                      
                      col1, col2, col3, col4 = st.columns(4)
                      
                      with col1:
                          if 'best_day' in insights:
                              st.metric("Most Predictive Day", insights['best_day'])
                      
                      with col2:
                          if 'best_month' in insights:
                              st.metric("Most Predictive Month", insights['best_month'])
                      
                      with col3:
                          if 'best_season' in insights:
                              st.metric("Most Predictive Season", insights['best_season'])
                      
                      with col4:
                          if 'weekend_pattern' in insights:
                              pattern = "Higher Impact" if insights['weekend_pattern'] == "Weekend" else "Weekday Pattern"
                              st.metric("Weekend vs Weekday", pattern)
                      
                      st.write("")
          
          # Model Optimization Summary
          st.header("🔧 Model Optimization Summary")
          
          optimization_summary = []
          for model_name, results in st.session_state.model_results.items():
              if model_name in get_neural_network_models() and 'best_params' in results:
                  optimization_summary.append({
                      'Model': model_name.upper(),
                      'Optimization': 'Optuna',
                      'Best Score': f"{results['test_results']['Overall_Score']:.1f}",
                      'Key Parameters': ', '.join([f"{k}={v}" for k, v in list(results['best_params'].items())[:2]])
                  })
              elif 'best_params' in results and results['best_params']:
                  optimization_summary.append({
                      'Model': model_name.upper(),
                      'Optimization': 'Grid/Random Search',
                      'Best Score': f"{results['test_results']['Overall_Score']:.1f}",
                      'Key Parameters': ', '.join([f"{k}={v}" for k, v in list(results['best_params'].items())[:2]])
                  })
              else:
                  optimization_summary.append({
                      'Model': model_name.upper(),
                      'Optimization': 'Default Parameters',
                      'Best Score': f"{results['test_results']['Overall_Score']:.1f}",
                      'Key Parameters': 'None'
                  })
          
          if optimization_summary:
              opt_df = pd.DataFrame(optimization_summary)
              st.dataframe(opt_df, use_container_width=True)
              
              # Show optimization insights
              optuna_models = [row['Model'] for row in optimization_summary if row['Optimization'] == 'Optuna']
              if optuna_models:
                  st.success(f"🎯 **Models optimized with Optuna:** {', '.join(optuna_models)}")
                  st.info("Optuna-optimized models typically show superior performance due to systematic hyperparameter search.")
          
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
                  
                  # Show optimization details for neural networks
                  if model_name in get_neural_network_models():
                      results = st.session_state.model_results[model_name]
                      if 'best_params' in results:
                          st.write(f"- Optimized with Optuna: {results['best_params']}")
                  
                  st.write("")
          
          # Recommendations
          st.header("Recommendations")
          
          if not summary_df.empty:
              best_model_name = summary_df.iloc[0]['Model'].lower()
              best_model_score = float(summary_df.iloc[0]['Overall Score'])
              
              st.write("Based on the comprehensive analysis:")
              
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
                  st.write("🌳 Tree-based model selected - excellent for capturing non-linear patterns and categorical relationships in short-term PM10 variations.")
              elif best_model_name in ['knn']:
                  st.write("🎯 KNN selected - performs well with local patterns and benefits significantly from categorical encoding!")
              elif best_model_name in ['lstm', 'ann', 'gru']:
                  st.write("🧠 Neural network model selected - optimized with Optuna for superior performance in capturing complex temporal dependencies.")
              elif best_model_name in ['arima', 'prophet']:
                  st.write("📉 Time series model selected - suitable for short-term forecasting with clear seasonal patterns.")
              
              # Optimization insights
              if best_model_name in get_neural_network_models():
                  best_results = st.session_state.model_results[best_model_name]
                  if 'best_params' in best_results:
                      st.write("**🔧 Optuna Optimization Benefits:**")
                      st.write("- Systematic exploration of hyperparameter space")
                      st.write("- Automated selection of optimal model architecture")
                      st.write("- Superior performance compared to default parameters")
                      st.write(f"- Best configuration: {best_results['best_params']}")
              
              # Feature engineering insights
              st.write("**Categorical Feature Benefits:**")
              st.write("- One-hot encoding provides better representation of temporal patterns")
              st.write("- Day-of-week and seasonal patterns are explicitly captured")
              st.write("- Model can learn specific impacts of weekends, holidays, and seasonal cycles")
              
              # Data quality recommendations
              if best_model_name in st.session_state.model_results and 'outliers' in st.session_state.model_results[best_model_name]:
                  outlier_info = st.session_state.model_results[best_model_name]['outliers']
                  
                  st.write("**Data Quality Summary:**")
                  st.write(f"- Total outliers detected: {outlier_info['total_detected']}")
                  st.write(f"- Outliers effectively handled: {outlier_info['total_handled']}")
                  st.write(f"- Outliers remaining: {outlier_info['total_remaining']}")
                  
                  if outlier_info['total_remaining'] > 0:
                      st.write(f"- Note: {outlier_info['total_remaining']} outliers remain after processing (this is normal with capping method)")
              
              # Short-term forecasting specific recommendations
              st.write("**Short-term Forecasting Notes:**")
              st.write("- 1-3 day forecasts are most reliable for immediate air quality planning")
              st.write("- Categorical features significantly improve temporal pattern recognition")
              st.write("- Neural networks with Optuna optimization show superior performance for complex patterns")
              st.write("- Consider real-time data updates for improved accuracy")
              st.write("- Monitor model performance regularly for optimal forecasting")
              
              # Data quality reminder
              if missing_info:
                  st.warning("⚠️ **Important:** Missing values were detected in the current dataset. "
                            "Ensure data quality is maintained for optimal model performance.")
              
              # Final optimization recommendation
              neural_models_available = [model for model in st.session_state.model_results.keys() if model in get_neural_network_models()]
              if neural_models_available:
                  st.info("🎯 **Pro Tip:** Neural network models (LSTM, ANN, GRU) with Optuna optimization "
                         "often provide the best performance for complex temporal forecasting tasks.")
      
      else:
          if st.session_state.data is None:
              st.info("Please upload a dataset using the sidebar.")
          else:
              st.info("Please train at least one model in the 'Model Training' tab.")

if __name__ == "__main__":
  main()