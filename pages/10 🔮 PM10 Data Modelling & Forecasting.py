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

# Smart Core Allocation Implementation
import os
import multiprocessing as mp

# Suppress warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="PM10 Data Modelling & Forecasting - Daily", page_icon="🔮", layout="wide")
if 'uploaded_file' not in st.session_state:
        st.session_state['uploaded_file'] = 'no'
uploaded_file = st.session_state['uploaded_file']

if(uploaded_file == 'yes'):
    if st.session_state.data_confirmed:
        imputed = st.session_state['imputed']
        if(imputed =='yes'):
            input_df = st.session_state['imputed_df'] 
            st.info('⚪ Imputed dataset is loaded for this page.') #💡
        else:
            input_df = st.session_state['df']
            st.info('⚪ Original dataset is loaded for this page.')

        # Checking null counts for PM10 and PM2.5
        nulls = input_df.isnull().sum().sum()
        
        if(nulls > 0):
            st.error('❌ Missing values are found in the dataset. For a better forecasting, please use the data cleaning and imputation tool to fill in the missing values.')

        else:

            #### START HERE #### 
            # Header
            st.title("🔮 PM10 Data Modelling & Forecasting (Short-Term Daily )")
            st.write("Enhanced forecasting with composite environmental features, excluding AQI, PM2.5, and NOx to prevent data leakage and improve model generalization.")

            # ============ SMART CORE ALLOCATION SYSTEM ============

            def get_optimal_core_allocation():
                """
                Smart core allocation based on system resources and task type
                Returns optimal n_jobs values for different operations
                """
                total_cores = mp.cpu_count()
                
                # Detect system type and adjust accordingly
                if total_cores <= 2:
                    # Low-end systems
                    core_config = {
                        'search_n_jobs': 1,
                        'cv_n_jobs': 1,
                        'tree_n_jobs': 1,
                        'ensemble_n_jobs': 2 if total_cores == 2 else 1
                    }
                elif total_cores <= 4:
                    # Mid-range systems
                    core_config = {
                        'search_n_jobs': 2,
                        'cv_n_jobs': 2,
                        'tree_n_jobs': 2,
                        'ensemble_n_jobs': 3
                    }
                elif total_cores <= 8:
                    # High-end consumer systems
                    core_config = {
                        'search_n_jobs': 4,
                        'cv_n_jobs': 3,
                        'tree_n_jobs': 4,
                        'ensemble_n_jobs': 6
                    }
                else:
                    # Server/workstation systems
                    reserve_cores = max(2, total_cores // 8)  # Reserve cores for system
                    available_cores = total_cores - reserve_cores
                    
                    core_config = {
                        'search_n_jobs': min(6, available_cores // 2),
                        'cv_n_jobs': min(4, available_cores // 3),
                        'tree_n_jobs': min(8, available_cores // 2),
                        'ensemble_n_jobs': min(12, int(available_cores * 0.8))
                    }
                
                # Add system info for debugging
                core_config['total_cores'] = total_cores
                core_config['detected_system'] = (
                    'low-end' if total_cores <= 2 else
                    'mid-range' if total_cores <= 4 else
                    'high-end' if total_cores <= 8 else
                    'server/workstation'
                )
                
                return core_config

            # Initialize core allocation (call this once at startup)
            CORE_CONFIG = get_optimal_core_allocation()

            # Extract specific values for easy access
            search_n_jobs = CORE_CONFIG['search_n_jobs']
            cv_n_jobs = CORE_CONFIG['cv_n_jobs'] 
            tree_n_jobs = CORE_CONFIG['tree_n_jobs']
            ensemble_n_jobs = CORE_CONFIG['ensemble_n_jobs']

            #### ---------------------------------------- ####
            def create_composite_features(df):
                """Create enhanced composite features for better PM10 prediction"""
                df_comp = df.copy()
                
                # 1. Traffic-Weather Interaction (Enhanced)
                if all(col in df.columns for col in ['TrafficV', 'WS', 'Temp']):
                    # Enhanced dispersion considering temperature inversion
                    df_comp['Traffic_Dispersion'] = df['TrafficV'] / (df['WS'] + 1)
                    df_comp['Traffic_Temp_Index'] = df['TrafficV'] * (df['Temp'] + 273.15) / 300
                    
                    # New: Temperature inversion indicator
                    df_comp['Temp_Inversion_Risk'] = np.where(df['Temp'] < 10, 
                                                            1 + (10 - df['Temp']) / 10, 
                                                            1.0)
                    df_comp['Effective_Dispersion'] = df_comp['Traffic_Dispersion'] * df_comp['Temp_Inversion_Risk']
                
                # 2. Wind Effectiveness (Enhanced)
                if all(col in df.columns for col in ['WS', 'WD']):
                    wind_x = df['WS'] * np.cos(np.radians(df['WD']))
                    wind_y = df['WS'] * np.sin(np.radians(df['WD']))
                    df_comp['Wind_Consistency'] = np.sqrt(wind_x**2 + wind_y**2)
                    
                    # Wind categories with better PM10 correlation
                    def get_wind_sector(wd):
                        if pd.isna(wd):
                            return 'Calm'
                        wd = float(wd) % 360
                        if 0 <= wd < 45 or 315 <= wd < 360:
                            return 'North'
                        elif 45 <= wd < 135:
                            return 'East'
                        elif 135 <= wd < 225:
                            return 'South'
                        else:
                            return 'West'
                    
                    df_comp['Wind_Sector'] = df['WD'].apply(get_wind_sector)
                    
                    # New: Calm conditions indicator (critical for PM10 accumulation)
                    df_comp['Calm_Conditions'] = (df['WS'] < 0.5).astype(float)
                    df_comp['Low_Wind'] = (df['WS'] < 2.0).astype(float)
                
                # 3. Enhanced Human Activity Index
                if all(col in df.columns for col in ['Total_Pedestrians', 'City_Centre_TVCount', 'TrafficV']):
                    # Normalize with robust scaling
                    ped_norm = (df['Total_Pedestrians'] - df['Total_Pedestrians'].quantile(0.1)) / \
                            (df['Total_Pedestrians'].quantile(0.9) - df['Total_Pedestrians'].quantile(0.1) + 1e-8)
                    tv_norm = (df['City_Centre_TVCount'] - df['City_Centre_TVCount'].quantile(0.1)) / \
                            (df['City_Centre_TVCount'].quantile(0.9) - df['City_Centre_TVCount'].quantile(0.1) + 1e-8)
                    traffic_norm = (df['TrafficV'] - df['TrafficV'].quantile(0.1)) / \
                                (df['TrafficV'].quantile(0.9) - df['TrafficV'].quantile(0.1) + 1e-8)
                    
                    # Clip to reasonable range
                    ped_norm = np.clip(ped_norm, 0, 1)
                    tv_norm = np.clip(tv_norm, 0, 1)
                    traffic_norm = np.clip(traffic_norm, 0, 1)
                    
                    df_comp['Activity_Index'] = (ped_norm * 0.2 + tv_norm * 0.3 + traffic_norm * 0.5)
                    
                    # New: Rush hour indicator
                    df_comp['Peak_Traffic'] = (traffic_norm > 0.7).astype(float)
                
                # 4. Enhanced Atmospheric Stability
                if all(col in df.columns for col in ['Temp', 'WS']):
                    df_comp['Atmospheric_Stability'] = df['Temp'] / (df['WS'] + 0.1)
                    
                    # Pasquill stability classes approximation
                    df_comp['Very_Stable'] = (df_comp['Atmospheric_Stability'] > df_comp['Atmospheric_Stability'].quantile(0.8)).astype(float)
                
                # 5. Nitrogen Chemistry (Enhanced)
                if all(col in df.columns for col in ['NO2', 'NO']):
                    df_comp['NO2_NO_Ratio'] = df['NO2'] / (df['NO'] + 1)
                    
                    # Total nitrogen oxides normalized
                    df_comp['Total_NOx_Normalized'] = (df['NO2'] + df['NO']) / \
                                                    ((df['NO2'] + df['NO']).quantile(0.9) + 1e-8)
                    
                    # Photo-oxidation indicator
                    df_comp['Photo_Oxidation'] = df_comp['NO2_NO_Ratio'] * (df['Temp'] > 15).astype(float)
                
                # 6. PM10-specific predictors
                if 'PM10' in df.columns:
                    # PM10 percentile for anomaly detection
                    df_comp['PM10_Percentile'] = df['PM10'].rank(pct=True)
                    
                    # High pollution episodes
                    df_comp['High_PM10'] = (df['PM10'] > df['PM10'].quantile(0.75)).astype(float)
                
                # 7. Emission accumulation potential
                if all(col in df.columns for col in ['TrafficV', 'WS', 'Temp']):
                    df_comp['Accumulation_Potential'] = (df['TrafficV'] * df_comp.get('Temp_Inversion_Risk', 1)) / \
                                                    (df['WS'] + 0.5)
                
                # Clean composite features
                composite_cols = list(set(df_comp.columns) - set(df.columns))
                
                for col in composite_cols:
                    if col in df_comp.columns:
                        df_comp[col] = df_comp[col].replace([np.inf, -np.inf], np.nan)
                        if df_comp[col].isnull().any():
                            df_comp[col] = df_comp[col].fillna(df_comp[col].median())
                
                return df_comp

            def check_missing_values(df, model_types=None):
                """Check for missing values in the dataframe"""
                missing_info = {}
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
                return ['knn',  'svr'] # 'arima', 'sarima', 'prophet', 'linear', 'ridge', 'lasso', , 'naive_bayes', 'lstm', 'ann', 'gru'

            def get_models_handling_missing_data():
                return ['Decision Tree', 'Random Forest', 'XGBoost']

            def get_neural_network_models():
                return ['lstm', 'ann', 'gru']

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

            def display_missing_value_warning(missing_info, model_type):
                """Display warning for models that can handle missing values"""
                if not missing_info:
                    return False
                
                st.warning(f"⚠️ **Missing Values Detected - {model_type.upper()} Will Handle Them Automatically**")
                
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
            def load_data():

                df = input_df
                # EXCLUDE SPECIFIED COLUMNS to prevent data leakage
                columns_to_exclude = ['AQI', 'PM2.5', 'NOx']
                st.info(f"Excluded columns: {', '.join(columns_to_exclude) }")
                
                # Convert Date to datetime
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
                df['Weekend'] = df['Date'].dt.dayofweek.apply(lambda x: 'Weekend' if x >= 5 else 'Weekday')
                df['Quarter'] = df['Date'].dt.quarter.map({1: 'Q1', 2: 'Q2', 3: 'Q3', 4: 'Q4'})
                df['MonthPart'] = df['Date'].dt.day.apply(lambda x: 'Early' if x <= 10 else ('Mid' if x <= 20 else 'Late'))
                
                return df

            @st.cache_data
            def create_daily_dataset(df):
                """Create daily aggregated dataset"""
                pm_cols = ['PM10']
                gas_cols = ['NO2', 'NO']
                weather_cols = ['Temp', 'WD', 'WS']
                traffic_cols = ['Total_Pedestrians', 'City_Centre_TVCount', 'TrafficV']
                
                agg_dict = {}
                for col in pm_cols + gas_cols:
                    if col in df.columns:
                        agg_dict[col] = 'mean'
                for col in weather_cols:
                    if col in df.columns:
                        agg_dict[col] = 'mean'
                for col in traffic_cols:
                    if col in df.columns:
                        agg_dict[col] = 'sum'
                
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

            def create_one_hot_features(df, categorical_cols=None):
                """Create one-hot encoded features for categorical variables"""
                if categorical_cols is None:
                    categorical_cols = get_categorical_features()
                    # Add Wind_Sector only if it exists in the dataframe
                    if 'Wind_Sector' in df.columns:
                        categorical_cols.append('Wind_Sector')
                
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

            def apply_one_hot_encoding(df, encoders, categorical_cols=None):
                """Apply existing one-hot encoders to new data - FIXED for categorical errors"""
                if categorical_cols is None:
                    categorical_cols = get_categorical_features()
                    if 'Wind_Sector' in df.columns and 'Wind_Sector' in encoders:
                        categorical_cols.append('Wind_Sector')
                
                df_encoded = df.copy()
                
                for col in categorical_cols:
                    if col in df.columns and col in encoders:
                        try:
                            # Ensure the column is string type, not categorical
                            df_encoded[col] = df_encoded[col].astype(str)
                            
                            encoded_features = encoders[col].transform(df_encoded[[col]])
                            feature_names = [f"{col}_{category}" for category in encoders[col].categories_[0][1:]]
                            encoded_df = pd.DataFrame(encoded_features, columns=feature_names, index=df.index)
                            df_encoded = pd.concat([df_encoded, encoded_df], axis=1)
                        except Exception as e:
                            # If encoding fails, skip this categorical column
                            st.warning(f"Warning: Could not encode {col}: {str(e)}")
                            continue
                
                return df_encoded

            def get_categorical_features():
                """Get list of categorical date features (excluding Wind_Sector if not present)"""
                return ['Month', 'DayOfWeek', 'Season', 'Weekend', 'Quarter', 'MonthPart']

            def get_base_features():
                """Get list of base features including composite features"""
                base_features = ['Year', 'Temp', 'WS', 'WD', 'NO2', 'NO']
                composite_features = [
                    'Traffic_Dispersion', 'Traffic_Temp_Index', 'Wind_Consistency', 
                    'Activity_Index', 'Atmospheric_Stability', 'NO2_NO_Ratio', 'Traffic_Emission_Factor'
                ]
                return base_features + composite_features

            def get_one_hot_feature_names(encoders):
                """Get all one-hot encoded feature names"""
                feature_names = []
                for col, encoder in encoders.items():
                    feature_names.extend([f"{col}_{category}" for category in encoder.categories_[0][1:]])
                return feature_names

            def detect_outliers(df, target_col, method='iqr', threshold=1.5):
                """Detect outliers using IQR method"""
                Q1 = df[target_col].quantile(0.25)
                Q3 = df[target_col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                outliers = (df[target_col] < lower_bound) | (df[target_col] > upper_bound)
                return outliers

            def handle_outliers_data(df, target_col, outliers, method='cap'):
                """Handle outliers using capping method"""
                df_clean = df.copy()
                outliers_before = int(outliers.sum())
                
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
                
                outlier_info = {
                    'detected': outliers_before,
                    'handled': outliers_handled,
                    'remaining': outliers_remaining,
                    'modified': values_modified,
                    'method': method,
                    'effectiveness': f"{(outliers_handled/outliers_before*100):.1f}%" if outliers_before > 0 else "0%"
                }
                
                return df_clean, outlier_info

            def add_lag_features(df, target_col, lag_days=[1, 3, 7], show_message=True):
                """Add enhanced lag features with focus on PM10 prediction"""
                df_copy = df.copy()
                
                # First create composite features
                df_copy = create_composite_features(df_copy)
                
                n_rows = len(df_copy)
                max_lag = max(lag_days) if lag_days else 0
                min_required = max_lag + 15
                
                # Adaptive lag days based on dataset size
                if n_rows < min_required:
                    if show_message:
                        st.info(f"📊 **Adapting features for dataset size ({n_rows} days)**")
                    
                    if n_rows < 50:
                        lag_days = [1, 2]
                        window_sizes = []
                    elif n_rows < 100:
                        lag_days = [1, 3, 7]
                        window_sizes = [3, 7]
                    else:
                        lag_days = [1, 3, 7, 14]
                        window_sizes = [3, 7, 14]
                else:
                    lag_days = [1, 3, 7, 14, 21]  # Extended lags for better patterns
                    window_sizes = [3, 7, 14, 21]
                
                # Features to lag - prioritize PM10-relevant features
                features_to_lag = [target_col]
                
                # Add composite features that correlate with PM10
                pm10_relevant_features = [
                    'Traffic_Dispersion', 'Effective_Dispersion', 'Wind_Consistency', 
                    'Activity_Index', 'Atmospheric_Stability', 'Accumulation_Potential',
                    'Calm_Conditions', 'Low_Wind', 'Very_Stable', 'Total_NOx_Normalized'
                ]
                
                for feature in pm10_relevant_features:
                    if feature in df_copy.columns:
                        features_to_lag.append(feature)
                
                # Create lag features
                for feature in features_to_lag:
                    for lag in lag_days:
                        if lag < n_rows - 10:
                            col_name = f'{feature}_lag_{lag}'
                            df_copy[col_name] = df_copy[feature].shift(lag)
                
                # Enhanced rolling statistics
                for window in window_sizes:
                    if window <= n_rows // 3:
                        min_periods = max(1, window // 3)
                        
                        # PM10 rolling statistics
                        df_copy[f'{target_col}_rolling_mean_{window}'] = df_copy[target_col].rolling(
                            window=window, min_periods=min_periods
                        ).mean().shift(1)
                        
                        df_copy[f'{target_col}_rolling_std_{window}'] = df_copy[target_col].rolling(
                            window=window, min_periods=min_periods
                        ).std().shift(1)
                        
                        df_copy[f'{target_col}_rolling_max_{window}'] = df_copy[target_col].rolling(
                            window=window, min_periods=min_periods
                        ).max().shift(1)
                        
                        # Exponentially weighted moving average (more weight to recent values)
                        df_copy[f'{target_col}_ewm_{window}'] = df_copy[target_col].ewm(
                            span=window, min_periods=min_periods
                        ).mean().shift(1)
                        
                        # Traffic rolling mean (emission proxy)
                        if 'TrafficV' in df_copy.columns:
                            df_copy[f'TrafficV_rolling_mean_{window}'] = df_copy['TrafficV'].rolling(
                                window=window, min_periods=min_periods
                            ).mean().shift(1)
                
                # Interaction features
                if 'Traffic_Dispersion' in df_copy.columns and f'{target_col}_lag_1' in df_copy.columns:
                    df_copy['PM10_Dispersion_Interaction'] = df_copy[f'{target_col}_lag_1'] * df_copy['Traffic_Dispersion']
                
                # Handle NaN values more carefully
                initial_length = len(df_copy)
                
                # Drop rows where target is NaN
                target_nan_count = df_copy[target_col].isnull().sum()
                if target_nan_count > 0:
                    if show_message:
                        st.info(f"Removing {target_nan_count} rows with missing target values")
                    df_copy = df_copy.dropna(subset=[target_col])
                
                # Smart filling for feature columns
                feature_cols = [col for col in df_copy.columns if any(x in col for x in ['_lag_', '_rolling_', '_ewm_'])]
                
                for col in feature_cols:
                    if df_copy[col].isnull().any():
                        # Use forward fill then backward fill with limits
                        df_copy[col] = df_copy[col].fillna(method='ffill', limit=3)
                        df_copy[col] = df_copy[col].fillna(method='bfill', limit=3)
                        
                        # Fill remaining with median
                        if df_copy[col].isnull().any():
                            df_copy[col] = df_copy[col].fillna(df_copy[col].median())
                
                # Final cleanup
                df_copy = df_copy.dropna()
                df_copy = df_copy.reset_index(drop=True)
                final_length = len(df_copy)
                
                if show_message:
                    rows_lost = initial_length - final_length
                    if rows_lost > 0:
                        st.info(f"📉 Feature engineering removed {rows_lost} rows - {final_length} rows remaining")
                    st.success("✅ **Enhanced feature engineering completed**")
                
                return df_copy

            def add_enhanced_features(df, target_col='PM10'):
                """Add enhanced features for PM10 prediction"""
                df_out = df.copy()
                
                # Change features
                df_out[f'{target_col}_daily_change'] = df_out[target_col].diff()
                df_out[f'{target_col}_daily_pct_change'] = df_out[target_col].pct_change()
                
                # Acceleration (change of change)
                df_out[f'{target_col}_acceleration'] = df_out[f'{target_col}_daily_change'].diff()
                
                # Pollution persistence
                if 'Traffic_Dispersion' in df_out.columns:
                    df_out['Pollution_Momentum'] = df_out[f'{target_col}_daily_change'] * df_out['Traffic_Dispersion']
                    df_out['Dispersion_Effectiveness'] = df_out[target_col] / (df_out['Traffic_Dispersion'] + 1)
                
                # Weekend effect
                if 'Date' in df_out.columns:
                    df_out['Is_Weekend'] = pd.to_datetime(df_out['Date']).dt.dayofweek.isin([5, 6]).astype(float)
                
                # Fill NaNs carefully
                for col in df_out.columns:
                    if df_out[col].dtype in ['float64', 'int64']:
                        df_out[col] = df_out[col].replace([np.inf, -np.inf], np.nan)
                        if df_out[col].isnull().any():
                            # Use interpolation for smooth features
                            if 'change' in col or 'momentum' in col:
                                df_out[col] = df_out[col].interpolate(method='linear', limit=3)
                            df_out[col] = df_out[col].fillna(df_out[col].median())
                
                return df_out

            def split_data(df, target_col, train_size=0.60, val_size=0.20, test_size=0.20):
                """Split data with conservative ratios to prevent overfitting"""
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
                """Extract feature and target arrays with composite features and one-hot encoding"""
                
                # Get available categorical columns
                available_categorical_cols = [col for col in categorical_cols if col in df.columns]
                
                # Apply one-hot encoding for categorical features
                if encoders is None:
                    df_encoded, encoders = create_one_hot_features(df, available_categorical_cols)
                else:
                    df_encoded = apply_one_hot_encoding(df, encoders, available_categorical_cols)
                
                # Get composite features that exist in the dataframe
                composite_features = [
                    'Traffic_Dispersion', 'Traffic_Temp_Index', 'Wind_Consistency', 
                    'Activity_Index', 'Atmospheric_Stability', 'NO2_NO_Ratio', 'Traffic_Emission_Factor',
                    'Pollution_Momentum', 'Dispersion_Effectiveness'
                ]
                
                available_composite = [f for f in composite_features if f in df_encoded.columns]
                
                # Get lag features for composite features
                composite_lag_features = [col for col in df_encoded.columns 
                                        if any(comp in col for comp in composite_features) and '_lag_' in col]
                
                # Get all feature columns
                one_hot_features = get_one_hot_feature_names(encoders)
                all_features = base_features + lag_features + one_hot_features + available_composite + composite_lag_features
                
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

            def perform_kfold_cv(model, X, y, n_folds=3, scoring='neg_mean_squared_error'):
                """Reduced K-fold CV to prevent overfitting assessment bias"""
                cv = TimeSeriesSplit(n_splits=n_folds)
                scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=cv_n_jobs) #PREVIOUS: n_jobs=-1
                return -scores

            # Enhanced hyperparameter spaces for better PM10 prediction #rf 'max_samples': [0.7, 0.8, 0.9]
            def get_hyperparameter_space(model_type):
                """Get optimized hyperparameter search space for PM10 prediction"""
                param_spaces = {
                    'linear': {'fit_intercept': [True, False]},
                    'ridge': {'alpha': [0.1, 1, 10, 100, 1000], 'fit_intercept': [True, False]},
                    'lasso': {'alpha': [0.01, 0.1, 1, 10, 100], 'fit_intercept': [True, False]},
                    'Decision Tree': {
                        # 'max_depth': [5, 10, 15, 20],
                        # 'min_samples_split': [20, 50, 100],
                        # 'min_samples_leaf': [10, 20, 50],
                        # 'max_features': ['sqrt', 'log2', None]
                        #OPTIMIZED1
                        # 'max_depth': [10, 15, 20],  # Reduced from [5, 10, 15, 20]
                        # 'min_samples_split': [30, 50, 80],  # Reduced from [20, 50, 100]
                        # 'min_samples_leaf': [15, 25],  # Reduced from [10, 20, 50]
                        # 'max_features': ['sqrt', None],  # Kept essential options
                        # # Total: 3 * 3 * 2 * 2 = 36 combinations (62% reduction)
                        # J- OPTIMIZED2
                        'max_depth': [11, 12, 13, 14],  # Narrowed around your best (12)
                        'min_samples_split': [50, 60, 70, 80],  # Centered on your best (60)
                        'min_samples_leaf': [25, 30, 35],  # Focused around your best (30)
                        'max_features': [0.6, 0.7, 0.8, 'sqrt']  # Your best was 0.7
                    },
                    'knn': {
                        'n_neighbors': [5, 10, 15, 20, 30],
                        'weights': ['uniform', 'distance'],
                        'metric': ['euclidean', 'manhattan']
                    },
                    'Random Forest': {
                        # 'n_estimators': [100, 200, 300],
                        # 'max_depth': [50, 60, 70, 80],
                        # 'min_samples_split': c 50],
                        # 'min_samples_leaf': [3, 5, 7, 10],
                        # 'max_features': ['sqrt', 'log2'], 
                        # 'max_samples':[0.7,0.8,0.9],
                        # 'ccp_alpha': [0.0, 0.01],  # Add pruning   

                        #87.6
                        # 'n_estimators': [175, 200, 250, 300],
                        # 'max_depth': [15, 20, 25, 30],          
                        # 'min_samples_split': [2, 3, 4, 5],
                        # 'min_samples_leaf': [3, 5, 6, 7],  
                        # 'max_features': [0.25, 0.3, 0.35, 0.4],      
                        # 'max_samples': [0.7, 0.8, 0.9],      
                        # 'min_weight_fraction_leaf': [0.0, 0.005, 0.001]
                        # 'bootstrap': [True]     
                        #TEST -84.2 (J)
                        # 'n_estimators': [150, 175, 200, 250],
                        # 'max_depth': [5, 10, 15],          
                        # 'min_samples_split': [25, 35, 45],
                        # 'min_samples_leaf': [9, 12, 15],  
                        # 'max_features': [0.25, 0.3, 0.35],      
                        # 'max_samples': [0.6, 0.7, 0.8, 0.9],      
                        # 'min_weight_fraction_leaf': [0.0, 0.005, 0.01]
                        # TEST (C)
                        # 'n_estimators': [200, 300, 400, 500],  # More trees for stability
                        # 'max_depth': [10, 15, 20, 25, None],   # Allow deeper trees like DT
                        # 'min_samples_split': [2, 5, 10, 20],   # Less restrictive
                        # 'min_samples_leaf': [1, 2, 4, 8],      # Less restrictive
                        # 'max_features': ['sqrt', 'log2', 0.5, 0.7, None],  # More features per split
                        # 'max_samples': [0.8, 0.9, 1.0],         # Include full sampling for clean data
                        # 'min_weight_fraction_leaf': [0.0],  
                        #24/06  
                        # 'n_estimators': [450, 500],
                        # 'max_depth': [12, 13],
                        # 'min_samples_split': [15, 16, 17],
                        # 'min_samples_leaf': [5, 6],
                        # 'max_features': [0.36, 0.37],
                        # 'max_samples': [0.75, 0.77],
                        # 'min_weight_fraction_leaf': [0.0],   
                        ### EXHAUSTIVE
                        'n_estimators': [400, 500, 600],
                        'max_depth': [20, 25, None],
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 4],
                        'max_features': [0.6, 0.7, 0.8],  # This is the key change!
                        'max_samples': [0.9, 1.0], 
                        'random_state': [42] 
                        #OPTIMIZED1  
                        # 'max_depth': [10, 15, 20],  # Reduced from [5, 10, 15, 20]
                        # 'min_samples_split': [30, 50, 80],  # Reduced from [20, 50, 100]
                        # 'min_samples_leaf': [15, 25],  # Reduced from [10, 20, 50]
                        # 'max_features': ['sqrt', None],  # Kept essential options
                        # Total: 3 * 3 * 2 * 2 = 36 combinations (62% reduction)   
                        #UPDATED 6.12 PM - JANICE
                        # 'max_depth': [11, 12],  # Slightly shallower options
                        # 'min_samples_split': [65, 70],  # Higher = more conservative
                        # 'min_samples_leaf': [30, 35],  # Higher = more stable
                        # 'max_features': [0.6, 0.65],
                        # 'n_estimators': [300, 400],  # More trees
                        # 'random_state': [42]
                    },
                    'XGBoost': {
                        #EXHAUSTIVE
                        # 'n_estimators': [100, 200, 300],
                        # 'learning_rate': [0.01, 0.05, 0.1],
                        # 'max_depth': [5, 7, 10],
                        # 'subsample': [0.7, 0.8, 0.9],
                        # 'colsample_bytree': [0.7, 0.8, 0.9],
                        # 'reg_alpha': [0, 0.1, 1, 10],
                        # 'reg_lambda': [1, 10, 100],
                        # 'min_child_weight': [1, 5, 10] 
                        #24/06
                        # 'n_estimators': [450, 500, 550, 600],         # Center around 500
                        # 'learning_rate': [0.08, 0.09, 0.1, 0.11],     # Fine-tune around 0.1
                        # 'max_depth': [6, 7, 8],                       # Include 6 and 8
                        # 'subsample': [0.65, 0.7, 0.75],               # Fine-tune around 0.7
                        # 'colsample_bytree': [0.75, 0.8, 0.85],        # Fine-tune around 0.8
                        # 'reg_alpha': [0.05, 0.1, 0.15],               # Fine-tune around 0.1
                        # 'reg_lambda': [80, 100, 120],                 # Fine-tune around 100
                        # 'min_child_weight': [4, 5, 6],                # Fine-tune around 5
                        # 'gamma': [0.05, 0.1, 0.15]                    # Fine-tune around 0.1
                        #OPTIMIZED1
                        'n_estimators': [200, 300],  # Reduced from [100, 200, 300]
                        'learning_rate': [0.05, 0.07],  # Focused around your best (0.05)
                        'max_depth': [8, 10],  # Focused around your best (10)
                        'subsample': [0.7, 0.8],  # Focused around your best (0.7)
                        'colsample_bytree': [0.8, 0.9],  # Focused around your best (0.9)
                        'reg_alpha': [10, 20],  # Focused around your best (10)
                        'reg_lambda': [10, 30],  # Focused around your best (10)
                        'min_child_weight': [10, 15],  # Focused around your best (10)
                        # Total: 2 * 2 * 2 * 2 * 2 * 2 * 2 * 2 = 256 combinations (95% reduction!)
                    },
                    'svr': {
                        'C': [0.1, 1.0, 10.0, 100.0],
                        'epsilon': [0.01, 0.1, 0.5, 1.0],
                        'kernel': ['linear', 'rbf', 'poly'],
                        'gamma': ['scale', 'auto']
                    },
                    'naive_bayes': {'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]}
                }
                
                return param_spaces.get(model_type, {})

            # def tune_hyperparameters(model_class, X_train, y_train, param_space, cv_folds=3, scoring='neg_mean_squared_error', method='random'):
            def tune_hyperparameters(model_class, X_train, y_train, param_space, cv_folds=3, 
                                    scoring='neg_mean_squared_error', method='random', auto_cv=True, progress_callback=None): #ADDED PROGRESS
                """Conservative hyperparameter tuning"""
                if len(param_space) == 0:
                    return model_class(), {}, 0
                
                # Adaptive CV folds based on dataset size (only if auto_cv is True)
                if auto_cv:
                    dataset_size = len(X_train)
                    if dataset_size > 10000:
                        cv_folds = 3  # Sufficient for large datasets
                    elif dataset_size > 5000:
                        cv_folds = 4
                    else:
                        cv_folds = 5
                # If auto_cv is False, use the user's cv_folds value directly

                cv = TimeSeriesSplit(n_splits=cv_folds)
                    
                if method == 'grid':
                    search = GridSearchCV(
                        model_class(),
                        param_space,
                        cv=cv,
                        scoring=scoring,
                        n_jobs=search_n_jobs,  # UPDATED: Smart allocation; PREVIOUS = -1
                        verbose=1 if progress_callback else 0  # Add verbose if progress callback
                    )
                else:
                    search = RandomizedSearchCV(
                        model_class(),
                        param_space,
                        n_iter=50,
                        cv=cv,
                        scoring=scoring,
                        n_jobs=search_n_jobs,  # UPDATED: Smart allocation; PREVIOUS = -1
                        verbose=1 if progress_callback else 0,  # Add verbose if progress callback
                        random_state=42
                    )
                
                search.fit(X_train, y_train)
                return search.best_estimator_, search.best_params_, search.best_score_


            def prepare_data_for_distance_models(X_train, X_val, X_test, y_train):
                """Ensure no NaN values and apply robust scaling"""
                # Use SimpleImputer to fill any potential remaining NaN values
                imputer = SimpleImputer(strategy='median')
                X_train_clean = imputer.fit_transform(X_train)
                X_val_clean = imputer.transform(X_val)
                X_test_clean = imputer.transform(X_test)
                
                # Clean target variable
                y_train_clean = np.nan_to_num(y_train, nan=np.nanmedian(y_train))

                # Apply RobustScaler
                scaler = RobustScaler()
                X_train_scaled = scaler.fit_transform(X_train_clean)
                X_val_scaled = scaler.transform(X_val_clean)
                X_test_scaled = scaler.transform(X_test_clean)

                return X_train_scaled, X_val_scaled, X_test_scaled, y_train_clean, scaler

            # OPTUNA OPTIMIZATION
            def optimize_lstm_with_optuna(X_train, y_train, X_val, y_val, time_steps=7, n_trials=10):
                """Conservative LSTM optimization"""
                
                def objective(trial):
                    units = trial.suggest_categorical('units', [32, 64])
                    dropout = trial.suggest_float('dropout', 0.3, 0.6)
                    batch_size = trial.suggest_categorical('batch_size', [32, 64])
                    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-3, log=True)
                    
                    try:
                        X_train_seq, y_train_seq = create_sequences(X_train, y_train, time_steps)
                        X_val_seq, y_val_seq = create_sequences(X_val, y_val, time_steps)
                        
                        if len(X_train_seq) == 0 or len(X_val_seq) == 0:
                            return float('inf')
                        
                        model = Sequential([
                            LSTM(units, return_sequences=False, input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])),
                            Dropout(dropout),
                            Dense(16),
                            Dropout(dropout),
                            Dense(1)
                        ])
                        
                        model.compile(
                            optimizer=Adam(learning_rate=learning_rate),
                            loss='mse',
                            metrics=['mae']
                        )
                        
                        early_stopping = EarlyStopping(
                            monitor='val_loss', 
                            patience=5,
                            restore_best_weights=True,
                            min_delta=0.001
                        )
                        
                        history = model.fit(
                            X_train_seq, y_train_seq,
                            epochs=50,
                            batch_size=batch_size,
                            validation_data=(X_val_seq, y_val_seq),
                            callbacks=[early_stopping],
                            verbose=0
                        )
                        
                        val_loss = min(history.history['val_loss'])
                        return val_loss
                        
                    except Exception as e:
                        return float('inf')
                
                study = optuna.create_study(direction='minimize')
                study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
                
                return study.best_params, study.best_value

            def optimize_ann_with_optuna(X_train, y_train, X_val, y_val, n_trials=10):
                """Conservative ANN optimization"""
                
                def objective(trial):
                    hidden_layers = trial.suggest_int('hidden_layers', 1, 2)
                    neurons = trial.suggest_categorical('neurons', [32, 64])
                    dropout = trial.suggest_float('dropout', 0.3, 0.6)
                    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-3, log=True)
                    batch_size = trial.suggest_categorical('batch_size', [32, 64])
                    
                    try:
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
                        
                        early_stopping = EarlyStopping(
                            monitor='val_loss', 
                            patience=5,
                            restore_best_weights=True,
                            min_delta=0.001
                        )
                        
                        history = model.fit(
                            X_train, y_train,
                            epochs=50,
                            batch_size=batch_size,
                            validation_data=(X_val, y_val),
                            callbacks=[early_stopping],
                            verbose=0
                        )
                        
                        val_loss = min(history.history['val_loss'])
                        return val_loss
                        
                    except Exception as e:
                        return float('inf')
                
                study = optuna.create_study(direction='minimize')
                study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
                
                return study.best_params, study.best_value

            def optimize_gru_with_optuna(X_train, y_train, X_val, y_val, time_steps=7, n_trials=10):
                """Conservative GRU optimization"""
                
                def objective(trial):
                    units = trial.suggest_categorical('units', [32, 64])
                    dropout = trial.suggest_float('dropout', 0.3, 0.6)
                    batch_size = trial.suggest_categorical('batch_size', [32, 64])
                    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-3, log=True)
                    
                    try:
                        X_train_seq, y_train_seq = create_sequences(X_train, y_train, time_steps)
                        X_val_seq, y_val_seq = create_sequences(X_val, y_val, time_steps)
                        
                        if len(X_train_seq) == 0 or len(X_val_seq) == 0:
                            return float('inf')
                        
                        model = Sequential([
                            GRU(units, return_sequences=False, input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])),
                            Dropout(dropout),
                            Dense(16),
                            Dropout(dropout),
                            Dense(1)
                        ])
                        
                        model.compile(
                            optimizer=Adam(learning_rate=learning_rate),
                            loss='mse',
                            metrics=['mae']
                        )
                        
                        early_stopping = EarlyStopping(
                            monitor='val_loss', 
                            patience=5,
                            restore_best_weights=True,
                            min_delta=0.001
                        )
                        
                        history = model.fit(
                            X_train_seq, y_train_seq,
                            epochs=50,
                            batch_size=batch_size,
                            validation_data=(X_val_seq, y_val_seq),
                            callbacks=[early_stopping],
                            verbose=0
                        )
                        
                        val_loss = min(history.history['val_loss'])
                        return val_loss
                        
                    except Exception as e:
                        return float('inf')
                
                study = optuna.create_study(direction='minimize')
                study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
                
                return study.best_params, study.best_value

            # MODEL TRAINING FUNCTIONS
            def train_decision_tree(X_train, y_train, X_val, y_val, tune_params=True, n_folds=3, auto_cv=True):
                """Train Decision Tree with conservative hyperparameters"""
                if tune_params:
                    param_space = get_hyperparameter_space('Decision Tree')
                    model, best_params, _ = tune_hyperparameters(
                        DecisionTreeRegressor, X_train, y_train, param_space, 
                        method='random', cv_folds=n_folds, auto_cv=auto_cv
                    )
                else:
                    model = DecisionTreeRegressor(random_state=42)
                    model.fit(X_train, y_train)
                    best_params = {}
                
                # Use adaptive CV for cross-validation scores too
                if auto_cv:
                    dataset_size = len(X_train)
                    if dataset_size > 10000:
                        n_folds = 3
                    elif dataset_size > 5000:
                        n_folds = 4
                    else:
                        n_folds = 5

                cv_scores = perform_kfold_cv(model, X_train, y_train, n_folds=n_folds)
                y_train_pred = model.predict(X_train)
                y_val_pred = model.predict(X_val)
                
                train_results = evaluate_model(y_train, y_train_pred, "Decision Tree (Train)")
                val_results = evaluate_model(y_val, y_val_pred, "Decision Tree (Validation)")
                
                train_results['CV_MSE_Mean'] = np.mean(cv_scores)
                train_results['CV_MSE_Std'] = np.std(cv_scores)
                
                return model, train_results, val_results, best_params

            def train_knn(X_train, y_train, X_val, y_val, tune_params=True, n_folds=3, auto_cv=True):
                """Train K-Nearest Neighbors with robust data preparation"""
                st.info("🔍 **KNN Data Prep**: Applying robust imputation and scaling...")
                
                X_train_scaled, X_val_scaled, _, y_train_clean, scaler = prepare_data_for_distance_models(
                    X_train, X_val, X_train, y_train
                )
                
                y_val_clean = np.nan_to_num(y_val, nan=np.nanmedian(y_train))

                st.success("✅ **KNN Data Prep Complete**")
                
                if tune_params:
                    param_space = get_hyperparameter_space('knn')
                    model, best_params, _ = tune_hyperparameters(
                        KNeighborsRegressor, X_train_scaled, y_train_clean, param_space, 
                        method='random', cv_folds=n_folds,  auto_cv=auto_cv
                    )
                else:
                    model = KNeighborsRegressor(n_neighbors=5, weights='distance', n_jobs=cv_n_jobs) # UPDATED: Smart allocation for KNN
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
                
                return model, train_results, val_results, best_params, scaler

            def train_random_forest(X_train, y_train, X_val, y_val, tune_params=True, n_folds=3, auto_cv=True):
                """Train Random Forest with conservative hyperparameters"""
                # Add this to your training functions:

                if tune_params:
                    param_space = get_hyperparameter_space('Random Forest')
                    if 'bootstrap' not in param_space:
                        param_space['bootstrap'] = [True] #True

                    model, best_params, _ = tune_hyperparameters(
                        RandomForestRegressor, X_train, y_train, param_space, 
                        method='random', cv_folds=n_folds, auto_cv=auto_cv
                    )
                else:
                    model = RandomForestRegressor(n_estimators=500, 
                        max_depth=None, min_samples_split=2,
                        min_samples_leaf=1,max_samples=1.0,
                        random_state=42, n_jobs=ensemble_n_jobs, bootstrap=True) #True , max_samples=0.7, #SMART: n_jobs=search_n_jobs
                    model.fit(X_train, y_train)
                    # best_params = {'n_estimators': 100}
                
                cv_scores = perform_kfold_cv(model, X_train, y_train, n_folds=n_folds)
                y_train_pred = model.predict(X_train)
                y_val_pred = model.predict(X_val)
                
                train_results = evaluate_model(y_train, y_train_pred, "Random Forest (Train)")
                val_results = evaluate_model(y_val, y_val_pred, "Random Forest (Validation)")
                
                train_results['CV_MSE_Mean'] = np.mean(cv_scores)
                train_results['CV_MSE_Std'] = np.std(cv_scores)
                
                return model, train_results, val_results, best_params

            def train_xgboost(X_train, y_train, X_val, y_val, tune_params=True, n_folds=3, auto_cv=True):
                """Train XGBoost with conservative hyperparameters"""
                if tune_params:
                    param_space = get_hyperparameter_space('XGBoost')
                    model, best_params, _ = tune_hyperparameters(
                        XGBRegressor, X_train, y_train, param_space, 
                        method='random', cv_folds=n_folds, auto_cv=auto_cv
                    )
                else:
                    model = XGBRegressor(n_estimators=100, random_state=42, n_jobs=tree_n_jobs) # UPDATED: Smart allocation n_jobs=-1
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

            def train_svr(X_train, y_train, X_val, y_val, tune_params=True, n_folds=3, auto_cv=True):
                """Train SVR with robust data preparation"""
                st.info("🔍 **SVR Data Prep**: Applying robust imputation and scaling...")

                X_train_scaled, X_val_scaled, _, y_train_clean, scaler = prepare_data_for_distance_models(
                    X_train, X_val, X_train, y_train
                )

                y_val_clean = np.nan_to_num(y_val, nan=np.nanmedian(y_train))

                st.success("✅ **SVR Data Prep Complete**")
                
                if tune_params:
                    param_space = get_hyperparameter_space('svr')
                    model, best_params, _ = tune_hyperparameters(
                        SVR, X_train_scaled, y_train_clean, param_space, 
                        method='random', cv_folds=n_folds, auto_cv=auto_cv
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
                
                return model, train_results, val_results, best_params, scaler

            def train_linear_regression(X_train, y_train, X_val, y_val, tune_params=True, n_folds=3, auto_cv=True):
                """Train Linear Regression"""
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

            def train_ridge(X_train, y_train, X_val, y_val, tune_params=True, n_folds=3, auto_cv=True):
                """Train Ridge Regression"""
                if tune_params:
                    param_space = get_hyperparameter_space('ridge')
                    model, best_params, _ = tune_hyperparameters(
                        Ridge, X_train, y_train, param_space, 
                        method='grid', cv_folds=n_folds, auto_cv=auto_cv
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

            def train_lasso(X_train, y_train, X_val, y_val, tune_params=True, n_folds=3, auto_cv=True):
                """Train Lasso Regression"""
                if tune_params:
                    param_space = get_hyperparameter_space('lasso')
                    model, best_params, _ = tune_hyperparameters(
                        Lasso, X_train, y_train, param_space, 
                        method='grid', cv_folds=n_folds, auto_cv=auto_cv
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
                """Create sequences for LSTM/GRU models"""
                if len(X) < time_steps + 1:
                    return np.array([]), np.array([])
                
                Xs, ys = [], []
                for i in range(len(X) - time_steps):
                    Xs.append(X[i:(i + time_steps)])
                    ys.append(y[i + time_steps])
                return np.array(Xs), np.array(ys)

            def train_lstm(X_train, y_train, X_val, y_val, tune_params=True, n_folds=3, time_steps=7, use_optuna=True):
                """Train LSTM with Optuna optimization"""
                
                st.info("🧠 **Training LSTM Model** - Creating sequences and optimizing...")
                
                X_train_seq, y_train_seq = create_sequences(X_train, y_train, time_steps)
                X_val_seq, y_val_seq = create_sequences(X_val, y_val, time_steps)
                
                if len(X_train_seq) == 0 or len(X_val_seq) == 0:
                    raise ValueError("Not enough data to create sequences for LSTM")
                
                if tune_params and use_optuna:
                    st.info("🔧 **Optimizing LSTM with Optuna**")
                    best_params, best_score = optimize_lstm_with_optuna(
                        X_train, y_train, X_val, y_val, time_steps=time_steps, n_trials=10
                    )
                    st.success(f"✅ **Optuna optimization complete** - Best validation loss: {best_score:.4f}")
                else:
                    best_params = {'units': 64, 'dropout': 0.2, 'batch_size': 32, 'learning_rate': 0.001}
                
                # Build final model with best parameters
                model = Sequential([
                    LSTM(best_params['units'], return_sequences=False, input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])),
                    Dropout(best_params['dropout']),
                    Dense(16),
                    Dropout(best_params['dropout']),
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
                
                # Predictions
                y_train_pred = model.predict(X_train_seq).flatten()
                y_val_pred = model.predict(X_val_seq).flatten()
                
                y_train_actual = y_train_seq
                y_val_actual = y_val_seq
                
                train_results = evaluate_model(y_train_actual, y_train_pred, "LSTM (Train)")
                val_results = evaluate_model(y_val_actual, y_val_pred, "LSTM (Validation)")
                
                train_results['CV_MSE_Mean'] = train_results['MSE']
                train_results['CV_MSE_Std'] = 0.0
                
                st.success("✅ **LSTM Training Complete!**")
                
                return model, train_results, val_results, best_params

            def train_gru(X_train, y_train, X_val, y_val, tune_params=True, n_folds=3, time_steps=7, use_optuna=True):
                """Train GRU with Optuna optimization"""
                
                st.info("🧠 **Training GRU Model** - Creating sequences and optimizing...")
                
                X_train_seq, y_train_seq = create_sequences(X_train, y_train, time_steps)
                X_val_seq, y_val_seq = create_sequences(X_val, y_val, time_steps)
                
                if len(X_train_seq) == 0 or len(X_val_seq) == 0:
                    raise ValueError("Not enough data to create sequences for GRU")
                
                if tune_params and use_optuna:
                    st.info("🔧 **Optimizing GRU with Optuna**")
                    best_params, best_score = optimize_gru_with_optuna(
                        X_train, y_train, X_val, y_val, time_steps=time_steps, n_trials=10
                    )
                    st.success(f"✅ **Optuna optimization complete** - Best validation loss: {best_score:.4f}")
                else:
                    best_params = {'units': 64, 'dropout': 0.2, 'batch_size': 32, 'learning_rate': 0.001}
                
                # Build final model with best parameters
                model = Sequential([
                    GRU(best_params['units'], return_sequences=False, input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])),
                    Dropout(best_params['dropout']),
                    Dense(16),
                    Dropout(best_params['dropout']),
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
                
                # Predictions
                y_train_pred = model.predict(X_train_seq).flatten()
                y_val_pred = model.predict(X_val_seq).flatten()
                
                y_train_actual = y_train_seq
                y_val_actual = y_val_seq
                
                train_results = evaluate_model(y_train_actual, y_train_pred, "GRU (Train)")
                val_results = evaluate_model(y_val_actual, y_val_pred, "GRU (Validation)")
                
                train_results['CV_MSE_Mean'] = train_results['MSE']
                train_results['CV_MSE_Std'] = 0.0
                
                st.success("✅ **GRU Training Complete!**")
                
                return model, train_results, val_results, best_params

            def train_ann(X_train, y_train, X_val, y_val, tune_params=True, n_folds=3, use_optuna=True):
                """Train ANN with Optuna optimization"""
                
                st.info("🧠 **Training ANN Model** - Optimizing architecture...")
                
                # Ensure proper shapes
                if y_train.ndim > 1:
                    y_train = y_train.ravel()
                if y_val.ndim > 1:
                    y_val = y_val.ravel()
                
                if tune_params and use_optuna:
                    st.info("🔧 **Optimizing ANN with Optuna**")
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
                
                # Predictions
                y_train_pred = model.predict(X_train).flatten()
                y_val_pred = model.predict(X_val).flatten()
                
                train_results = evaluate_model(y_train, y_train_pred, "ANN (Train)")
                val_results = evaluate_model(y_val, y_val_pred, "ANN (Validation)")
                
                train_results['CV_MSE_Mean'] = train_results['MSE']
                train_results['CV_MSE_Std'] = 0.0
                
                st.success("✅ **ANN Training Complete!**")
                
                return model, train_results, val_results, best_params

            # Time Series Models
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

            # Forecasting Functions
            def create_next_row_features_categorical(next_date, current_df, target_col, feature_cols, forecasts, encoders):
                """Create feature row for next prediction with composite features - FIXED for categorical errors"""
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
                
                # Estimate base environmental features (use last known values)
                base_env_features = ['Temp', 'WS', 'WD', 'NO2', 'NO', 'TrafficV', 'Total_Pedestrians', 'City_Centre_TVCount']
                for feature in base_env_features:
                    if feature in current_df.columns:
                        next_row[feature] = current_df[feature].iloc[-1]
                
                # Create dataframe for encoding
                next_df = pd.DataFrame([next_row])
                
                # Create composite features for the next row
                next_df = create_composite_features(next_df)
                
                # Apply one-hot encoding - get available categorical features
                available_categorical_cols = [col for col in get_categorical_features() if col in next_df.columns]
                if 'Wind_Sector' in next_df.columns and 'Wind_Sector' in encoders:
                    available_categorical_cols.append('Wind_Sector')
                
                next_df_encoded = apply_one_hot_encoding(next_df, encoders, available_categorical_cols)
                
                # Add lag and rolling features
                for col in feature_cols:
                    if col not in next_df_encoded.columns:
                        if '_lag_' in col:
                            lag_num = int(col.split('_')[-1])
                            base_feature = '_'.join(col.split('_')[:-2])
                            
                            if base_feature == target_col:
                                if lag_num <= len(forecasts):
                                    next_df_encoded[col] = forecasts[-lag_num]
                                else:
                                    idx = -(lag_num - len(forecasts))
                                    if idx >= -len(current_df):
                                        next_df_encoded[col] = current_df[target_col].iloc[idx]
                                    else:
                                        next_df_encoded[col] = current_df[target_col].median()
                            else:
                                # For composite feature lags, use last known value
                                if base_feature in current_df.columns:
                                    next_df_encoded[col] = current_df[base_feature].iloc[-1]
                                else:
                                    next_df_encoded[col] = 0
                                    
                        elif '_rolling_' in col:
                            parts = col.split('_')
                            window = int(parts[-1])
                            stat = parts[-2]
                            
                            # Get values for rolling calculation
                            values = list(current_df[target_col].iloc[-(window-1):].values) + forecasts
                            if len(values) >= window:
                                values = values[-window:]
                            
                            if stat == 'mean':
                                next_df_encoded[col] = np.nanmean(values) if len(values) > 0 else current_df[target_col].mean()
                            elif stat == 'std':
                                next_df_encoded[col] = np.nanstd(values) if len(values) > 1 else current_df[target_col].std()
                            else:
                                next_df_encoded[col] = current_df[target_col].median()
                        
                        # Handle enhanced features
                        elif 'daily_change' in col:
                            if len(forecasts) > 0:
                                next_df_encoded[col] = forecasts[-1] - current_df[target_col].iloc[-1]
                            else:
                                next_df_encoded[col] = 0
                        
                        elif 'Pollution_Momentum' in col:
                            if len(forecasts) > 0 and 'Traffic_Dispersion' in next_df_encoded.columns:
                                daily_change = forecasts[-1] - current_df[target_col].iloc[-1]
                                next_df_encoded[col] = daily_change * next_df_encoded['Traffic_Dispersion'].iloc[0]
                            else:
                                next_df_encoded[col] = 0
                        
                        elif 'Dispersion_Effectiveness' in col:
                            if len(forecasts) > 0 and 'Traffic_Dispersion' in next_df_encoded.columns:
                                next_df_encoded[col] = forecasts[-1] / (next_df_encoded['Traffic_Dispersion'].iloc[0] + 1)
                            else:
                                next_df_encoded[col] = current_df[target_col].iloc[-1] if target_col in current_df.columns else 0
                        
                        else:
                            # For any other missing features, use default values
                            next_df_encoded[col] = 0
                
                # Final NaN cleanup
                for col in feature_cols:
                    if col in next_df_encoded.columns:
                        if pd.isna(next_df_encoded[col].iloc[0]):
                            if any(x in col for x in ['std', 'volatility']):
                                next_df_encoded[col] = 0
                            elif 'percentile' in col:
                                next_df_encoded[col] = 50.0
                            else:
                                next_df_encoded[col] = 0
                
                return next_df_encoded

            def generate_ml_forecasts_categorical(model, test_data, feature_cols, target_col, days, scaler, encoders):
                """Generate forecasts for ML models with categorical features - FIXED for categorical errors"""
                current_df = test_data.copy()
                forecasts = []
                last_date = test_data['Date'].iloc[-1]
                
                for i in range(days):
                    next_date = last_date + timedelta(days=i+1)
                    
                    try:
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
                        
                        # Estimate other features for next row
                        for feature in ['Temp', 'WS', 'WD', 'NO2', 'NO', 'TrafficV', 'Total_Pedestrians', 'City_Centre_TVCount']:
                            if feature in current_df.columns:
                                next_row_basic[feature] = current_df[feature].iloc[-1]
                        
                        current_df = pd.concat([current_df, pd.DataFrame([next_row_basic])], ignore_index=True)
                        
                    except Exception as e:
                        st.warning(f"Warning during forecast generation for day {i+1}: {str(e)}")
                        # Use last forecast value or median as fallback
                        if forecasts:
                            forecasts.append(forecasts[-1])
                        else:
                            forecasts.append(current_df[target_col].median())
                
                return forecasts

            def forecast_future_enhanced_categorical(model, model_type, test_data, feature_cols, target_col, days=3, 
                                            scaler=None, encoders=None):
                """Enhanced forecasting with categorical features"""
                days = min(days, 3)
                
                last_date = pd.Timestamp(test_data['Date'].iloc[-1])
                future_dates = [last_date + timedelta(days=i+1) for i in range(days)]
                
                if model_type in ['Random Forest', 'Decision Tree'] and hasattr(model, 'estimators_' if model_type == 'Random Forest' else 'tree_'):
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
                        
                        if model_type == 'Random Forest':
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
                
                if model_type in ['arima']:
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
            #
            def plot_feature_importance_categorical(model, feature_names, model_name):
                """Plot feature importance with comprehensive feature grouping - FIXED"""
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    
                    importance_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': importances
                    })
                    
                    # IMPROVED GROUPING LOGIC (same as before)
                    def categorize_feature(feature_name):
                        """Comprehensive feature categorization"""
                        feature = str(feature_name).lower()
                        
                        # Categorical date features
                        categorical_prefixes = ['month_', 'dayofweek_', 'season_', 'weekend_', 'quarter_', 'monthpart_', 'wind_sector_']
                        for prefix in categorical_prefixes:
                            if feature.startswith(prefix):
                                return prefix.replace('_', '').title()
                        
                        # PM10 related features
                        if 'pm10' in feature:
                            if 'lag' in feature:
                                return 'PM10 Lags'
                            elif 'rolling' in feature or 'ewm' in feature:
                                return 'PM10 Rolling Stats'
                            elif 'change' in feature:
                                return 'PM10 Changes'
                            else:
                                return 'PM10 Base'
                        
                        # Traffic and activity features
                        if any(x in feature for x in ['traffic', 'pedestrian', 'city_centre', 'activity']):
                            if 'lag' in feature or 'rolling' in feature:
                                return 'Traffic/Activity Lags'
                            else:
                                return 'Traffic/Activity'
                        
                        # Weather features
                        if any(x in feature for x in ['temp', 'ws', 'wd', 'wind']):
                            return 'Weather'
                        
                        # Chemical features
                        if any(x in feature for x in ['no2', 'no', 'nox']):
                            return 'Chemical'
                        
                        # Composite environmental features
                        if any(x in feature for x in ['dispersion', 'atmospheric', 'stability', 'consistency']):
                            return 'Environmental Composite'
                        
                        # Enhanced features
                        if any(x in feature for x in ['momentum', 'effectiveness', 'interaction']):
                            return 'Enhanced Features'
                        
                        # Base year feature
                        if feature == 'year':
                            return 'Temporal Base'
                        
                        # Lag features (catch remaining)
                        if 'lag' in feature:
                            return 'Other Lags'
                        
                        # Rolling features (catch remaining)
                        if 'rolling' in feature or 'ewm' in feature:
                            return 'Other Rolling Stats'
                        
                        return 'Other'
                    
                    # Apply improved grouping
                    importance_df['Feature_Group'] = importance_df['Feature'].apply(categorize_feature)
                    
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
                    
                    # FIXED: Group importance by category - separate aggregations
                    grouped_data = importance_df.groupby('Feature_Group')
                    
                    grouped_importance = pd.DataFrame({
                        'Total_Importance': grouped_data['Importance'].sum(),
                        'Count': grouped_data['Importance'].count(),
                        'Max_Individual': grouped_data['Importance'].max()
                    }).sort_values('Total_Importance', ascending=False)
                    
                    # Create grouped figure with more information
                    fig_grouped = go.Figure()
                    
                    fig_grouped.add_trace(go.Bar(
                        x=grouped_importance['Total_Importance'],
                        y=grouped_importance.index,
                        orientation='h',
                        name='Total Importance',
                        text=[f"Count: {count}, Max: {max_val:.3f}" 
                            for count, max_val in zip(grouped_importance['Count'], grouped_importance['Max_Individual'])],
                        textposition='auto',
                    ))
                    
                    fig_grouped.update_layout(
                        title=f'Feature Group Importance for {model_name}',
                        xaxis_title='Total Importance',
                        yaxis_title='Feature Group',
                        height=600,
                        showlegend=False
                    )
                    
                    # Create summary dataframe for display
                    summary_df = grouped_importance.reset_index()
                    summary_df.columns = ['Feature Group', 'Total Importance', 'Feature Count', 'Max Individual Importance']
                    summary_df = summary_df.round(3)
                    
                    return fig, fig_grouped, importance_df, summary_df
                
                return None, None, None, None

            def plot_forecast_with_intervals(historical_data, forecast_data, target_col, model_name):
                """Plot forecast with prediction intervals - FIXED"""
                fig = go.Figure()
                
                # Plot historical data
                fig.add_trace(go.Scatter(
                    x=historical_data['Date'],
                    y=historical_data[target_col],
                    mode='lines',
                    name='Historical Data',
                    line=dict(color='blue', width=1)
                ))
                
                # Plot forecast
                fig.add_trace(go.Scatter(
                    x=forecast_data['Date'],
                    y=forecast_data[target_col],
                    mode='lines+markers',
                    name='Forecast',
                    line=dict(color='red', width=2),
                    marker=dict(size=8)
                ))
                
                # Add prediction intervals if available
                if f'{target_col}_lower' in forecast_data.columns and f'{target_col}_upper' in forecast_data.columns:
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
                
                # Add forecast start line
                fig.add_shape(
                    type="line",
                    x0=historical_data['Date'].iloc[-1],
                    y0=0,
                    x1=historical_data['Date'].iloc[-1],
                    y1=max(
                        historical_data[target_col].max() if not historical_data.empty else 0,
                        forecast_data[target_col].max() if not forecast_data.empty else 0
                    ) * 1.1,
                    line=dict(color="black", width=1, dash="dash")
                )
                
                fig.add_annotation(
                    x=historical_data['Date'].iloc[-1],
                    y=max(
                        historical_data[target_col].max() if not historical_data.empty else 0,
                        forecast_data[target_col].max() if not forecast_data.empty else 0
                    ) * 1.05,
                    text="Forecast Start",
                    showarrow=False
                )
                
                fig.update_layout(
                    title=f'{model_name} Forecast for {target_col}',
                    xaxis_title='Date',
                    yaxis_title=f'{target_col} (µg/m³)',
                    height=500,
                    hovermode='x unified'
                )
                
                return fig

            def create_performance_summary_table(all_results):
                """Create summary table of all model performances - FIXED for TypeError"""
                summary_data = []
                
                for model_type, results in all_results.items():
                    if 'test_results' in results:
                        test_res = results['test_results']
                        train_res = results['train_results']
                        
                        # Fix: Ensure proper formatting of numeric values
                        row_data = {
                            'Model': model_type.upper(),
                            'RMSE': f"{float(test_res['RMSE']):.2f}",
                            'MAE': f"{float(test_res['MAE']):.2f}",
                            'R²': f"{float(test_res['R²']):.3f}",
                            'MAPE': f"{float(test_res['MAPE']):.1f}%",
                            'Direction Acc.': f"{float(test_res['Directional_Accuracy']):.1f}%",
                            'Overall Score': f"{float(test_res.get('Overall_Score', 0)):.1f}"
                        }
                        
                        if 'CV_MSE_Mean' in train_res:
                            row_data['CV MSE'] = f"{float(train_res['CV_MSE_Mean']):.2f} ± {float(train_res['CV_MSE_Std']):.2f}"
                        
                        summary_data.append(row_data)
                
                summary_df = pd.DataFrame(summary_data)
                if not summary_df.empty:
                    # Extract numeric values for sorting
                    summary_df['_sort_score'] = summary_df['Overall Score'].str.replace(r'[^0-9.-]', '', regex=True).astype(float)
                    summary_df = summary_df.sort_values('_sort_score', ascending=False)
                    summary_df = summary_df.drop('_sort_score', axis=1)
                
                return summary_df

            # Main forecast model enhanced function with ALL fixes
            def run_forecast_model_enhanced(model_type, train_data, val_data, test_data, target_col='PM10', 
                                    feature_cols=None, forecast_days=3, tune_hyperparams=True,
                                    handle_outliers_flag=True, outlier_method='cap',
                                    n_folds=3, auto_cv=True, progress_bar=None, status_text=None, **kwargs): # ADD auto_cv=True parameter, PROGRESS BAR
                """Enhanced forecast model with ALL fixes and Optuna optimization"""
                results = {}
                
                forecast_days = min(forecast_days, 3)
                
                # Update progress
                if progress_bar and status_text:
                    status_text.text('Checking for missing values...')
                    progress_bar.progress(35)

                # IMMEDIATE MISSING VALUE CHECK
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

                    if progress_bar and status_text:
                        status_text.text('Detecting and handling outliers...')
                        progress_bar.progress(40)

                    train_outliers = detect_outliers(train_data, target_col, method='iqr')
                    val_outliers = detect_outliers(val_data, target_col, method='iqr')
                    test_outliers = detect_outliers(test_data, target_col, method='iqr')
                    
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
                
                if model_type in ['arima', 'prophet']:
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
                    elif model_type == 'prophet':
                        model, train_results, val_results, _, _, best_params = train_prophet(train_data, val_data, tune_params=tune_hyperparams)
                        prophet_test = test_data[['Date', target_col]].rename(columns={'Date': 'ds', target_col: 'y'})
                        test_forecast = model.predict(prophet_test[['ds']])
                        y_test_pred, y_test = test_forecast['yhat'].values, prophet_test['y'].values
                        scaler_use = None
                        test_data_use = test_data
                else:
                    # Update progress
                    if progress_bar and status_text:
                        status_text.text('Engineering features and creating lag variables...')
                        progress_bar.progress(45)

                    # Add lag features with ULTRA-ROBUST cleaning
                    train_with_lags = add_lag_features(train_data, target_col, show_message=False)
                    val_with_lags = add_lag_features(val_data, target_col, show_message=False)
                    test_with_lags = add_lag_features(test_data, target_col, show_message=False)

                    # Update progress
                    if progress_bar and status_text:
                        status_text.text('Adding enhanced features...')
                        progress_bar.progress(50)

                    # Add enhanced domain features
                    train_with_lags = add_enhanced_features(train_with_lags, target_col)
                    val_with_lags = add_enhanced_features(val_with_lags, target_col)
                    test_with_lags = add_enhanced_features(test_with_lags, target_col)

                    # Update progress
                    if progress_bar and status_text:
                        status_text.text('Creating feature arrays and encoding...')
                        progress_bar.progress(55)
                    
                    lag_features = [col for col in train_with_lags.columns if any(x in col for x in ['_lag_', '_rolling_', '_change', 'Pollution_Momentum', 'Dispersion_Effectiveness'])]
                    
                    # Create feature arrays with one-hot encoding
                    X_train, y_train, feature_names, encoders = create_feature_target_arrays_with_encoding(train_with_lags, target_col, categorical_cols, base_features, lag_features)
                    X_val, y_val, _, _ = create_feature_target_arrays_with_encoding(val_with_lags, target_col, categorical_cols, base_features, lag_features, encoders)
                    X_test, y_test, _, _ = create_feature_target_arrays_with_encoding(test_with_lags, target_col, categorical_cols, base_features, lag_features, encoders)
                    
                    results['encoders'], results['feature_names'] = encoders, feature_names
                    scaler_use = None
                    
                    # Training logic for different model types
                    if progress_bar and status_text:
                        if tune_hyperparams:
                            status_text.text(f'Tuning hyperparameters for {model_type.upper()}...')
                        else:
                            status_text.text(f'Training {model_type.upper()} with default parameters...')
                        progress_bar.progress(60)

                    # Training logic for different model types
                    if model_type == 'Decision Tree':
                        model, train_results, val_results, best_params = train_decision_tree(X_train, y_train, X_val, y_val, tune_params=tune_hyperparams, n_folds=n_folds, auto_cv=auto_cv) # Added auto_cv
                        y_test_pred = model.predict(X_test)
                        test_data_use = test_with_lags
                    elif model_type == 'Random Forest':
                        model, train_results, val_results, best_params = train_random_forest(X_train, y_train, X_val, y_val, tune_params=tune_hyperparams, n_folds=n_folds, auto_cv=auto_cv) # Added auto_cv)
                        y_test_pred = model.predict(X_test) 
                        test_data_use = test_with_lags
                    elif model_type == 'XGBoost':
                        model, train_results, val_results, best_params = train_xgboost(X_train, y_train, X_val, y_val, tune_params=tune_hyperparams, n_folds=n_folds, auto_cv=auto_cv) # Added auto_cv
                        y_test_pred = model.predict(X_test)
                        test_data_use = test_with_lags
                    elif model_type == 'knn':
                        model, train_results, val_results, best_params, scaler_use = train_knn(X_train, y_train, X_val, y_val, tune_params=tune_hyperparams, n_folds=n_folds, auto_cv=auto_cv) # Added auto_cv
                        _, _, X_test_scaled, y_test, _ = prepare_data_for_distance_models(X_train, X_val, X_test, y_train)
                        y_test_pred = model.predict(X_test_scaled)
                        scaling_info['feature_scaling_applied'] = True
                        scaling_info['feature_scaling_method'] = "RobustScaler (median=0, IQR=1)"
                        test_data_use = test_with_lags
                    elif model_type == 'svr':
                        model, train_results, val_results, best_params, scaler_use = train_svr(X_train, y_train, X_val, y_val, tune_params=tune_hyperparams, n_folds=n_folds, auto_cv=auto_cv) # Added auto_cv
                        _, _, X_test_scaled, y_test, _ = prepare_data_for_distance_models(X_train, X_val, X_test, y_train)
                        y_test_pred = model.predict(X_test_scaled)
                        scaling_info['feature_scaling_applied'] = True
                        scaling_info['feature_scaling_method'] = "RobustScaler (median=0, IQR=1)"
                        test_data_use = test_with_lags
                    elif model_type == 'linear':
                        model, train_results, val_results, best_params = train_linear_regression(X_train, y_train, X_val, y_val, tune_params=tune_hyperparams, n_folds=n_folds, auto_cv=auto_cv) # Added auto_cv
                        y_test_pred = model.predict(X_test)
                        test_data_use = test_with_lags
                    elif model_type == 'ridge':
                        model, train_results, val_results, best_params = train_ridge(X_train, y_train, X_val, y_val, tune_params=tune_hyperparams, n_folds=n_folds, auto_cv=auto_cv) # Added auto_cv
                        y_test_pred = model.predict(X_test)
                        test_data_use = test_with_lags
                    elif model_type == 'lasso':
                        model, train_results, val_results, best_params = train_lasso(X_train, y_train, X_val, y_val, tune_params=tune_hyperparams, n_folds=n_folds, auto_cv=auto_cv) # Added auto_cv
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

                # Update progress before evaluation
                if progress_bar and status_text:
                    status_text.text('Evaluating model performance...')
                    progress_bar.progress(80)

                test_results = evaluate_model(y_test, y_test_pred, f"{model_type.upper()} (Test)")
                test_results['Overall_Score'] = calculate_overall_score(test_results)
                
                    # Update progress before forecasting
                if progress_bar and status_text:
                    status_text.text(f'Generating {forecast_days}-day forecast...')
                    progress_bar.progress(90)

                if model_type not in ['arima', 'prophet']:
                    forecast_df = forecast_future_enhanced_categorical(model, model_type, test_data_use, feature_names, target_col, days=forecast_days, scaler=scaler_use, encoders=encoders)
                else:
                    forecast_df = forecast_future_enhanced(model, model_type, test_data, feature_cols, target_col, days=forecast_days)
                
                results.update({'model': model, 'test_pred': y_test_pred, 'train_results': train_results, 'val_results': val_results, 'test_results': test_results, 'best_params': best_params, 'forecast': forecast_df, 'model_type': model_type, 'scaling_info': scaling_info})
                
                if model_type in ['Random Forest', 'XGBoost', 'Decision Tree']:
                    importance_results = plot_feature_importance_categorical(model, feature_names, model_type.upper())
                    if importance_results[0] is not None:  # Check if results exist
                        results['importance_fig'] = importance_results[0]
                        results['importance_fig_grouped'] = importance_results[1] 
                        results['importance_df'] = importance_results[2]
                        results['importance_summary'] = importance_results[3]
                    # importance_fig, importance_fig_grouped, importance_df = plot_feature_importance_categorical(model, feature_names, model_type.upper())
                    # results['importance_fig'], results['importance_fig_grouped'], results['importance_df'] = importance_fig, importance_fig_grouped, importance_df
                    # importance_fig, importance_fig_grouped, importance_df, importance_summary = plot_feature_importance_categorical(model, feature_names, model_type.upper())
                    # results['importance_fig'] = importance_fig
                    # results['importance_fig_grouped'] = importance_fig_grouped
                    # results['importance_df'] = importance_df
                    # results['importance_summary'] = importance_summary
                
                return results

            def data_explorer_split_section():
                """Data split configuration section for Data Explorer page"""
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


            def show_how_to_use_guide():
                """Display comprehensive how-to-use guide for the PM10 forecasting tool"""
                
                with st.expander("📚 How to Use This PM10 Data Modelling & Forecasting Tool", expanded=False):
                    st.markdown("""
                    ## 🚀 Quick Start Guide
                                     
                    ### **Step 1: Explore Your Data** 📊
                    Navigate to **Data Explorer** to:
                    - View data quality overview and missing value compatibility
                    - Understand composite features automatically created
                    - Configure train/validation/test split ratios
                    - Visualize time series and correlation patterns
                    - Check statistical summaries
                    
                    ### **Step 2: Analyze Outliers** 🔍
                    Use **Outlier Analysis** to:
                    - Detect anomalous PM10 values using IQR method
                    - Visualize outliers in time series context
                    - Apply outlier handling (capping method)
                    - Compare before/after outlier treatment
                    
                    ### **Step 3: Train Models** ⛳
                    In **Model Training**:
                    - Choose from the different algorithms 
                    - Enable hyperparameter tuning for better performance
                    - Configure cross-validation and forecast horizon (1-3 days)
                    
                    ### **Step 4: Compare Results** 📈
                    Use **Model Comparison** to:
                    - View performance summary table sorted by overall score
                    - Compare error metrics (RMSE, MAE, R², MAPE, Directional Accuracy)
                    - Visualize forecast comparisons across multiple models
                    - Identify the best-performing model
                    
                    ### **Step 5: Generate Dashboard** 📊
                    **Forecasting Dashboard** provides:
                    - Executive summary with best model recommendations
                    - Interactive forecast visualizations
                    - Downloadable results (forecasts, metrics, parameters)
                    - Key insights and recommendations
                                        
                    ## 📊 Interpreting Results
                    
                    ### **Performance Metrics**
                    - **RMSE**: Lower is better (root mean square error)
                    - **MAE**: Lower is better (mean absolute error)
                    - **R²**: Higher is better (0-1, proportion of variance explained)
                    - **MAPE**: Lower is better (mean absolute percentage error)
                    - **Overall Score**: Combined metric (0-100, higher is better)
                    
                    ### **Forecast Reliability**
                    - **1-day forecasts**: Most reliable
                    - **2-3 day forecasts**: Good for planning
                    - **Beyond 3 days**: Limited by model constraints
                    
                    ### **Feature Importance**
                    - Shows which variables most influence PM10 predictions
                    - Composite features often rank highly
                    - Temporal features (day/month/season) reveal patterns
                    
                    ---
                    
                    ## 🔧 Advanced Settings
                    
                    ### **Data Split Configuration**
                    - **Conservative (60/20/20)**: More robust evaluation
                    - **Standard (70/15/15)**: Balanced approach
                    - **Training Heavy (80/10/10)**: Maximum training data
                    
                    ### **Hyperparameter Tuning**
                    - Enable for better performance (takes longer)
                    - Uses cross-validation for robust evaluation
                    - Neural networks use Optuna optimization
                    
                    ### **Outlier Handling**
                    - Capping method preserves data while reducing extreme values
                    - Recommended for most air quality datasets
                    
                    ---
                    
                    ## 🚨 Troubleshooting
                    
                    ### **Common Issues**
                    - **"Missing values detected"**: Use tree-based models or clean data
                    - **"Not enough data"**: Need minimum 30 days, preferably 6+ months
                    - **Poor performance**: Try different models, check data quality
                    - **Neural network errors**: Ensure no missing values, adequate data size
                    
                    ### **Performance Tips**
                    - Start with Random Forest (handles most issues)
                    - Enable outlier handling for noisy data
                    
                    ---
                    
                    ## 📈 Best Practices
                    
                    1. **Start Simple**: Begin with Random Forest or XGBoost
                    2. **Check Data Quality**: Review missing values and outliers
                    3. **Use Cross-Validation**: Enable hyperparameter tuning
                    4. **Compare Multiple Models**: Different models capture different patterns
                    5. **Validate Results**: Check forecast reasonableness against domain knowledge
                    6. **Regular Updates**: Retrain models with new data periodically
                    
                    ---
                    
                    ## 📞 Need Help?
                    
                    This tool provides comprehensive PM10 forecasting with automatic feature engineering and model optimization. 
                    The composite features are designed based on atmospheric science principles to improve prediction accuracy.
                    
                    **Remember**: The tool is designed for short-term forecasting (1-3 days) and works best with clean, 
                    comprehensive environmental datasets including meteorological and traffic data.
                    """)

            # ============ SYSTEM INFO DISPLAY FUNCTION ============

            def display_core_allocation_info():
                """Display core allocation information to users"""
                st.sidebar.markdown("---")
                st.sidebar.subheader("🔧 System Optimization")
                
                core_info = get_optimal_core_allocation()
                
                with st.sidebar.expander("Core Allocation Details", expanded=False):
                    st.write(f"**System Type:** {core_info['detected_system'].title()}")
                    st.write(f"**Total CPU Cores:** {core_info['total_cores']}")
                    st.write("")
                    st.write("**Optimized Allocation:**")
                    st.write(f"• Hyperparameter Search: {core_info['search_n_jobs']} cores")
                    st.write(f"• Cross-Validation: {core_info['cv_n_jobs']} cores")
                    st.write(f"• Tree Models: {core_info['tree_n_jobs']} cores")
                    st.write(f"• Ensemble Models: {core_info['ensemble_n_jobs']} cores")
                    st.write("")
                    st.success("✅ Smart allocation prevents resource conflicts!")

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
                    st.session_state.train_ratio = 0.60
                    st.session_state.val_ratio = 0.20
                    st.session_state.test_ratio = 0.20
                    st.session_state.training_confirmed = False
                
                # ADD THE HOW-TO-USE GUIDE AT THE TOP OF EVERY PAGE
                show_how_to_use_guide()
                
                if uploaded_file is not None:
                    with st.spinner("Loading data..."):
                        df = load_data()
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
                        # Enhanced Feature Information Panel
                        st.header("📋 Enhanced Feature Information")
                        
                        with st.expander("🧠 **Composite Features Added**", expanded=True):
                            st.write("**The following composite features are automatically created to enhance PM10 forecasting:**")
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write("**🌬️ Traffic-Weather Interactions:**")
                                st.write("- **Traffic_Dispersion** = TrafficV / (WindSpeed + 1)")
                                st.write("  *Higher values indicate poor pollutant dispersion*")
                                st.write("- **Traffic_Temp_Index** = TrafficV × Temperature_normalized")
                                st.write("  *Captures temperature effects on emissions*")
                                
                                st.write("**🌪️ Wind Effectiveness:**")
                                st.write("- **Wind_Consistency** = √(wind_x² + wind_y²)")
                                st.write("  *Effective wind for pollution dispersion*")
                                st.write("- **Wind_Sector** = North/East/South/West")
                                st.write("  *Categorical wind direction sectors (if wind data available)*")
                                
                                st.write("**📈 Enhanced Features:**")
                                st.write("- **Pollution_Momentum** = daily_change × Traffic_Dispersion")
                                st.write("- **Dispersion_Effectiveness** = PM10 / (Traffic_Dispersion + 1)")
                            
                            with col2:
                                st.write("**🚶 Human Activity Index:**")
                                st.write("- **Activity_Index** = 0.2×Pedestrians + 0.3×CityTV + 0.5×Traffic")
                                st.write("  *Weighted combination of all human activity*")
                                
                                st.write("**🌡️ Atmospheric Stability:**")
                                st.write("- **Atmospheric_Stability** = Temperature / (WindSpeed + 0.1)")
                                st.write("  *Higher values = more stable conditions = worse mixing*")
                                
                                st.write("**⚗️ Chemical Ratios:**")
                                st.write("- **NO2_NO_Ratio** = NO2 / (NO + 1)")
                                st.write("  *Indicates photochemical activity*")
                                st.write("- **Traffic_Emission_Factor** = TrafficV × exp(Temp/50)")
                                st.write("  *Temperature-dependent emission efficiency*")
                                
                                st.write("**📊 Lag Features:**")
                                st.write("- **PM10 lags:** 1, 3, 7 days")
                                st.write("- **Composite feature lags:** 1, 3 days")
                                st.write("- **Rolling statistics:** 7-day mean & std")
                        
                        st.info("💡 **Why These Features?** These composite features capture the **physical relationships** between environmental variables that affect PM10 formation, transport, and dispersion, making predictions more scientifically grounded and less prone to overfitting.")
                        
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
                        else:
                            st.success("✅ **No missing values detected** - All models can be trained!")

                        st.header("Data Overview")
                        st.subheader("Daily Aggregated Data")
                        st.dataframe(st.session_state.daily_data.head(10))
                        
                        # Show excluded columns
                        st.subheader("📋 Data Processing Summary")
                        excluded_cols = ['AQI', 'PM2.5', 'NOx']
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.success("**✅ Columns Used for Training:**")
                            used_cols = [col for col in st.session_state.daily_data.columns if col not in ['Date'] + get_categorical_features()]
                            for col in used_cols:
                                st.write(f"- {col}")
                        
                        with col2:
                            st.info("**ℹ️ Columns Excluded (prevent data leakage):**")
                            for col in excluded_cols:
                                st.write(f"- {col}")
                            st.write("*These are excluded to prevent data leakage and improve model generalization*")

                        # ===== DATA SPLIT CONFIGURATION =====
                        data_explorer_split_section()
                        
                        # Categorical Features Section
                        st.header("🏷️ Categorical Date Features")
                        categorical_features = get_categorical_features()
                        # Only show features that exist in the data
                        available_categorical = [col for col in categorical_features if col in st.session_state.daily_data.columns]
                        if 'Wind_Sector' in st.session_state.daily_data.columns:
                            available_categorical.append('Wind_Sector')
                        
                        if available_categorical:
                            sample_data = st.session_state.daily_data[available_categorical].head(10)
                            st.dataframe(sample_data)
                            
                            st.info("**Note:** These categorical features will be one-hot encoded for machine learning models, " +
                                "providing better representation of temporal patterns compared to numerical encoding.")
                            
                            # Show unique values for each categorical feature
                            with st.expander("View Unique Values in Categorical Features"):
                                for feature in available_categorical:
                                    if feature in st.session_state.daily_data.columns:
                                        unique_vals = st.session_state.daily_data[feature].unique()
                                        st.write(f"**{feature}:** {', '.join(map(str, unique_vals))}")
                        else:
                            st.warning("No categorical features found in the dataset.")
                        
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
                    st.title("⛳ Model Training with Feature Engineering and K-fold Validation")
                    
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
                        with st.expander("🏗️ Composite Feature Engineering Details", expanded=False):
                            st.write("**Composite Features Created (One-Hot Encoded):**")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write("**🌬️ Environmental Interactions:**")
                                st.write("- Traffic_Dispersion (traffic/wind relationship)")
                                st.write("- Wind_Consistency (effective dispersion)")
                                st.write("- Atmospheric_Stability (mixing conditions)")
                                st.write("- Traffic_Emission_Factor (temperature effects)")
                                st.write("**📊 Activity & Chemical Features:**")
                                st.write("- Activity_Index (human activity composite)")
                                st.write("- NO2_NO_Ratio (photochemical activity)")
                                st.write("- Pollution_Momentum (change dynamics)")
                                st.write("- Dispersion_Effectiveness (relative performance)")
                            
                            with col2:                               
                                st.write("**🏷️ Categorical Date Features (One-Hot Encoded):**")
                                st.write("- **Month**: January through December")
                                st.write("- **DayOfWeek**: Monday through Sunday")
                                st.write("- **Season**: Winter/Spring/Summer/Autumn")
                                st.write("- **Weekend**: Weekend vs Weekday")
                                st.write("- **Quarter**: Q1/Q2/Q3/Q4")
                                st.write("- **MonthPart**: Early/Mid/Late month")
                                st.write("- **Wind_Sector**: North/East/South/West/Calm")

                        
                        st.sidebar.title("Model Settings")
                        model_type = st.sidebar.selectbox(
                            "Select Model", 
                            ["Decision Tree",  "Random Forest", "XGBoost" ] #"knn", "svr", "lstm", "ann", "gru", "arima", "prophet", "linear", "ridge", "lasso"                        
                        )
                        
                        # Show model-specific information
                        missing_info = check_missing_values(st.session_state.daily_data)
                        can_proceed, compatibility_message = check_model_missing_value_compatibility(model_type, missing_info)
                        
                        if model_type in ['Decision Tree', 'Random Forest', 'XGBoost']:
                            st.sidebar.success("🌳 **Tree-based model**: Excellent for handling composite features and capturing non-linear patterns !") #missing values
                            if missing_info:
                                st.sidebar.info(f"✅ {compatibility_message}")
                        elif model_type in ['knn', 'svr']:
                            st.sidebar.info("📏 **Distance-based model**: Benefits from composite features and feature scaling!")
                            if missing_info:
                                st.sidebar.warning(f"⚠️ {compatibility_message}")
                        elif model_type in ['lstm', 'ann', 'gru']:
                            st.sidebar.info("🧠 **Neural Network**: Requires feature scaling and uses Optuna optimization!")
                            if missing_info:
                                st.sidebar.error(f"❌ {compatibility_message}")
                        elif model_type in ['arima', 'prophet']:
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
                            # Add this checkbox
                            auto_cv = st.checkbox("Auto-adapt CV folds", value=True, 
                                                help="Automatically adjust CV folds based on dataset size")

                            if handle_outliers_flag:
                                outlier_method = st.selectbox("Outlier Handling Method", ["cap"])
                            else:
                                outlier_method = None
                            
                            # This slider is now only used if auto_cv is False
                            n_folds = st.slider("K-Fold Cross-Validation Folds", 3, 5, 3,
                                        disabled=auto_cv)  # Disable when auto_cv is True
                            # n_folds = st.slider("K-Fold Cross-Validation Folds", 3, 5, 3)
                            forecast_days = st.slider("Forecast Days", 1, 3, 3)
                            
                            # Special settings for neural networks
                            if model_type in get_neural_network_models():
                                st.write("**🔧 Neural Network & Optuna Settings:**")
                                use_optuna = st.checkbox("Use Optuna Optimization", value=True, 
                                                    help="Use Optuna for hyperparameter optimization (recommended)")
                                
                                if model_type in ['lstm', 'gru']:
                                    time_steps = st.slider("Time Steps (Sequence Length)", 3, 14, 7)
                                    st.info(f"Using {time_steps} previous days to predict next day")
                                
                                if use_optuna:
                                    st.success(f"✅ Optuna will optimize with 10 trials")
                                else:
                                    st.info("ℹ️ Using default hyperparameters (faster but may be suboptimal)")
                        
                        # Training button and logic
                        if st.sidebar.button("⛳ Train Model"):
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
                            
                            # CREATE PROGRESS BAR
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            # Update progress for different stages
                            status_text.text(f'Initializing {model_type.upper()} training...')
                            progress_bar.progress(10)

                            # Proceed with training
                            with st.spinner(f"Training {model_type.upper()} model with {n_folds}-fold CV..."):
                                try:
                                    # Update progress
                                    status_text.text('Preparing features and data...')
                                    progress_bar.progress(20)

                                    kwargs = {}
                                    if model_type in ['lstm', 'gru']:
                                        kwargs['time_steps'] = time_steps
                                        kwargs['use_optuna'] = use_optuna
                                    elif model_type == 'ann':
                                        kwargs['use_optuna'] = use_optuna
                                    
                                    # Update progress before main training
                                    status_text.text(f'Training {model_type.upper()} model...')
                                    progress_bar.progress(30)

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
                                        auto_cv=auto_cv,  # ADDED THIS LINE
                                        progress_bar=progress_bar,  # ADD PROGRESS BAR
                                        status_text=status_text,     # ADD PROGRESS BAR
                                        **kwargs
                                    )
                                    
                                    # Final progress update
                                    if 'error' not in results:
                                        progress_bar.progress(100)
                                        status_text.text('✅ Training complete!')

                                    # Handle missing value errors
                                    if 'error' in results and results['error'] == 'missing_values_incompatible':
                                        st.error(f"❌ **{model_type.upper()} Model Training Failed**")
                                        st.write(results['message'])
                                        missing_info = results['missing_info']['all']
                                        display_missing_value_error(missing_info, [model_type])
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
                                    st.error(f"❌ **Unexpected Error in {model_type.upper()} Training:**")
                                    st.code(str(e))
                                    st.write("**Suggestions:**")
                                    st.write("- Check data quality and format")
                                    st.write("- Try a different model type")
                                    st.write("- Reduce dataset size if memory issues")

                        display_core_allocation_info()

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
                                    st.success("✅ **Composite Features Applied**")
                                    st.write("**Features:** Traffic-Weather, Wind, Activity, Atmospheric, Chemical")
                                
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
                                st.markdown("**Individual Features** show single feature importance")
                                
                                tab1, tab2, tab3 = st.tabs(["Individual Features", "Feature Groups", "Summary Table"])
    
                                with tab1:
                                    st.plotly_chart(results['importance_fig'], use_container_width=True)
                                
                                with tab2:
                                    if 'importance_fig_grouped' in results:
                                        st.plotly_chart(results['importance_fig_grouped'], use_container_width=True)
                                
                                with tab3:
                                    if 'importance_summary' in results:
                                        st.dataframe(results['importance_summary'], use_container_width=True)
                                        st.info("💡 **Total Importance** shows the sum of all features in that group.")
                            
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
                            st.success(f"🏆 Best Model: {best_model} with Overall Score: {best_score}")
                        
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
                                title='Model Forecast Comparison with Composite Features',
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
                                st.metric("Best Model", f"🏆{best_model}")
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

                        # Get available models
                        available_models = list(st.session_state.model_results.keys())

                        # Determine the best model index
                        default_index = 0  # Default to first model if no match found
                        
                        if not summary_df.empty and available_models:
                            # Get the best model name from summary (it's in uppercase like 'XGBOOST')
                            best_model_upper = summary_df.iloc[0]['Model']
                            
                            # Find the matching key in model_results
                            for idx, model_key in enumerate(available_models):
                                if model_key.upper() == best_model_upper:
                                    default_index = idx
                                    break

                        download_model = st.selectbox(
                            "Select Model for Download",
                            available_models,
                            index=default_index #Use the determined index
                            #available_models.index(best_model_for_download) if best_model_for_download and best_model_for_download in available_models else 0
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

                        # Get available models from session state
                        available_models = list(st.session_state.model_results.keys())

                        # Determine default model safely
                        default_models = []
                        if not summary_df.empty and available_models:
                            # Get the best model name from summary (it's in uppercase like 'XGBOOST')
                            best_model_upper = summary_df.iloc[0]['Model']
                            
                            # Find the matching key in model_results (which might be lowercase)
                            for model_key in available_models:
                                if model_key.upper() == best_model_upper:
                                    default_models = [model_key]
                                    break
                            
                            # If no match found, use the first available model
                            if not default_models and available_models:
                                default_models = [available_models[0]]

                        viz_models = st.multiselect(
                            "Select Models to Visualize",
                            available_models,
                            default=default_models  # Use the safely determined default
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
                                title="PM10 Forecast Comparison (1-3 Days) with Composite Features & Optuna",
                                xaxis_title="Date",
                                yaxis_title="PM10 (µg/m³)",
                                height=600,
                                hovermode='x unified'
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Enhanced Composite Feature Insights
                        st.header("🏷️ Composite Feature Insights")
                        
                        if viz_models:
                            # Collect insights from tree-based models
                            # composite_insights = {}
                            # UPDATED VERSION - Extract insights from tree-based models
                            composite_insights = {}
                            for model_name in viz_models:
                                if model_name in st.session_state.model_results:
                                    results = st.session_state.model_results[model_name]
                                    if 'importance_df' in results and results['importance_df'] is not None:
                                        imp_df = results['importance_df']
                                        
                                        insights = {}
                                        
                                        # Find the SINGLE most important feature (consistent with Model Training)
                                        most_important_feature = imp_df.loc[imp_df['Importance'].idxmax(), 'Feature']
                                        most_important_value = imp_df['Importance'].max()
                                        insights['top_individual_feature'] = {
                                            'name': most_important_feature,
                                            'importance': most_important_value
                                        }
                                        
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
                                        
                                        # Composite feature insights (AGGREGATED for category comparison)
                                        composite_features = ['Traffic_Dispersion', 'Wind_Consistency', 'Activity_Index', 'Atmospheric_Stability']
                                        composite_totals = {}
                                        
                                        for comp_feature in composite_features:
                                            comp_rows = imp_df[imp_df['Feature'].str.contains(comp_feature)]
                                            if not comp_rows.empty:
                                                total_importance = comp_rows['Importance'].sum()
                                                composite_totals[comp_feature] = total_importance
                                                insights[f'{comp_feature}_importance'] = total_importance
                                        
                                        # Find top composite CATEGORY (not individual feature)
                                        if composite_totals:
                                            top_composite_category = max(composite_totals, key=composite_totals.get)
                                            insights['top_composite_category'] = {
                                                'name': top_composite_category,
                                                'total_importance': composite_totals[top_composite_category]
                                            }
                                        
                                        composite_insights[model_name] = insights   
                            
                            if composite_insights:
                                st.subheader("📈 Key Temporal & Composite Patterns Discovered")
                                
                                st.markdown("💡 **Individual Features** show single feature importance | **Composite Categories** show aggregated importance across all related features (base + lags)")

                                for model_name, insights in composite_insights.items():
                                    st.write(f"**{model_name.upper()} Model Findings:**")
                                    
                                    # OPTION 1: Use wider layout with better formatting
                                    col1, col2 = st.columns([1, 1])  # Split into 2 main columns
                                    
                                    with col1:
                                        st.write("**🎯 Feature Importance:**")
                                        # Top individual feature with full name
                                        if 'top_individual_feature' in insights:
                                            top_feat = insights['top_individual_feature']
                                            # Clean up the feature name for display
                                            display_name = top_feat['name'].replace('_', ' ').title()
                                            if len(display_name) > 15:  # Truncate long names
                                                display_name = display_name[:12] + "..."
                                            st.metric("Top Individual Feature", 
                                                    display_name,
                                                    f"{top_feat['importance']:.3f}")
                                            # top_feat = insights['top_individual_feature']
                                            # st.metric(
                                            #     label="Top Individual Feature",
                                            #     value=f"{top_feat['importance']:.3f}",
                                            #     help=f"Full name: {top_feat['name']}"  # Shows full name on hover
                                            # )
                                            # st.caption(f"**{top_feat['name']}**")  # Full name below metric
                                        
                                        # Top composite category with full name
                                        if 'top_composite_category' in insights:
                                            top_comp = insights['top_composite_category']
                                            category_name = top_comp['name'].replace('_', ' ')
                                            # if len(category_name) > 15:  # Truncate long names
                                            #     category_name = category_name[:12] + "..."
                                            st.metric("Top Composite Category", 
                                                    category_name,
                                                    f"{top_comp['total_importance']:.3f}" )

                                            # # top_comp = insights['top_composite_category']
                                            # # st.metric(
                                            # #     label="Top Composite Category",
                                            # #     value=f"{top_comp['total_importance']:.3f}",
                                            # #     help=f"Aggregated importance across all {top_comp['name']} features"
                                            # )
                                            # st.caption(f"**{top_comp['name'].replace('_', ' ')}**")
                                    
                                    with col2:
                                        st.write("**📅 Temporal Patterns:**")
                                        
                                        # Create sub-columns for temporal features
                                        temp_col1, temp_col2 = st.columns(2)
                                        
                                        with temp_col1:
                                            if 'best_day' in insights:
                                                st.metric("Most Predictive Day", insights['best_day'])
                                            if 'best_season' in insights:
                                                st.metric("Most Predictive Season", insights['best_season'])
                                        
                                        with temp_col2:
                                            if 'best_month' in insights:
                                                st.metric("Most Predictive Month", insights['best_month'])
                                    
                                    st.write("")
                        
                        # Key Insights
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
                        
                        # Recommendations
                        st.header("Recommendations")
                        
                        if not summary_df.empty:
                            best_model_name = summary_df.iloc[0]['Model'].lower()
                            best_model_score = float(summary_df.iloc[0]['Overall Score'])
                            
                            st.write("Based on the comprehensive analysis with composite features:")
                            
                            # Model recommendation
                            st.write(f"✅ **Recommended Model:** {best_model_name.upper()} with an overall score of {best_model_score:.1f}/100")
                            
                            # Performance insights
                            if best_model_score > 80:
                                st.write("📈 The model shows excellent performance with high accuracy and reliability for short-term forecasting.")
                            elif best_model_score > 60:
                                st.write("📊 The model shows good performance for short-term forecasting. Composite features improved prediction accuracy.")
                            else:
                                st.write("⚠️ Consider ensemble methods or additional composite features to improve short-term forecasting performance.")
                            
                            # Specific recommendations based on model type
                            if best_model_name in ['Random Forest', 'XGBoost', 'Decision Tree']:
                                st.write("🌳 Tree-based model selected - excellent for capturing non-linear patterns in composite features.")
                            elif best_model_name in ['knn']:
                                st.write("🎯 KNN selected - performs well with composite features and robust scaling!")
                            elif best_model_name in ['lstm', 'ann', 'gru']:
                                st.write("🧠 Neural network model selected - optimized with Optuna for superior performance with composite features.")
                            elif best_model_name in ['arima', 'prophet']:
                                st.write("📉 Time series model selected - suitable for short-term forecasting with temporal patterns.")
                            
                            # Composite features insights
                            st.write("**🔬 Composite Feature Benefits:**")
                            st.write("- Traffic-weather interactions capture pollution dispersion dynamics")
                            st.write("- Atmospheric stability indicators improve prediction accuracy")
                            st.write("- Human activity indices consolidate multiple emission sources")
                            st.write("- Chemical ratios provide insights into pollution formation processes")
                            
                            # Data quality recommendations
                            if best_model_name in st.session_state.model_results and 'outliers' in st.session_state.model_results[best_model_name]:
                                outlier_info = st.session_state.model_results[best_model_name]['outliers']
                                
                                st.write("**Data Quality Summary:**")
                                st.write(f"- Total outliers detected: {outlier_info['total_detected']}")
                                st.write(f"- Outliers effectively handled: {outlier_info['total_handled']}")
                                st.write(f"- Outliers remaining: {outlier_info['total_remaining']}")
                            
                            # Short-term forecasting specific recommendations
                            st.write("**Short-term Forecasting Notes:**")
                            st.write("- 1-3 day forecasts are most reliable for immediate air quality planning")
                            st.write("- Composite features significantly improve temporal pattern recognition")
                            # st.write("- Neural networks with Optuna optimization show superior performance for complex patterns")
                            st.write("- Consider real-time data updates for improved accuracy")
                            st.write("- Monitor model performance regularly for optimal forecasting")
                            
                            # Data quality reminder
                            if missing_info:
                                st.warning("⚠️ **Important:** Missing values were detected in the current dataset. "
                                            "Ensure data quality is maintained for optimal model performance.")
                            
                            # Final optimization recommendation
                            neural_models_available = [model for model in st.session_state.model_results.keys() if model in get_neural_network_models()]
                            if neural_models_available:
                                st.info("🎯 **Pro Tip:** Neural network models with Optuna optimization and composite features "
                                        "often provide the best performance for complex environmental forecasting tasks.")
                
                    else:
                        if st.session_state.data is None:
                            st.info("Please upload a dataset using the sidebar.")
                        else:
                            st.info("Please train at least one model in the 'Model Training' tab.")

            if __name__ == "__main__":
                main()


            #### END ####      
    else:
        st.error('⛔ [ATTENTION] Please confirm your data in the main page to proceed.')
else:
    st.markdown('⛔ [ATTENTION] No file found. Please upload and process file/s in the main page to access this module.')