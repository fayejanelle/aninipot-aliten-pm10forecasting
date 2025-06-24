import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.interpolate import interp1d
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Data Cleaning & Imputation Tool",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 0rem 0rem;
    }
    .stProgress .st-bo {
        background-color: #667eea;
    }
    .stat-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    .stat-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #667eea;
    }
    .stat-label {
        font-size: 1rem;
        color: #666;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'raw_data' not in st.session_state:
    st.session_state.raw_data = None
if 'processing_log' not in st.session_state:
    st.session_state.processing_log = []

# Helper functions
def log_message(message, type='info'):
    """Add message to processing log"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.processing_log.append({
        'time': timestamp,
        'message': message,
        'type': type
    })

def detect_anomalies(df, column):
    """Detect anomalies using multiple methods"""
    anomalies = {
        'negative_values': [],
        'outliers': [],
        'impossible_values': []
    }
    
    # Check for negative values in columns that shouldn't have them
    non_negative_cols = ['AQI', 'PM10', 'PM2.5', 'NO2', 'NO', 'NOx', 'WS', 
                        'Total_Pedestrians', 'City_Centre_TVCount', 'TrafficV']
    
    if column in non_negative_cols:
        neg_mask = df[column] < 0
        anomalies['negative_values'] = df[neg_mask].index.tolist()
    
    # IQR method for outliers
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outlier_mask = (df[column] < lower_bound) | (df[column] > upper_bound)
    anomalies['outliers'] = df[outlier_mask].index.tolist()
    
    # Domain-specific impossible values
    if column == 'WD':
        impossible_mask = (df[column] < 0) | (df[column] > 360)
        anomalies['impossible_values'] = df[impossible_mask].index.tolist()
    elif column == 'Temp':
        impossible_mask = (df[column] < -50) | (df[column] > 60)
        anomalies['impossible_values'] = df[impossible_mask].index.tolist()
    
    return anomalies

def clean_data(df):
    """Clean data by fixing or removing anomalies"""
    df_cleaned = df.copy()
    cleaning_stats = {}
    
    numeric_columns = ['AQI', 'PM10', 'PM2.5', 'NO2', 'NO', 'NOx', 'Temp', 
                      'WD', 'WS', 'Total_Pedestrians', 'City_Centre_TVCount', 'TrafficV']
    
    for col in numeric_columns:
        if col in df_cleaned.columns:
            cleaning_stats[col] = {'cleaned': 0}
            
            # Fix negative values
            if col in ['AQI', 'PM10', 'PM2.5', 'NO2', 'NO', 'NOx', 'WS', 
                      'Total_Pedestrians', 'City_Centre_TVCount', 'TrafficV']:
                neg_mask = df_cleaned[col] < 0
                cleaning_stats[col]['cleaned'] += neg_mask.sum()
                df_cleaned.loc[neg_mask, col] = np.nan
            
            # Fix wind direction
            if col == 'WD':
                # Wrap around values
                df_cleaned[col] = df_cleaned[col] % 360
                # Fix negative values
                neg_mask = df_cleaned[col] < 0
                df_cleaned.loc[neg_mask, col] = df_cleaned.loc[neg_mask, col] + 360
                cleaning_stats[col]['cleaned'] += neg_mask.sum()
            
            # Fix extreme temperatures
            if col == 'Temp':
                extreme_mask = (df_cleaned[col] < -50) | (df_cleaned[col] > 60)
                cleaning_stats[col]['cleaned'] += extreme_mask.sum()
                df_cleaned.loc[extreme_mask, col] = np.nan
    
    return df_cleaned, cleaning_stats

def seasonal_imputation(df, column):
    """Impute using seasonal patterns (hourly averages)"""
    df_copy = df.copy()
    
    # Extract hour from Time column
    df_copy['Hour'] = pd.to_datetime(df_copy['Time'], format='%H:%M:%S', errors='coerce').dt.hour
    
    # Calculate hourly averages
    hourly_avg = df_copy.groupby('Hour')[column].mean()
    
    # Fill missing values with hourly average + small random variation
    missing_mask = df_copy[column].isna()
    for idx in df_copy[missing_mask].index:
        hour = df_copy.loc[idx, 'Hour']
        if hour in hourly_avg.index and not pd.isna(hourly_avg[hour]):
            variation = 0.1 * hourly_avg[hour] * (np.random.random() - 0.5)
            df_copy.loc[idx, column] = hourly_avg[hour] + variation
    
    return df_copy[column]

def pattern_based_imputation(df, column):
    """Impute using day-of-week and hourly patterns"""
    df_copy = df.copy()
    
    # Parse datetime
    df_copy['DateTime'] = pd.to_datetime(df_copy['Date'] + ' ' + df_copy['Time'], errors='coerce')
    df_copy['DayOfWeek'] = df_copy['DateTime'].dt.dayofweek
    df_copy['Hour'] = df_copy['DateTime'].dt.hour
    
    # Calculate patterns
    pattern_avg = df_copy.groupby(['DayOfWeek', 'Hour'])[column].mean()
    
    # Fill missing values
    missing_mask = df_copy[column].isna()
    for idx in df_copy[missing_mask].index:
        dow = df_copy.loc[idx, 'DayOfWeek']
        hour = df_copy.loc[idx, 'Hour']
        if (dow, hour) in pattern_avg.index and not pd.isna(pattern_avg[(dow, hour)]):
            variation = 0.05 * pattern_avg[(dow, hour)] * (np.random.random() - 0.5)
            df_copy.loc[idx, column] = max(0, pattern_avg[(dow, hour)] + variation)
    
    return df_copy[column]

def multivariate_imputation(df, column, related_columns):
    """Impute using multivariate regression"""
    df_copy = df.copy()
    
    # Prepare training data
    train_mask = df_copy[column].notna()
    
    # Check if we have related columns with enough data
    valid_related = [col for col in related_columns if col in df_copy.columns and df_copy[col].notna().sum() > 10]
    
    if len(valid_related) == 0 or train_mask.sum() < 10:
        # Fall back to mean imputation
        mean_value = df_copy[column].mean()
        df_copy[column].fillna(mean_value, inplace=True)
        return df_copy[column]
    
    # Prepare features
    X_train = df_copy.loc[train_mask, valid_related].values
    y_train = df_copy.loc[train_mask, column].values
    
    # Handle any remaining NaN in features
    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train)
    
    # Train model
    rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    # Predict missing values
    missing_mask = df_copy[column].isna()
    if missing_mask.sum() > 0:
        X_missing = df_copy.loc[missing_mask, valid_related].values
        X_missing = imputer.transform(X_missing)
        predictions = rf.predict(X_missing)
        df_copy.loc[missing_mask, column] = np.maximum(0, predictions)
    
    return df_copy[column]

def perform_imputation(df, method='auto'):
    """Main imputation function"""
    df_imputed = df.copy()
    imputation_details = {}
    
    numeric_columns = ['AQI', 'PM10', 'PM2.5', 'NO2', 'NO', 'NOx', 'Temp', 
                      'WD', 'WS', 'Total_Pedestrians', 'City_Centre_TVCount', 'TrafficV']
    
    # Define relationships for multivariate imputation
    relationships = {
        'AQI': ['PM2.5', 'PM10', 'NO2'],
        'PM10': ['PM2.5', 'AQI', 'WS'],
        'PM2.5': ['PM10', 'AQI', 'WS'],
        'NO2': ['NO', 'NOx', 'TrafficV'],
        'NO': ['NO2', 'NOx', 'TrafficV'],
        'NOx': ['NO', 'NO2', 'TrafficV']
    }
    
    for col in numeric_columns:
        if col in df_imputed.columns:
            missing_count = df_imputed[col].isna().sum()
            missing_pct = (missing_count / len(df_imputed)) * 100
            
            if missing_count > 0:
                if method == 'auto':
                    # Automatic method selection based on data characteristics
                    if missing_pct < 5:
                        # Linear interpolation for low missing
                        df_imputed[col] = df_imputed[col].interpolate(method='linear', limit_direction='both')
                        imputation_details[col] = 'Linear Interpolation'
                    elif col in ['Temp', 'WS', 'WD']:
                        # Seasonal patterns for weather data
                        df_imputed[col] = seasonal_imputation(df_imputed, col)
                        imputation_details[col] = 'Seasonal Decomposition'
                    elif col in ['Total_Pedestrians', 'City_Centre_TVCount', 'TrafficV']:
                        # Pattern-based for traffic data
                        df_imputed[col] = pattern_based_imputation(df_imputed, col)
                        imputation_details[col] = 'Pattern-based (Day/Hour)'
                    elif col in relationships:
                        # Multivariate for air quality
                        df_imputed[col] = multivariate_imputation(df_imputed, col, relationships[col])
                        imputation_details[col] = 'Multivariate Regression'
                    else:
                        # Default to KNN
                        imputer = KNNImputer(n_neighbors=5)
                        df_imputed[col] = imputer.fit_transform(df_imputed[[col]])
                        imputation_details[col] = 'KNN Imputation'
                else:
                    # Use specified method
                    if method == 'mean':
                        df_imputed[col].fillna(df_imputed[col].mean(), inplace=True)
                    elif method == 'median':
                        df_imputed[col].fillna(df_imputed[col].median(), inplace=True)
                    elif method == 'forward_fill':
                        df_imputed[col].fillna(method='ffill', inplace=True)
                    elif method == 'interpolate':
                        df_imputed[col] = df_imputed[col].interpolate(method='linear', limit_direction='both')
                    imputation_details[col] = method
            else:
                imputation_details[col] = 'No imputation needed'
    
    return df_imputed, imputation_details

# Main app
st.title("🔧 Advanced Data Cleaning & Imputation Tool")
st.markdown("Transform rawX.csv to imputed_dataX.csv with scientific standards and ML methods")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # File upload
    uploaded_file = st.file_uploader("Upload rawX.csv", type=['csv'])
    
    if uploaded_file is not None:
        # Imputation method selection
        st.subheader("Imputation Method")
        imputation_method = st.selectbox(
            "Select method",
            ['auto', 'mean', 'median', 'forward_fill', 'interpolate'],
            help="Auto mode selects the best method for each column"
        )
        
        # Processing options
        st.subheader("Processing Options")
        remove_outliers = st.checkbox("Remove extreme outliers", value=True)
        validate_ranges = st.checkbox("Validate data ranges", value=True)
        add_noise = st.checkbox("Add small random noise", value=True)
        
        # Start processing button
        if st.button("🚀 Start Processing", type="primary"):
            with st.spinner("Processing data..."):
                # Clear previous results
                st.session_state.processing_log = []
                
                # Load data
                log_message("Loading data...")
                df = pd.read_csv(uploaded_file)
                st.session_state.raw_data = df.copy()
                log_message(f"Loaded {len(df)} rows with {len(df.columns)} columns", 'success')
                
                # Data cleaning
                log_message("Starting data cleaning...")
                df_cleaned, cleaning_stats = clean_data(df)
                total_cleaned = sum(stat['cleaned'] for stat in cleaning_stats.values())
                log_message(f"Cleaned {total_cleaned} anomalous values", 'success')
                
                # Imputation
                log_message("Starting intelligent imputation...")
                df_imputed, imputation_details = perform_imputation(df_cleaned, method=imputation_method)
                log_message("Imputation completed", 'success')
                
                # Validation
                if validate_ranges:
                    log_message("Validating data ranges...")
                    validation_rules = {
                        'AQI': (0, 500),
                        'PM10': (0, 1000),
                        'PM2.5': (0, 500),
                        'NO2': (0, 200),
                        'NO': (0, 200),
                        'NOx': (0, 400),
                        'Temp': (-30, 50),
                        'WD': (0, 360),
                        'WS': (0, 50),
                        'Total_Pedestrians': (0, 100000),
                        'City_Centre_TVCount': (0, 100000),
                        'TrafficV': (0, 100000)
                    }
                    
                    for col, (min_val, max_val) in validation_rules.items():
                        if col in df_imputed.columns:
                            df_imputed[col] = df_imputed[col].clip(min_val, max_val)
                    
                    log_message("Data validation completed", 'success')
                
                st.session_state.processed_data = df_imputed
                log_message("Processing pipeline completed!", 'success')

# Main content area
if st.session_state.raw_data is not None:
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "🔍 Data Quality", "🛠️ Processing", "📈 Results", "💾 Export"])
    
    with tab1:
        st.header("Data Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="stat-card">
                <div class="stat-label">Total Rows</div>
                <div class="stat-value">{:,}</div>
            </div>
            """.format(len(st.session_state.raw_data)), unsafe_allow_html=True)
        
        with col2:
            total_missing = st.session_state.raw_data.isna().sum().sum()
            st.markdown("""
            <div class="stat-card">
                <div class="stat-label">Missing Values</div>
                <div class="stat-value">{:,}</div>
            </div>
            """.format(total_missing), unsafe_allow_html=True)
        
        with col3:
            completeness = ((len(st.session_state.raw_data) * len(st.session_state.raw_data.columns) - total_missing) / 
                           (len(st.session_state.raw_data) * len(st.session_state.raw_data.columns)) * 100)
            st.markdown("""
            <div class="stat-card">
                <div class="stat-label">Completeness</div>
                <div class="stat-value">{:.1f}%</div>
            </div>
            """.format(completeness), unsafe_allow_html=True)
        
        with col4:
            if st.session_state.processed_data is not None:
                processed_missing = st.session_state.processed_data.isna().sum().sum()
                st.markdown("""
                <div class="stat-card">
                    <div class="stat-label">After Processing</div>
                    <div class="stat-value">{:,}</div>
                </div>
                """.format(processed_missing), unsafe_allow_html=True)
        
        # Sample data
        st.subheader("Sample Data (First 10 rows)")
        st.dataframe(st.session_state.raw_data.head(10))
    
    with tab2:
        st.header("Data Quality Analysis")
        
        # Missing values analysis
        numeric_columns = ['AQI', 'PM10', 'PM2.5', 'NO2', 'NO', 'NOx', 'Temp', 
                          'WD', 'WS', 'Total_Pedestrians', 'City_Centre_TVCount', 'TrafficV']
        
        missing_data = []
        for col in numeric_columns:
            if col in st.session_state.raw_data.columns:
                missing_count = st.session_state.raw_data[col].isna().sum()
                missing_pct = (missing_count / len(st.session_state.raw_data)) * 100
                missing_data.append({
                    'Column': col,
                    'Missing Count': missing_count,
                    'Missing %': f"{missing_pct:.2f}%"
                })
        
        missing_df = pd.DataFrame(missing_data)
        
        # Create missing values visualization
        fig_missing = px.bar(missing_df, x='Column', y='Missing Count', 
                            title='Missing Values by Column',
                            color='Missing Count',
                            color_continuous_scale='Reds')
        st.plotly_chart(fig_missing, use_container_width=True)
        
        # Display missing values table
        st.subheader("Missing Values Summary")
        st.dataframe(missing_df)
        
        # Anomaly detection
        st.subheader("Anomaly Detection")
        selected_col = st.selectbox("Select column for anomaly analysis", numeric_columns)
        
        if selected_col in st.session_state.raw_data.columns:
            anomalies = detect_anomalies(st.session_state.raw_data, selected_col)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Negative Values", len(anomalies['negative_values']))
            with col2:
                st.metric("Outliers (IQR)", len(anomalies['outliers']))
            with col3:
                st.metric("Impossible Values", len(anomalies['impossible_values']))
            
            # Distribution plot
            fig_dist = go.Figure()
            fig_dist.add_trace(go.Histogram(x=st.session_state.raw_data[selected_col].dropna(), 
                                           name='Distribution',
                                           nbinsx=50))
            fig_dist.update_layout(title=f'Distribution of {selected_col}',
                                 xaxis_title=selected_col,
                                 yaxis_title='Frequency')
            st.plotly_chart(fig_dist, use_container_width=True)
    
    with tab3:
        st.header("Processing Log")
        
        if len(st.session_state.processing_log) > 0:
            for log in st.session_state.processing_log:
                if log['type'] == 'success':
                    st.success(f"[{log['time']}] {log['message']}")
                elif log['type'] == 'warning':
                    st.warning(f"[{log['time']}] {log['message']}")
                elif log['type'] == 'error':
                    st.error(f"[{log['time']}] {log['message']}")
                else:
                    st.info(f"[{log['time']}] {log['message']}")
        else:
            st.info("No processing performed yet. Click 'Start Processing' in the sidebar.")
    
    with tab4:
        st.header("Processing Results")
        
        if st.session_state.processed_data is not None:
            # Before/After comparison
            st.subheader("Before vs After Comparison")
            
            comparison_data = []
            for col in numeric_columns:
                if col in st.session_state.raw_data.columns:
                    before_missing = st.session_state.raw_data[col].isna().sum()
                    after_missing = st.session_state.processed_data[col].isna().sum()
                    improvement = before_missing - after_missing
                    
                    comparison_data.append({
                        'Column': col,
                        'Before Missing': before_missing,
                        'After Missing': after_missing,
                        'Improvement': improvement,
                        'Improvement %': f"{(improvement/before_missing*100):.1f}%" if before_missing > 0 else "N/A"
                    })
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df)
            
            # Visualization of improvements
            fig_improvement = px.bar(comparison_df, x='Column', y='Improvement',
                                    title='Data Quality Improvements by Column',
                                    color='Improvement',
                                    color_continuous_scale='Greens')
            st.plotly_chart(fig_improvement, use_container_width=True)
            
            # Sample of processed data
            st.subheader("Processed Data Sample")
            st.dataframe(st.session_state.processed_data.head(10))
        else:
            st.info("No processed data available. Please run the processing pipeline first.")
    
    with tab5:
        st.header("Export Processed Data")
        
        if st.session_state.processed_data is not None:
            # Download button
            csv = st.session_state.processed_data.to_csv(index=False)
            st.download_button(
                label="📥 Download imputed_dataX.csv",
                data=csv,
                file_name="imputed_dataX.csv",
                mime="text/csv"
            )
            
            # Export statistics
            st.subheader("Export Summary")
            st.write(f"- **File name:** imputed_dataX.csv")
            st.write(f"- **Rows:** {len(st.session_state.processed_data):,}")
            st.write(f"- **Columns:** {len(st.session_state.processed_data.columns)}")
            st.write(f"- **Size:** ~{len(csv) / 1024 / 1024:.2f} MB")
            
            # Quality report
            st.subheader("Data Quality Report")
            report = f"""
            Data Cleaning and Imputation Report
            Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            
            Original Data:
            - Rows: {len(st.session_state.raw_data):,}
            - Missing Values: {st.session_state.raw_data.isna().sum().sum():,}
            - Completeness: {((len(st.session_state.raw_data) * len(st.session_state.raw_data.columns) - st.session_state.raw_data.isna().sum().sum()) / (len(st.session_state.raw_data) * len(st.session_state.raw_data.columns)) * 100):.2f}%
            
            Processed Data:
            - Rows: {len(st.session_state.processed_data):,}
            - Missing Values: {st.session_state.processed_data.isna().sum().sum():,}
            - Completeness: {((len(st.session_state.processed_data) * len(st.session_state.processed_data.columns) - st.session_state.processed_data.isna().sum().sum()) / (len(st.session_state.processed_data) * len(st.session_state.processed_data.columns)) * 100):.2f}%
            
            Processing Methods Used:
            - Anomaly Detection: IQR method
            - Cleaning: Domain-specific rules
            - Imputation: {'Automatic (ML-based)' if imputation_method == 'auto' else imputation_method}
            """
            
            st.text_area("Quality Report", report, height=300)
            
            # Download report
            st.download_button(
                label="📄 Download Quality Report",
                data=report,
                file_name="data_quality_report.txt",
                mime="text/plain"
            )
        else:
            st.info("No processed data available for export.")
else:
    # Welcome screen
    st.info("👈 Please upload your file in the sidebar to begin processing.")
    
    # Instructions
    with st.expander("📖 How to use this tool"):
        st.markdown("""
        1. **Upload your data**: Use the sidebar to upload rawX.csv
        2. **Configure options**: Select imputation method and processing options
        3. **Start processing**: Click the 'Start Processing' button
        4. **Review results**: Check the different tabs for analysis and results
        5. **Export data**: Download the cleaned and imputed data as imputed_dataX.csv
        
        **Features:**
        - Automatic anomaly detection
        - Multiple imputation methods (ML-based)
        - Data validation and quality checks
        - Interactive visualizations
        - Detailed processing logs
        - Quality reports
        """)
    
    # Feature highlights
    st.subheader("🌟 Key Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🔍 Anomaly Detection**
        - Negative value detection
        - Statistical outlier detection
        - Domain-specific validation
        - Impossible value identification
        """)
    
    with col2:
        st.markdown("""
        **🤖 ML-Based Imputation**
        - Linear interpolation
        - Seasonal decomposition
        - Pattern recognition
        - Multivariate regression
        """)
    
    with col3:
        st.markdown("""
        **📊 Quality Assurance**
        - Data completeness tracking
        - Before/after comparisons
        - Validation reports
        - Processing transparency
        """)