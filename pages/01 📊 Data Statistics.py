import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import re

def detect_variable_types(df):
    """
    Detect variable types based on column names and data patterns
    Returns a dictionary mapping column names to detected types
    """
    variable_types = {}
    
    # Define patterns for different variable types
    type_patterns = {
        'Weather': {
            'keywords': ['temp', 'temperature', 'humidity', 'rain', 'precipitation', 'wind', 'weather', 
                        'pressure', 'solar', 'radiation', 'cloud', 'visibility', 'dew', 'frost',
                        'snow', 'storm', 'lightning', 'thunder', 'hail', 'fog', 'mist', 'wd', 'ws'],
            'units': ['°c', '°f', 'celsius', 'fahrenheit', 'mph', 'kmh', 'kph', 'mb', 'hpa', 
                     'mm', 'inches', '%rh', 'w/m2', 'm/s', 'degrees', '°'],
            'exact_matches': ['wd', 'ws']  # Exact column name matches for wind direction/speed
        },
        'Traffic': {
            'keywords': ['traffic', 'vehicle', 'car', 'speed', 'flow', 'volume', 'count', 
                        'occupancy', 'congestion', 'lane', 'road', 'highway', 'street',
                        'pedestrian', 'bike', 'bicycle', 'bus', 'truck', 'motorcycle'],
            'units': ['vph', 'vehicles/hour', 'km/h', 'mph', 'pcu'],
            'exact_matches': []
        },
        'Energy': {
            'keywords': ['power', 'energy', 'electricity', 'voltage', 'current', 'watt', 'kwh',
                        'consumption', 'generation', 'load', 'demand', 'battery', 'solar',
                        'wind', 'grid', 'frequency', 'phase'],
            'units': ['kw', 'mw', 'gw', 'kwh', 'mwh', 'gwh', 'v', 'a', 'hz', 'var', 'pf'],
            'exact_matches': []
        },
        'Environmental': {
            'keywords': ['pollution', 'air_quality', 'co2', 'no2', 'so2', 'pm2.5', 'pm10', 'o3',
                        'aqi', 'emission', 'carbon', 'ozone', 'particulate', 'smog',
                        'noise', 'decibel', 'db', 'water_quality', 'ph', 'no', 'nox'],
            'units': ['ppm', 'ppb', 'µg/m3', 'mg/m3', 'db', 'dba'],
            'exact_matches': ['aqi', 'no', 'nox']  # Exact column name matches for air quality indicators
        },
        'Economic': {
            'keywords': ['price', 'cost', 'revenue', 'profit', 'sales', 'income', 'expense',
                        'budget', 'financial', 'economic', 'gdp', 'inflation', 'currency',
                        'exchange', 'market', 'stock', 'investment'],
            'units': ['$', '€', '£', '¥', 'usd', 'eur', 'gbp', 'cny'],
            'exact_matches': []
        },
        'Health': {
            'keywords': ['health', 'medical', 'hospital', 'patient', 'disease', 'symptom',
                        'diagnosis', 'treatment', 'medication', 'heart_rate', 'blood_pressure',
                        'glucose', 'cholesterol', 'bmi', 'weight', 'height'],
            'units': ['bpm', 'mmhg', 'mg/dl', 'kg', 'lbs', 'cm', 'ft', 'in'],
            'exact_matches': []
        },
        'Demographic': {
            'keywords': ['population', 'age', 'gender', 'income', 'education', 'employment',
                        'household', 'family', 'birth', 'death', 'migration', 'census',
                        'demographic', 'ethnicity', 'race'],
            'units': ['years', 'people', 'persons', '%'],
            'exact_matches': []
        },
        'Temporal': {
            'keywords': ['date', 'time', 'timestamp', 'year', 'month', 'day', 'hour', 'minute',
                        'second', 'week', 'quarter', 'season', 'period'],
            'units': ['sec', 'min', 'hr', 'days', 'weeks', 'months', 'years'],
            'exact_matches': []
        },
        'Geographic': {
            'keywords': ['location', 'address', 'city', 'state', 'country', 'region', 'latitude',
                        'longitude', 'lat', 'lon', 'coordinate', 'postal', 'zip', 'area',
                        'distance', 'elevation', 'altitude'],
            'units': ['km', 'miles', 'meters', 'feet', 'degrees', '°'],
            'exact_matches': []
        }
    }
    
    for column in df.columns:
        col_lower = column.lower().replace('_', ' ').replace('-', ' ')
        detected_types = []
        
        # Check for temporal data types first
        if df[column].dtype in ['datetime64[ns]', 'datetime64[ns, UTC]'] or 'timestamp' in col_lower:
            detected_types.append('Temporal')
        
        # Check column name against patterns
        for var_type, patterns in type_patterns.items():
            # Check exact matches first (for abbreviations like WD, WS, AQI, NO, NOx)
            if 'exact_matches' in patterns:
                for exact_match in patterns['exact_matches']:
                    if col_lower.strip() == exact_match:
                        detected_types.append(var_type)
                        break
            
            # Check keywords
            for keyword in patterns['keywords']:
                if keyword in col_lower:
                    detected_types.append(var_type)
                    break
            
            # Check units in column name
            for unit in patterns['units']:
                if unit in col_lower:
                    detected_types.append(var_type)
                    break
        
        # Additional checks based on data content
        if len(detected_types) == 0:
            # Check if numeric column has specific patterns
            if pd.api.types.is_numeric_dtype(df[column]):
                sample_values = df[column].dropna().head(100)
                
                # Check value ranges for common types
                if not sample_values.empty:
                    min_val, max_val = sample_values.min(), sample_values.max()
                    
                    # Temperature-like ranges
                    if -50 <= min_val <= 50 and -50 <= max_val <= 50:
                        if any(keyword in col_lower for keyword in ['temp', 'temperature']):
                            detected_types.append('Weather')
                    
                    # Percentage-like ranges
                    elif 0 <= min_val <= 100 and 0 <= max_val <= 100:
                        if any(keyword in col_lower for keyword in ['humidity', 'percent', '%', 'rate']):
                            detected_types.append('Weather' if 'humidity' in col_lower else 'General')
                    
                    # Large numbers could be financial
                    elif max_val > 10000:
                        if any(keyword in col_lower for keyword in ['amount', 'value', 'total']):
                            detected_types.append('Economic')
            
            # Check string columns for geographic patterns
            elif df[column].dtype == 'object':
                sample_values = df[column].dropna().astype(str).head(20)
                
                # Check for coordinate patterns
                coord_pattern = r'^-?\d+\.?\d*$'
                if sample_values.str.match(coord_pattern).any():
                    if 'lat' in col_lower or 'lon' in col_lower:
                        detected_types.append('Geographic')
        
        # Assign the most specific type or 'General' if none detected
        if detected_types:
            # Remove duplicates and prioritize certain types
            detected_types = list(set(detected_types))
            if 'Temporal' in detected_types:
                variable_types[column] = 'Temporal'
            elif len(detected_types) == 1:
                variable_types[column] = detected_types[0]
            else:
                # If multiple types detected, choose the most specific one
                priority_order = ['Weather', 'Traffic', 'Environmental', 'Energy', 'Health', 
                                'Economic', 'Demographic', 'Geographic', 'Temporal']
                for priority_type in priority_order:
                    if priority_type in detected_types:
                        variable_types[column] = priority_type
                        break
                else:
                    variable_types[column] = detected_types[0]
        else:
            variable_types[column] = 'General'
    
    return variable_types

if 'uploaded_file' not in st.session_state:
        st.session_state['uploaded_file'] = 'no'
uploaded_file = st.session_state['uploaded_file']

if(uploaded_file == 'yes'):
    if st.session_state.data_confirmed:
        df = st.session_state['df']

        # Detect variable types
        variable_types = detect_variable_types(df)

        # Set page configuration
        st.set_page_config(
            page_title="DataFrame Explorer",
            layout="wide"
        )
        # Header
        st.info('⚪ Original dataset is loaded for this page.')
        st.title("📊 DataFrame Explorer")
        # Sidebar for options
        with st.sidebar:
                
            # Plot options
            st.subheader("Missing Values Explorer")
            plot_type = st.radio(
                "Select plot type:",
                ["Bar Chart", "Heatmap"],
                index=0
            )
            
            # Show percentages option
            show_percent = st.checkbox("Show percentages", value=False)

            available_types = list(set(variable_types.values()))

            selected_types = available_types #

        # Display basic info about dataframe
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<div class='header-style'>DataFrame Info</div>", unsafe_allow_html=True)
            st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
            
            # Count missing values
            total_cells = df.size
            missing_cells = df.isna().sum().sum()
            missing_percentage = (missing_cells / total_cells) * 100
            
            st.write(f"🛑 Total missing values: {missing_cells} ({missing_percentage:.2f}%)")
            if (missing_cells > 0):
                st.error("⚠ Missing values found. Please use the data cleaning and imputation tool to fill in the missing values.")

        with col2:
            st.markdown("<div class='header-style'>Data Types</div>", unsafe_allow_html=True)
            dtypes_df = pd.DataFrame(df.dtypes, columns=['Data Type'])
            dtypes_df = dtypes_df.reset_index().rename(columns={'index': 'Column'})
            st.dataframe(dtypes_df, use_container_width=True)
     
        # Create variable types summary
        var_types_summary = pd.DataFrame(list(variable_types.items()), 
                                       columns=['Column', 'Variable Type'])
        
        # Add color coding for different types
        type_colors = {
            'Weather': '🌤️',
            'Traffic': '🚗',
            'Energy': '⚡',
            'Environmental': '🌍',
            'Economic': '💰',
            'Health': '🏥',
            'Demographic': '👥',
            'Temporal': '🕐',
            'Geographic': '📍',
            'General': '📊'
        }

        # Create a detailed visual breakdown of columns by variable type
        st.markdown("##### 🏷️ Variable Types Detected")
        
        # Create columns for better layout
        num_types = len(set(variable_types.values()))
        cols_per_row = 2
        num_rows = (num_types + cols_per_row - 1) // cols_per_row
        
        for row in range(num_rows):
            cols = st.columns(cols_per_row)
            
            var_types_list = sorted(set(variable_types.values()))
            start_idx = row * cols_per_row
            end_idx = min(start_idx + cols_per_row, len(var_types_list))
            
            for i, var_type in enumerate(var_types_list[start_idx:end_idx]):
                with cols[i]:
                    cols_of_type = [col for col, vtype in variable_types.items() if vtype == var_type]
                    icon = type_colors.get(var_type, '📊')
                    
                    # Create a nice card-like display
                    st.markdown(f"""
                    <div style="
                        border: 2px solid #e0e0e0; 
                        border-radius: 10px; 
                        padding: 15px; 
                        margin: 10px 0; 
                        background-color: #f8f9fa;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    ">
                        <h4 style="margin: 0 0 10px 0; color: #333;">
                            {icon} {var_type}
                        </h4>
                        <p style="margin: 5px 0; color: #666; font-size: 14px;">
                            <strong>{len(cols_of_type)} columns</strong>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show column names in an expandable section
                    with st.expander(f"View {var_type} Columns", expanded=False):
                        for col in cols_of_type:
                            # Add description for special cases
                            description = ""
                            if col.upper() == "WD":
                                description = " (Wind Direction)"
                            elif col.upper() == "WS":
                                description = " (Wind Speed)"
                            elif col.upper() == "AQI":
                                description = " (Air Quality Index)"
                            elif col.upper() == "NO":
                                description = " (Nitric Oxide)"
                            elif col.upper() == "NO2":
                                description = " (Nitrogen Dioxide)"
                            elif col.upper() == "NOX":
                                description = " (Nitrogen Oxides)"
                            
                            dtype = str(df[col].dtype)
                            null_count = df[col].isnull().sum()
                            
                            st.markdown(f"""
                            <div style="
                                padding: 8px; 
                                margin: 5px 0; 
                                background-color: white; 
                                color: blue;
                                border-left: 3px solid #007bff;
                                border-radius: 3px;
                            ">
                                <strong>{col}</strong>{description}<br>
                                <small style="color: #666;">
                                    Type: {dtype} | Missing: {null_count}
                                </small>
                            </div>
                            """, unsafe_allow_html=True)

        # Add data count per year table
        st.markdown("<div class='header-style'>Data Count per Year</div>", unsafe_allow_html=True)
        df['timestamp'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str)) #F
        df['timestamp'] = pd.to_datetime(df['timestamp']) #F
        # Drop Date and Time columns, reset index to timestamp    
        df = df.drop(columns=['Date', 'Time'])#.set_index('timestamp') #F
        st.dataframe(df.groupby([df.timestamp.dt.year]).apply(lambda x: x.notna().sum()))
        
        # Add descriptive statistics table
        st.markdown("<div class='header-style'>Descriptive Statistics</div>", unsafe_allow_html=True)

        # Create tabs for different statistics views
        stats_tab1, stats_tab2, stats_tab3 = st.tabs(["Summary Statistics", "Distribution Statistics", "Correlation"])

        with stats_tab1:
            # Calculate descriptive statistics for numeric columns
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            
            # Filter numeric columns by selected variable types
            filtered_numeric_cols = [col for col in numeric_cols 
                                   if variable_types.get(col, 'General') in selected_types]
            
            if filtered_numeric_cols:
                # Basic statistics (count, mean, std, min, 25%, 50%, 75%, max)
                stats_df = df[filtered_numeric_cols].describe()
                
                # Add additional metrics
                stats_df.loc['missing'] = df[filtered_numeric_cols].isnull().sum()
                stats_df.loc['missing %'] = (df[filtered_numeric_cols].isnull().sum() / len(df)) * 100
                stats_df.loc['variable_type'] = [variable_types.get(col, 'General') for col in filtered_numeric_cols]
                
                # Display the statistics with formatter
                st.dataframe(
                    stats_df.style.format({
                        col: '{:.2f}' for col in stats_df.columns
                    }).format({
                        'missing %': '{:.2f}%' for col in stats_df.columns
                    }),
                    use_container_width=True
                )
            else:
                st.info("⚠ No numeric columns found in the selected variable types for statistics.")

        with stats_tab2:
            if filtered_numeric_cols:
                # Distribution statistics (skewness, kurtosis)
                dist_stats = pd.DataFrame(index=filtered_numeric_cols)
                
                try:
                    # Calculate skewness
                    dist_stats['skewness'] = df[filtered_numeric_cols].skew()
                    # Calculate kurtosis
                    dist_stats['kurtosis'] = df[filtered_numeric_cols].kurt()
                    # Calculate range
                    dist_stats['range'] = df[filtered_numeric_cols].max() - df[filtered_numeric_cols].min()
                    # Calculate coefficient of variation
                    dist_stats['CV'] = (df[filtered_numeric_cols].std() / df[filtered_numeric_cols].mean()) * 100
                    # Add variable type
                    dist_stats['variable_type'] = [variable_types.get(col, 'General') for col in filtered_numeric_cols]
                    
                    # Display distribution statistics
                    st.dataframe(
                        dist_stats.style.format('{:.2f}').format({'CV': '{:.2f}%'}),
                        use_container_width=True
                    )
                    
                    # Add interpretation guide
                    with st.expander("Interpretation Guide"):
                        st.markdown("""
                        - **Skewness**: Measures the asymmetry of the distribution
                            - Values close to 0 indicate symmetrical distribution
                            - Positive values indicate right skew (tail extends to the right)
                            - Negative values indicate left skew (tail extends to the left)
                        - **Kurtosis**: Measures the "tailedness" of the distribution
                            - Values close to 0 indicate normal distribution
                            - Positive values indicate heavy tails (more outliers)
                            - Negative values indicate light tails (fewer outliers)
                        - **Range**: The difference between the maximum and minimum values
                        - **CV (Coefficient of Variation)**: Standard deviation divided by mean (×100%)
                            - Lower values indicate less variability relative to the mean
                        """)
                except Exception as e:
                    st.error(f"⛔ Error calculating distribution statistics: {e}")
            else:
                st.info("⚠ No numeric columns found in the selected variable types for distribution statistics.")

        with stats_tab3:
            if len(filtered_numeric_cols) > 1:
                try:
                    # Calculate correlation matrix
                    corr_matrix = df[filtered_numeric_cols].corr()
                    
                    # Display correlation matrix
                    st.write("##### Correlation Matrix")
                    st.dataframe(corr_matrix.style.format('{:.2f}').background_gradient(cmap='coolwarm'), use_container_width=True)
                    
                    # Display correlation heatmap
                    st.write("##### Correlation Heatmap")
                    fig = px.imshow(
                        corr_matrix,
                        color_continuous_scale='RdBu_r',
                        labels=dict(x="Features", y="Features", color="Correlation"),
                        zmin=-1, zmax=1
                    )
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"⛔ Error calculating correlation: {e}")
            else:
                st.info("⚠ Need at least two numeric columns in selected variable types to calculate correlation.")

        # Show missing values if any
        if df.isnull().any().any():
            st.markdown("<div class='header-style'>Missing Values Analysis</div>", unsafe_allow_html=True)
            
            # Calculate null counts
            null_counts = df.isnull().sum().sort_values(ascending=False)
            null_counts = null_counts[null_counts > 0]
            
            if len(null_counts) > 0:
                null_df = null_counts.reset_index()
                null_df.columns = ['Column', 'Null Count']
                null_df['Percentage'] = (null_df['Null Count'] / len(df)) * 100
                null_df['Variable Type'] = null_df['Column'].map(variable_types)
                null_df['Icon'] = null_df['Variable Type'].map(type_colors)
                
                # Reorder columns
                null_df = null_df[['Icon', 'Column', 'Variable Type', 'Null Count', 'Percentage']]
                
                # Display missing values table
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.markdown("##### Missing Values Summary")
                    st.dataframe(
                        null_df.style.format({'Percentage': '{:.2f}%'}),
                        use_container_width=True,
                        hide_index=True
                    )
                
                with col2:
                    st.markdown("##### Missing Values Visualization")
                    
                    if plot_type == "Bar Chart":
                        # Create interactive bar chart with plotly
                        fig = px.bar(
                            null_df,
                            x='Column',
                            y='Null Count',
                            color='Variable Type',
                            text='Null Count' if not show_percent else None,
                            title="Missing Values by Column and Variable Type",
                            color_discrete_sequence=['#8B0000', '#FF1744', '#FF5722', '#E91E63', '#D32F2F', 
                                                   '#C62828', '#B71C1C', '#AD1457', '#880E4F', '#4A148C']
                        )
                        
                        if show_percent:
                            fig.add_trace(
                                px.bar(
                                    null_df, 
                                    x='Column', 
                                    y='Percentage',
                                    text=[f'{p:.1f}%' for p in null_df['Percentage']]
                                ).data[0]
                            )
                            fig.update_layout(yaxis_title="Count / Percentage")
                        
                        fig.update_layout(
                            xaxis_title="Columns",
                            yaxis_title="Missing Value Count",
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                    elif plot_type == "Heatmap":
                        # Create heatmap of missing values
                        fig = px.imshow(
                            df.isna().T,
                            color_continuous_scale=['white', 'red'],
                            labels=dict(x="Row Index", y="Column", color="Missing"),
                            title="Missing Values Heatmap"
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
    
                
                # Add some explanatory text
                st.markdown("""
                **Understanding the visualization:**
                
                - Columns with higher bars have more missing values
                - The percentages (if shown) indicate what portion of each column is missing
                - Colors represent different variable types (Weather, Traffic, Environmental, etc.)
                - Consider how these missing values might impact your analysis
                """)
                       
            # Create a styled dataframe highlighting missing values
            def highlight_missing(val):
                return 'background-color: rgba(255, 165, 0, 0.2)' if pd.isna(val) else ''
            
            # if (missing_cells > 0):
            st.markdown("<div class='header-style'>🛑 Missing Values Preview</div>", unsafe_allow_html=True)
            missing_df = df[df.isnull().any(axis=1)]
            max_rows = min(100, missing_df.shape[0])
            styled_df = missing_df.head(max_rows).style.applymap(highlight_missing)
            st.dataframe(styled_df, use_container_width=True)
            if missing_df.shape[0] > max_rows:
                st.info(f"Showing first {max_rows} rows. The dataset has {missing_df.shape[0]} rows with missing values in total.")

        else:
            st.success("✅ No missing values found in the DataFrame.")
            
            # Still show the dataframe
            st.markdown("<div class='header-style'>DataFrame Preview</div>", unsafe_allow_html=True)
            st.dataframe(df.head(100), use_container_width=True)
            max_rows = min(100, df.shape[0])
            st.info(f"Showing first {max_rows} rows. The dataset has {df.shape[0]} rows in total.")
    else:
        st.error('⛔ [ATTENTION] Please confirm your data in the main page to proceed.')
else:
    st.markdown('⛔ [ATTENTION] No file found. Please upload and process file/s in the main page to access this module.')

