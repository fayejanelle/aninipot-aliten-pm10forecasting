import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')
st.set_page_config(page_title="Exploratory Data Analysis for PM10 (Daily)", page_icon="🚀", layout="wide")
if 'uploaded_file' not in st.session_state:
        st.session_state['uploaded_file'] = 'no'
uploaded_file = st.session_state['uploaded_file']

if(uploaded_file == 'yes'):
    if st.session_state.data_confirmed:

        imputed = st.session_state['imputed']
        if(imputed =='yes'):
            df = st.session_state['imputed_df'] 
            st.info('⚪ Imputed dataset is loaded for this page.')
        else:
            df = st.session_state['df']
            st.info('⚪ Original dataset is loaded for this page.')

        # Checking null counts for PM10 and PM2.5
        nulls = df.isnull().sum().sum()
        
        if(nulls > 0):
            st.error('❌ Missing values are found in the dataset. To proceed, please use the available data cleaning and imputation tool to fill in the missing values.')

        else:
            # st.write('✅ No missing values found.')
            st.title('🚀 Exploratory Data Analysis for PM10 (Daily)')

            # Define functions
            @st.cache_data
            def load_data():

                df['Date'] = pd.to_datetime(df['Date'])
                
                # Create DateTime for potential use in some visualizations
                df['DateTime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str))
                
                # Extract additional time features
                df['Year'] = df['Date'].dt.year
                df['Month'] = df['Date'].dt.month
                df['Day'] = df['Date'].dt.day
                df['DayOfWeek'] = df['Date'].dt.dayofweek
                df['Weekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
                
                return df

            @st.cache_data
            def aggregate_to_daily(df):
                """Aggregate data to daily values"""
                # Define columns to aggregate
                aqi_cols = ['AQI', 'PM10', 'PM2.5', 'NO2', 'NO', 'NOx']
                weather_cols = ['Temp', 'WD', 'WS']
                traffic_cols = ['Total_Pedestrians', 'City_Centre_TVCount', 'TrafficV']
                
                # Aggregate with appropriate functions
                daily_df = df.groupby('Date').agg({
                    **{col: 'mean' for col in aqi_cols},  # Mean for air quality
                    **{col: 'mean' for col in weather_cols},  # Mean for weather
                    **{col: 'sum' for col in traffic_cols},  # Sum for traffic counts
                    'Weekend': 'max'  # If any hour is weekend, the day is weekend
                }).reset_index()
                
                # Add time features
                daily_df['Year'] = daily_df['Date'].dt.year
                daily_df['Month'] = daily_df['Date'].dt.month
                daily_df['Day'] = daily_df['Date'].dt.day
                daily_df['DayOfWeek'] = daily_df['Date'].dt.dayofweek
                daily_df['DayName'] = daily_df['DayOfWeek'].apply(lambda x: ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][x])
                daily_df['MonthName'] = daily_df['Month'].apply(lambda x: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][x-1])
                
                return daily_df

            def filter_data(df, date_range, month_range, day_of_week, pollutant_threshold):
                filtered_df = df.copy()
                
                # Filter by date range
                if date_range:
                    filtered_df = filtered_df[(filtered_df['Date'] >= pd.Timestamp(date_range[0])) & 
                                            (filtered_df['Date'] <= pd.Timestamp(date_range[1]))]
                
                # Filter by month range
                if month_range and month_range != (1, 12):
                    filtered_df = filtered_df[(filtered_df['Month'] >= month_range[0]) & 
                                            (filtered_df['Month'] <= month_range[1])]
                
                # Filter by day of week
                if day_of_week and len(day_of_week) < 7:
                    filtered_df = filtered_df[filtered_df['DayOfWeek'].isin(day_of_week)]
                
                # Filter by pollutant threshold if specified
                if pollutant_threshold:
                    pollutant = pollutant_threshold['pollutant']
                    threshold = pollutant_threshold['threshold']
                    operation = pollutant_threshold['operation']
                    
                    if operation == '>':
                        filtered_df = filtered_df[filtered_df[pollutant] > threshold]
                    elif operation == '>=':
                        filtered_df = filtered_df[filtered_df[pollutant] >= threshold]
                    elif operation == '<':
                        filtered_df = filtered_df[filtered_df[pollutant] < threshold]
                    elif operation == '<=':
                        filtered_df = filtered_df[filtered_df[pollutant] <= threshold]
                
                return filtered_df

            def plot_time_series(df, variables, title):
                fig = go.Figure()
                
                for var in variables:
                    fig.add_trace(go.Scatter(
                        x=df['Date'],
                        y=df[var],
                        mode='lines+markers',
                        name=var
                    ))
                
                fig.update_layout(
                    title=title,
                    xaxis_title='Date',
                    yaxis_title='Value',
                    legend_title='Variables',
                    height=500
                )
                
                return fig

            def plot_seasonal_patterns(df, variable):
                # Group by month and calculate average
                monthly_avg = df.groupby('Month')[variable].mean().reset_index()
                
                fig = px.line(
                    monthly_avg, 
                    x='Month', 
                    y=variable,
                    markers=True,
                    title=f'Seasonal Pattern of {variable}'
                )
                
                fig.update_layout(
                    xaxis=dict(
                        tickmode='linear',
                        tick0=1,
                        dtick=1,
                        ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                        tickvals=list(range(1, 13))
                    ),
                    height=500
                )
                
                return fig

            def plot_correlation_heatmap(df, variables):
                corr_df = df[variables].corr()
                
                fig = px.imshow(
                    corr_df,
                    text_auto=True,
                    color_continuous_scale='RdBu_r',
                    aspect="auto"
                )
                
                fig.update_layout(
                    title='Correlation Matrix',
                    height=600,
                    width=700
                )
                
                return fig

            def plot_scatter_matrix(df, variables):
                fig = px.scatter_matrix(
                    df,
                    dimensions=variables,
                    color='PM10',  # Changed from AQI to PM10
                    opacity=0.5
                )
                
                fig.update_layout(
                    title='Scatter Matrix',
                    height=800,
                    width=800
                )
                
                return fig

            def create_wind_rose(df):
                # Create bins for wind direction
                bins = np.arange(0, 361, 45)
                labels = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
                df['WD_bin'] = pd.cut(df['WD'], bins=bins, labels=labels, include_lowest=True)
                
                # Create bins for wind speed
                ws_bins = [0, 2, 4, 6, 8, np.inf]
                ws_labels = ['0-2', '2-4', '4-6', '6-8', '>8']
                df['WS_bin'] = pd.cut(df['WS'], bins=ws_bins, labels=ws_labels)
                
                # Count occurrences in each bin
                wind_rose_data = df.groupby(['WD_bin', 'WS_bin']).size().unstack(fill_value=0)
                
                # Convert to percentages
                wind_rose_data = wind_rose_data.div(wind_rose_data.sum().sum()) * 100
                
                # Create the wind rose chart
                fig = go.Figure()
                
                for i, ws_label in enumerate(ws_labels):
                    if ws_label in wind_rose_data.columns:
                        fig.add_trace(go.Barpolar(
                            r=wind_rose_data[ws_label],
                            theta=wind_rose_data.index,
                            name=ws_label,
                            marker_color=px.colors.sequential.Plasma[i]
                        ))
                
                fig.update_layout(
                    title='Wind Rose',
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, wind_rose_data.values.max() * 1.1]
                        )
                    ),
                    showlegend=True,
                    legend_title='Wind Speed (m/s)'
                )
                
                return fig

            def plot_boxplots(df, variables):
                fig = make_subplots(rows=len(variables), cols=1, 
                                    subplot_titles=variables,
                                    vertical_spacing=0.05)
                
                for i, var in enumerate(variables):
                    fig.add_trace(
                        go.Box(y=df[var], name=var),
                        row=i+1, col=1
                    )
                
                fig.update_layout(
                    height=300 * len(variables),
                    title_text="Distribution of Air Quality Parameters",
                    showlegend=False
                )
                
                return fig

            def plot_dayofweek_analysis(df, variable):
                # Group by day of week and calculate mean
                dow_avg = df.groupby('DayName')[variable].mean().reset_index()
                
                # Ensure proper order of days
                day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                dow_avg['DayName'] = pd.Categorical(
                    dow_avg['DayName'],
                    categories=day_order,
                    ordered=True
                )
                dow_avg = dow_avg.sort_values('DayName')
                
                # Create the plot
                fig = px.bar(
                    dow_avg,
                    x='DayName',
                    y=variable,
                    title=f'Average {variable} by Day of Week'
                )
                
                return fig

            def plot_yearly_comparison(df, variable):
                """Plot yearly comparison for the selected variable"""
                # Group by year and month
                yearly_data = df.groupby(['Year', 'Month'])[variable].mean().reset_index()
                
                # Create the plot
                fig = px.line(
                    yearly_data,
                    x='Month',
                    y=variable,
                    color='Year',
                    markers=True,
                    title=f'Yearly Comparison of {variable} by Month'
                )
                
                fig.update_layout(
                    xaxis=dict(
                        tickmode='linear',
                        tick0=1,
                        dtick=1,
                        ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                        tickvals=list(range(1, 13))
                    ),
                    height=500
                )
                
                return fig

            def plot_traffic_vs_pollution(df, pollutant):
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                
                fig.add_trace(
                    go.Scatter(
                        x=df['Date'],
                        y=df[pollutant],
                        name=pollutant,
                        line=dict(color='red')
                    ),
                    secondary_y=False
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=df['Date'],
                        y=df['TrafficV'],
                        name='Vehicle Traffic',
                        line=dict(color='blue')
                    ),
                    secondary_y=True
                )
                
                fig.update_layout(
                    title=f'{pollutant} vs Traffic Volume',
                    xaxis_title='Date',
                    height=500
                )
                
                fig.update_yaxes(title_text=pollutant, secondary_y=False)
                fig.update_yaxes(title_text="Traffic Volume", secondary_y=True)
                
                return fig

            def detect_anomalies(df, variable, threshold=3):
                z_scores = stats.zscore(df[variable])
                outliers = abs(z_scores) > threshold
                
                fig = go.Figure()
                
                # Regular points
                fig.add_trace(go.Scatter(
                    x=df['Date'][~outliers],
                    y=df[variable][~outliers],
                    mode='markers',
                    marker=dict(color='blue', size=8),
                    name='Regular'
                ))
                
                # Outliers
                fig.add_trace(go.Scatter(
                    x=df['Date'][outliers],
                    y=df[variable][outliers],
                    mode='markers',
                    marker=dict(color='red', size=12),
                    name='Anomalies'
                ))
                
                fig.update_layout(
                    title=f'Anomaly Detection for {variable} (Z-score threshold: {threshold})',
                    xaxis_title='Date',
                    yaxis_title=variable,
                    height=500
                )
                
                return fig, df[outliers].shape[0] / df.shape[0] * 100

            # Define PM10 category function
            def pm10_category(pm10):
                if pm10 <= 20:
                    return 'Good'
                elif pm10 <= 40:
                    return 'Moderate'
                elif pm10 <= 50:
                    return 'Unhealthy for Sensitive Groups'
                elif pm10 <= 100:
                    return 'Unhealthy'
                elif pm10 <= 150:
                    return 'Very Unhealthy'
                else:
                    return 'Hazardous'

            # Main app
            def main():
                st.header("PM10 Air Quality Daily Analysis Dashboard")
                st.write("This dashboard analyzes daily aggregated PM10 air quality data to identify patterns and relationships.")
                
                st.sidebar.header("Dataset Loaded")
                st.sidebar.write('✅ No missing values found.')
                # uploaded_file = st.sidebar.file_uploader("Upload your air quality CSV data", type="csv")
                df = load_data()

                # Aggregate to daily
                daily_df = aggregate_to_daily(df)
                
                # Show basic dataset stats
                st.sidebar.subheader("Dataset Statistics")
                st.sidebar.write(f"Original records: {df.shape[0]}")
                st.sidebar.write(f"Daily records: {daily_df.shape[0]}")
                st.sidebar.write(f"Time range: {daily_df['Date'].min().date()} to {daily_df['Date'].max().date()}")
                st.sidebar.write(f"PM10 range: {daily_df['PM10'].min():.2f} to {daily_df['PM10'].max():.2f} µg/m³")
                
                # Data filtering options
                st.sidebar.subheader("Data Filtering")
                
                # Date range filter
                min_date = daily_df['Date'].min().date()
                max_date = daily_df['Date'].max().date()
                date_range = st.sidebar.date_input(
                    "Date range",
                    [min_date, max_date],
                    min_value=min_date,
                    max_value=max_date
                )
                
                # Month range filter
                month_range = st.sidebar.slider(
                    "Month range", 
                    1, 12, (1, 12)
                )
                
                # Day of week filter
                all_days = [0, 1, 2, 3, 4, 5, 6]
                day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                day_options = {day_names[i]: i for i in range(7)}
                selected_days = st.sidebar.multiselect(
                    "Days of week",
                    options=list(day_options.keys()),
                    default=list(day_options.keys())
                )
                selected_day_indices = [day_options[day] for day in selected_days] if selected_days else None
                
                # PM10 threshold filter
                pm10_threshold = st.sidebar.slider(
                    "PM10 threshold (µg/m³)", 
                    float(daily_df['PM10'].min()), 
                    float(daily_df['PM10'].max()), 
                    50.0  # WHO guideline for PM10
                )
                
                pollutant_threshold = {
                    'pollutant': 'PM10',
                    'operation': '>',
                    'threshold': pm10_threshold
                } if st.sidebar.checkbox("Filter by PM10 threshold", False) else None
                
                # Apply filters
                filtered_df = filter_data(
                    daily_df, 
                    date_range if len(date_range) == 2 else None,
                    month_range,
                    selected_day_indices,
                    pollutant_threshold
                )
                
                st.sidebar.write(f"Filtered daily records: {filtered_df.shape[0]}")
                
                # Main content
                tab1, tab2, tab3, tab4, tab5 = st.tabs([
                    "PM10 Overview", 
                    "Time Series", 
                    "Seasonal Patterns", 
                    "Traffic Impact",
                    "Anomaly Detection"
                ])
                
                # Tab 1: PM10 Overview
                with tab1:
                    st.header("PM10 Overview")
                    
                    # Data preview
                    st.subheader("Data Preview")
                    preview_cols = ['Date', 'PM10', 'PM2.5', 'NO2', 'Temp', 'WS', 'TrafficV', 'Total_Pedestrians', 'DayName']
                    st.dataframe(filtered_df[preview_cols].head(10))
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Summary statistics for PM10
                        st.subheader("PM10 Summary Statistics")
                        st.dataframe(filtered_df[['PM10']].describe())
                        
                        # PM10 histogram
                        st.subheader("PM10 Distribution")
                        pm10_hist = px.histogram(
                            filtered_df,
                            x='PM10',
                            nbins=20,
                            title='Distribution of Daily PM10 Values',
                            labels={'PM10': 'PM10 (µg/m³)'}
                        )
                        st.plotly_chart(pm10_hist, use_container_width=True)
                    
                    with col2:
                        # Data completeness
                        st.subheader("Data Completeness")
                        missing_data = pd.DataFrame({
                            'Missing Values': filtered_df[['PM10', 'PM2.5', 'NO2', 'NO', 'NOx', 'Temp', 'WS']].isnull().sum(),
                            'Percentage': filtered_df[['PM10', 'PM2.5', 'NO2', 'NO', 'NOx', 'Temp', 'WS']].isnull().sum() / len(filtered_df) * 100
                        })
                        st.dataframe(missing_data)
                        
                        # PM10 boxplot
                        st.subheader("PM10 Boxplot")
                        pm10_box = px.box(
                            filtered_df,
                            y='PM10',
                            title='Boxplot of Daily PM10 Values',
                            labels={'PM10': 'PM10 (µg/m³)'}
                        )
                        st.plotly_chart(pm10_box, use_container_width=True)
                    
                    # PM10 categories distribution
                    st.subheader("PM10 Categories Distribution")
                    
                    # Define PM10 categories based on WHO guidelines and add to dataframe
                    filtered_df['PM10_Category'] = filtered_df['PM10'].apply(pm10_category)
                    pm10_counts = filtered_df['PM10_Category'].value_counts().reset_index()
                    pm10_counts.columns = ['Category', 'Count']
                    
                    # Sort by PM10 severity
                    category_order = ['Good', 'Moderate', 'Unhealthy for Sensitive Groups', 'Unhealthy', 'Very Unhealthy', 'Hazardous']
                    pm10_counts['Category'] = pd.Categorical(
                        pm10_counts['Category'], 
                        categories=category_order, 
                        ordered=True
                    )
                    pm10_counts = pm10_counts.sort_values('Category')
                    
                    # Color mapping for PM10 categories
                    color_map = {
                        'Good': 'green',
                        'Moderate': 'yellow',
                        'Unhealthy for Sensitive Groups': 'orange',
                        'Unhealthy': 'red',
                        'Very Unhealthy': 'purple',
                        'Hazardous': 'maroon'
                    }
                    
                    # Create PM10 categories bar chart
                    fig = px.bar(
                        pm10_counts, 
                        x='Category', 
                        y='Count',
                        color='Category',
                        color_discrete_map=color_map,
                        title='Distribution of PM10 Categories',
                        labels={'Count': 'Number of Days'}
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show WHO guideline
                    st.info("WHO Air Quality Guideline: The 24-hour mean PM10 concentration should not exceed 50 µg/m³.")
                    
                    # Correlation analysis
                    st.subheader("Correlation with PM10")
                    
                    corr_vars = ['AQI','PM10', 'PM2.5', 'NO2', 'NO', 'NOx', 'Temp', 'WS', 'TrafficV', 'Total_Pedestrians']
                    corr_fig = plot_correlation_heatmap(filtered_df, corr_vars)
                    st.plotly_chart(corr_fig)
                    
                    # Show top correlations with PM10
                    correlations = filtered_df[corr_vars].corr()['PM10'].sort_values(ascending=False)
                    st.write("Top correlations with PM10:")
                    st.dataframe(correlations.drop('PM10'))
                    
                    # Distribution Analysis
                    st.subheader("Comparative Distribution Analysis")
                    
                    dist_vars = st.multiselect(
                        "Select variables to compare with PM10",
                        ["PM2.5", "NO2", "NO", "NOx", "Temp", "WS"],
                        default=["PM2.5", "NO2"]
                    )
                    
                    if dist_vars:
                        boxplot_fig = plot_boxplots(filtered_df, ['PM10'] + dist_vars)
                        st.plotly_chart(boxplot_fig, use_container_width=True)
                
                # Tab 2: Time Series
                with tab2:
                    st.header("PM10 Time Series Analysis")
                    
                    # PM10 time series
                    st.subheader("PM10 Daily Levels")
                    pm10_ts_fig = plot_time_series(
                        filtered_df, 
                        ['PM10'], 
                        "Daily PM10 Levels"
                    )
                    
                    # Add WHO guideline line
                    pm10_ts_fig.add_shape(
                        type="line",
                        x0=filtered_df['Date'].min(),
                        y0=50,  # WHO guideline
                        x1=filtered_df['Date'].max(),
                        y1=50,
                        line=dict(
                            color="green",
                            width=2,
                            dash="dash",
                        ),
                        name="WHO Guideline"
                    )
                    
                    # Add annotation for WHO guideline
                    pm10_ts_fig.add_annotation(
                        x=filtered_df['Date'].min() + pd.Timedelta(days=7),
                        y=52,
                        text="WHO Guideline (50 µg/m³)",
                        showarrow=False,
                        font=dict(color="green")
                    )
                    
                    st.plotly_chart(pm10_ts_fig, use_container_width=True)
                    
                    # Compare with other pollutants
                    st.subheader("Compare PM10 with Other Pollutants")
                    other_pollutants = st.multiselect(
                        "Select pollutants to compare with PM10",
                        ["PM2.5", "NO2", "NO", "NOx"],
                        default=["PM2.5"]
                    )
                    
                    if other_pollutants:
                        # Create comparison figure
                        compare_fig = go.Figure()
                        
                        # Add PM10
                        compare_fig.add_trace(go.Scatter(
                            x=filtered_df['Date'],
                            y=filtered_df['PM10'],
                            mode='lines+markers',
                            name='PM10',
                            line=dict(color='red', width=2)
                        ))
                        
                        # Add other pollutants
                        for pollutant in other_pollutants:
                            # Scale the pollutant to match PM10 range for better comparison
                            compare_fig.add_trace(go.Scatter(
                                x=filtered_df['Date'],
                                y=filtered_df[pollutant],
                                mode='lines+markers',
                                name=pollutant
                            ))
                        
                        compare_fig.update_layout(
                            title="PM10 vs Other Pollutants",
                            xaxis_title='Date',
                            yaxis_title='Concentration (µg/m³)',
                            legend_title='Pollutants',
                            height=500
                        )
                        
                        st.plotly_chart(compare_fig, use_container_width=True)
                    
                    # Weather factors
                    st.subheader("Weather Factors and PM10")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Temperature vs PM10 scatter plot
                        temp_fig = px.scatter(
                            filtered_df,
                            x='Temp',
                            y='PM10',
                            trendline='ols',
                            title='Temperature vs PM10',
                            labels={'Temp': 'Temperature (°C)', 'PM10': 'PM10 (µg/m³)'}
                        )
                        st.plotly_chart(temp_fig, use_container_width=True)
                    
                    with col2:
                        # Wind Speed vs PM10 scatter plot
                        ws_fig = px.scatter(
                            filtered_df,
                            x='WS',
                            y='PM10',
                            trendline='ols',
                            title='Wind Speed vs PM10',
                            labels={'WS': 'Wind Speed (m/s)', 'PM10': 'PM10 (µg/m³)'}
                        )
                        st.plotly_chart(ws_fig, use_container_width=True)
                    ## ROSE
                    weather_vars = st.multiselect(
                        "Select weather variables",
                        ["Temp", "WS", "WD"],
                        default=["Temp"]
                    )
                    
                    if weather_vars:
                        weather_fig = plot_time_series(
                            filtered_df, 
                            weather_vars, 
                            "Weather Factors Time Series"
                        )
                        st.plotly_chart(weather_fig, use_container_width=True)
                
                    # Wind Rose if WD and WS are available
                    if 'WD' in filtered_df.columns and 'WS' in filtered_df.columns:
                        st.subheader("Wind Rose")
                        wind_rose_fig = create_wind_rose(filtered_df)
                        st.plotly_chart(wind_rose_fig, use_container_width=True)

                    # Day of Week Analysis
                    st.subheader("PM10 by Day of Week")
                    dow_fig = plot_dayofweek_analysis(filtered_df, 'PM10')
                    st.plotly_chart(dow_fig, use_container_width=True)
                    
                    # Yearly comparison if multiple years
                    if len(filtered_df['Year'].unique()) > 1:
                        st.subheader("Yearly Comparison of PM10")
                        yearly_fig = plot_yearly_comparison(filtered_df, 'PM10')
                        st.plotly_chart(yearly_fig, use_container_width=True)
                


                # Tab 3: Seasonal Patterns
                with tab3:
                    st.header("PM10 Seasonal Patterns")
                    
                    # PM10 seasonal patterns
                    st.subheader("PM10 Monthly Pattern")
                    seasonal_fig = plot_seasonal_patterns(filtered_df, 'PM10')
                    st.plotly_chart(seasonal_fig, use_container_width=True)
                    
                    # Compare with temperature
                    st.subheader("PM10 vs Temperature by Month")
                    
                    # Group by month
                    monthly_avg = filtered_df.groupby('MonthName').agg({
                        'PM10': 'mean',
                        'Temp': 'mean'
                    }).reset_index()
                    
                    # Order by month
                    month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                    monthly_avg['MonthName'] = pd.Categorical(
                        monthly_avg['MonthName'],
                        categories=month_order,
                        ordered=True
                    )
                    monthly_avg = monthly_avg.sort_values('MonthName')
                    
                    # Dual axis plot
                    pm10_temp_fig = make_subplots(specs=[[{"secondary_y": True}]])
                    
                    pm10_temp_fig.add_trace(
                        go.Bar(
                            x=monthly_avg['MonthName'],
                            y=monthly_avg['PM10'],
                            name='PM10',
                            marker_color='red'
                        ),
                        secondary_y=False
                    )
                    
                    pm10_temp_fig.add_trace(
                        go.Scatter(
                            x=monthly_avg['MonthName'],
                            y=monthly_avg['Temp'],
                            name='Temperature',
                            marker_color='orange',
                            mode='lines+markers'
                        ),
                        secondary_y=True
                    )
                    
                    pm10_temp_fig.update_layout(
                        title='PM10 and Temperature by Month',
                        height=500
                    )
                    
                    pm10_temp_fig.update_yaxes(title_text="PM10 (µg/m³)", secondary_y=False)
                    pm10_temp_fig.update_yaxes(title_text="Temperature (°C)", secondary_y=True)
                    
                    st.plotly_chart(pm10_temp_fig, use_container_width=True)
                    
                    # Weekday vs Weekend analysis
                    st.subheader("PM10: Weekday vs Weekend Analysis")
                    
                    day_type = filtered_df.groupby('Weekend')['PM10'].mean().reset_index()
                    day_type['Weekend'] = day_type['Weekend'].replace({0: 'Weekday', 1: 'Weekend'})
                    
                    weekday_fig = px.bar(
                        day_type,
                        x='Weekend',
                        y='PM10',
                        color='Weekend',
                        title=f'Average PM10 on Weekdays vs Weekends',
                        labels={'PM10': 'PM10 (µg/m³)'}
                    )
                    
                    st.plotly_chart(weekday_fig, use_container_width=True)
                    
                    # PM10 by month and year heatmap
                    if len(filtered_df['Year'].unique()) > 1:
                        st.subheader("PM10 Heatmap by Month and Year")
                        
                        # Create pivot table
                        pm10_pivot = filtered_df.pivot_table(
                            index='Year',
                            columns='MonthName',
                            values='PM10',
                            aggfunc='mean'
                        )
                        
                        # Reorder columns by month
                        pm10_pivot = pm10_pivot[month_order]
                        
                        # Create heatmap
                        heatmap_fig = px.imshow(
                            pm10_pivot,
                            text_auto=True,
                            color_continuous_scale='Reds',
                            title='PM10 Levels by Month and Year',
                            labels={'color': 'PM10 (µg/m³)'}
                        )
                        
                        heatmap_fig.update_layout(
                            xaxis_title='Month',
                            yaxis_title='Year',
                            height=500
                        )
                        
                        st.plotly_chart(heatmap_fig, use_container_width=True)
                    
                    # Multi-variable relationships
                    st.subheader("PM10 Relationships with Other Variables")
                    
                    scatter_vars = st.multiselect(
                        "Select variables for scatter matrix",
                        ["PM10", "PM2.5", "NO2", "NO", "NOx", "Temp", "WS", "TrafficV"],
                        default=["PM10", "PM2.5", "NO2", "Temp"]
                    )
                    
                    if len(scatter_vars) > 1:
                        if 'PM10' not in scatter_vars:
                            scatter_vars = ['PM10'] + scatter_vars
                            
                        if len(scatter_vars) <= 6:  # Limit to 6 variables for readability
                            scatter_fig = plot_scatter_matrix(filtered_df, scatter_vars)
                            st.plotly_chart(scatter_fig)
                        else:
                            st.warning("Please select 5 or fewer variables (plus PM10) for the scatter matrix.")
                
                # Tab 4: Traffic Impact
                with tab4:
                    st.header("Traffic Impact on PM10")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Traffic vs PM10
                        st.subheader("Traffic vs PM10")
                        traffic_fig = plot_traffic_vs_pollution(filtered_df, 'PM10')
                        st.plotly_chart(traffic_fig, use_container_width=True)
                        
                        # Scatter plot
                        scatter_fig = px.scatter(
                            filtered_df,
                            x='TrafficV',
                            y='PM10',
                            trendline='ols',
                            title='Traffic Volume vs PM10',
                            labels={'TrafficV': 'Traffic Volume', 'PM10': 'PM10 (µg/m³)'}
                        )
                        st.plotly_chart(scatter_fig, use_container_width=True)
                    
                    with col2:
                        # Pedestrian vs PM10
                        st.subheader("Pedestrian Activity vs PM10")
                        
                        # Create dual y-axis plot
                        fig = make_subplots(specs=[[{"secondary_y": True}]])
                        
                        fig.add_trace(
                            go.Scatter(
                                x=filtered_df['Date'],
                                y=filtered_df['PM10'],
                                name='PM10',
                                line=dict(color='red')
                            ),
                            secondary_y=False
                        )
                        
                        fig.add_trace(
                            go.Scatter(
                                x=filtered_df['Date'],
                                y=filtered_df['Total_Pedestrians'],
                                name='Pedestrian Count',
                                line=dict(color='green')
                            ),
                            secondary_y=True
                        )
                        
                        fig.update_layout(
                            title='Daily PM10 vs Pedestrian Activity',
                            xaxis_title='Date',
                            height=500
                        )
                        
                        fig.update_yaxes(title_text="PM10 (µg/m³)", secondary_y=False)
                        fig.update_yaxes(title_text="Pedestrian Count", secondary_y=True)
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Scatter plot
                        ped_scatter_fig = px.scatter(
                            filtered_df,
                            x='Total_Pedestrians',
                            y='PM10',
                            trendline='ols',
                            title='Pedestrian Count vs PM10',
                            labels={'Total_Pedestrians': 'Pedestrian Count', 'PM10': 'PM10 (µg/m³)'}
                        )
                        st.plotly_chart(ped_scatter_fig, use_container_width=True)
                    
                    # Traffic-PM10 Correlation Analysis
                    st.subheader("Traffic-PM10 Correlation Analysis")
                    
                    traffic_corr_vars = ["PM10", "PM2.5", "NO2", "NO", "NOx", "TrafficV", "Total_Pedestrians"]
                    traffic_corr_fig = plot_correlation_heatmap(filtered_df, traffic_corr_vars)
                    st.plotly_chart(traffic_corr_fig)
                    
                    # Day of week traffic vs PM10
                    st.subheader("Day of Week: Traffic vs PM10")
                    
                    # Group by day of week
                    dow_traffic = filtered_df.groupby('DayName').agg({
                        'PM10': 'mean',
                        'TrafficV': 'mean',
                        'Total_Pedestrians': 'mean'
                    }).reset_index()
                    
                    # Order by day of week
                    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                    dow_traffic['DayName'] = pd.Categorical(
                        dow_traffic['DayName'],
                        categories=day_order,
                        ordered=True
                    )
                    dow_traffic = dow_traffic.sort_values('DayName')
                    
                    # Create a bar chart
                    fig = make_subplots(specs=[[{"secondary_y": True}]])
                    
                    fig.add_trace(
                        go.Bar(
                            x=dow_traffic['DayName'],
                            y=dow_traffic['PM10'],
                            name='PM10',
                            marker_color='red'
                        ),
                        secondary_y=False
                    )
                    
                    fig.add_trace(
                        go.Bar(
                            x=dow_traffic['DayName'],
                            y=dow_traffic['TrafficV'],
                            name='Traffic Volume',
                            marker_color='blue'
                        ),
                        secondary_y=True
                    )
                    
                    fig.update_layout(
                        title='Day of Week: PM10 vs Traffic Volume',
                        barmode='group',
                        height=500
                    )
                    
                    fig.update_yaxes(title_text="PM10 (µg/m³)", secondary_y=False)
                    fig.update_yaxes(title_text="Traffic Volume", secondary_y=True)
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Monthly traffic vs PM10
                    st.subheader("Monthly: Traffic vs PM10")
                    
                    # Group by month
                    monthly_traffic = filtered_df.groupby('MonthName').agg({
                        'PM10': 'mean',
                        'TrafficV': 'mean',
                        'Total_Pedestrians': 'mean'
                    }).reset_index()
                    
                    # Order by month
                    month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                    monthly_traffic['MonthName'] = pd.Categorical(
                        monthly_traffic['MonthName'],
                        categories=month_order,
                        ordered=True
                    )
                    monthly_traffic = monthly_traffic.sort_values('MonthName')
                    
                    # Create a bar chart
                    fig = make_subplots(specs=[[{"secondary_y": True}]])
                    
                    fig.add_trace(
                        go.Bar(
                            x=monthly_traffic['MonthName'],
                            y=monthly_traffic['PM10'],
                            name='PM10',
                            marker_color='red'
                        ),
                        secondary_y=False
                    )
                    
                    fig.add_trace(
                        go.Bar(
                            x=monthly_traffic['MonthName'],
                            y=monthly_traffic['TrafficV'],
                            name='Traffic Volume',
                            marker_color='blue'
                        ),
                        secondary_y=True
                    )
                    
                    fig.update_layout(
                        title='Monthly: PM10 vs Traffic Volume',
                        barmode='group',
                        height=500
                    )
                    
                    fig.update_yaxes(title_text="PM10 (µg/m³)", secondary_y=False)
                    fig.update_yaxes(title_text="Traffic Volume", secondary_y=True)
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # PM10 levels by traffic volume groups
                    st.subheader("PM10 Levels by Traffic Volume Categories")
                    if filtered_df['TrafficV'].nunique() <= 1:
                        print("Not enough variance in data to create meaningful bins.")
                    else:
                        # Create traffic volume categories
                        traffic_bins = [0, filtered_df['TrafficV'].quantile(0.25), 
                                    filtered_df['TrafficV'].quantile(0.5), 
                                    filtered_df['TrafficV'].quantile(0.75), 
                                    filtered_df['TrafficV'].max()]
                        traffic_labels = ['Low', 'Medium-Low', 'Medium-High', 'High']
                        
                        filtered_df['TrafficCategory'] = pd.cut(
                            filtered_df['TrafficV'], 
                            bins=traffic_bins, 
                            labels=traffic_labels,
                            duplicates='drop' #FAYE
                        )
                    
                        # Create boxplot
                        traffic_box_fig = px.box(
                            filtered_df, 
                            x='TrafficCategory', 
                            y='PM10',
                            color='TrafficCategory',
                            title='PM10 by Traffic Volume Category',
                            labels={'PM10': 'PM10 (µg/m³)', 'TrafficCategory': 'Traffic Volume Category'}
                        )
                        
                        st.plotly_chart(traffic_box_fig, use_container_width=True)
                        
                        # Traffic impact on PM10 exceedances
                        st.subheader("Traffic Impact on PM10 Exceedances")
                        
                        # Create exceedance variable
                        filtered_df['PM10_Exceedance'] = filtered_df['PM10'] > 50  # WHO guideline
                        
                        # Group by exceedance
                        exceedance_group = filtered_df.groupby('PM10_Exceedance').agg({
                            'TrafficV': 'mean',
                            'Total_Pedestrians': 'mean',
                            'Temp': 'mean',
                            'WS': 'mean'
                        }).reset_index()
                        
                        exceedance_group['PM10_Exceedance'] = exceedance_group['PM10_Exceedance'].map({True: 'Exceeds WHO Guideline', False: 'Below WHO Guideline'})
                        
                        # Create comparison bar chart
                        exceedance_fig = go.Figure()
                        
                        for col in ['TrafficV', 'Total_Pedestrians', 'Temp', 'WS']:
                            # Normalize for better visualization
                            max_val = exceedance_group[col].max()
                            exceedance_group[f'{col}_normalized'] = exceedance_group[col] / max_val * 100
                            
                            exceedance_fig.add_trace(go.Bar(
                                x=[col],
                                y=[exceedance_group[f'{col}_normalized'][0]],
                                name='Below WHO Guideline',
                                marker_color='green',
                                text=[f"{exceedance_group[col][0]:.1f}"],
                                textposition='outside'
                            ))
                            
                            exceedance_fig.add_trace(go.Bar(
                                x=[col],
                                y=[exceedance_group[f'{col}_normalized'][1]],
                                name='Exceeds WHO Guideline',
                                marker_color='red',
                                text=[f"{exceedance_group[col][1]:.1f}"],
                                textposition='outside'
                            ))
                        
                        exceedance_fig.update_layout(
                            title='Conditions on PM10 Exceedance vs Non-Exceedance Days',
                            barmode='group',
                            height=500,
                            yaxis_title='Normalized Value (%)'
                        )
                        
                        st.plotly_chart(exceedance_fig, use_container_width=True)
                
                # Tab 5: Anomaly Detection
                with tab5:
                    st.header("PM10 Anomaly Detection")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        anomaly_var = st.selectbox(
                            "Select variable for anomaly detection",
                            ["PM10", "PM2.5", "NO2", "NO", "NOx"],
                            index=0  # Default to PM10
                        )
                    
                    with col2:
                        z_threshold = st.slider(
                            "Z-score threshold", 
                            1.0, 5.0, 3.0, 0.1
                        )
                    
                    anomaly_fig, anomaly_percentage = detect_anomalies(filtered_df, anomaly_var, z_threshold)
                    st.plotly_chart(anomaly_fig, use_container_width=True)
                    st.write(f"Anomaly percentage: {anomaly_percentage:.2f}%")
                    
                    # Show statistics for anomalies
                    st.subheader("Anomaly Statistics")
                    
                    # Detect anomalies
                    z_scores = stats.zscore(filtered_df[anomaly_var])
                    outliers = abs(z_scores) > z_threshold
                    
                    if outliers.sum() > 0:
                        anomaly_df = filtered_df[outliers].copy()
                        
                        # Summary table
                        st.write("Anomaly Data Summary:")
                        st.dataframe(anomaly_df[[anomaly_var, 'Date', 'Temp', 'WS', 'TrafficV', 'Total_Pedestrians']].describe())
                        
                        # Day of week distribution of anomalies
                        st.write("Day of Week Distribution of Anomalies:")
                        
                        day_counts = anomaly_df['DayName'].value_counts().reset_index()
                        day_counts.columns = ['Day', 'Count']
                        
                        # Order by day of week
                        day_counts['Day'] = pd.Categorical(
                            day_counts['Day'],
                            categories=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
                            ordered=True
                        )
                        day_counts = day_counts.sort_values('Day')
                        
                        day_fig = px.bar(
                            day_counts,
                            x='Day',
                            y='Count',
                            title='Anomalies by Day of Week'
                        )
                        st.plotly_chart(day_fig, use_container_width=True)
                        
                        # Month distribution of anomalies
                        st.write("Month Distribution of Anomalies:")
                        
                        month_counts = anomaly_df['MonthName'].value_counts().reset_index()
                        month_counts.columns = ['Month', 'Count']
                        
                        # Order by month
                        month_counts['Month'] = pd.Categorical(
                            month_counts['Month'],
                            categories=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                            ordered=True
                        )
                        month_counts = month_counts.sort_values('Month')
                        
                        month_fig = px.bar(
                            month_counts,
                            x='Month',
                            y='Count',
                            title='Anomalies by Month'
                        )
                        st.plotly_chart(month_fig, use_container_width=True)
                        
                        # Year distribution of anomalies if multiple years
                        if len(anomaly_df['Year'].unique()) > 1:
                            st.write("Year Distribution of Anomalies:")
                            
                            year_counts = anomaly_df['Year'].value_counts().reset_index()
                            year_counts.columns = ['Year', 'Count']
                            year_counts = year_counts.sort_values('Year')
                            
                            year_fig = px.bar(
                                year_counts,
                                x='Year',
                                y='Count',
                                title='Anomalies by Year'
                            )
                            st.plotly_chart(year_fig, use_container_width=True)
                        
                        # Traffic and weather on anomaly days
                        st.subheader("Conditions on Anomaly Days vs. Normal Days")
                        
                        # Create comparison dataframe
                        comparison_cols = ['PM10', 'TrafficV', 'Total_Pedestrians', 'Temp', 'WS']
                        anomaly_means = anomaly_df[comparison_cols].mean()
                        normal_means = filtered_df[~outliers][comparison_cols].mean()
                        
                        comparison_df = pd.DataFrame({
                            'Variable': comparison_cols,
                            'Anomaly Days': anomaly_means.values,
                            'Normal Days': normal_means.values
                        })
                        
                        # Calculate percentage difference
                        comparison_df['% Difference'] = ((comparison_df['Anomaly Days'] / comparison_df['Normal Days'] - 1) * 100).round(1)
                        
                        # Create comparison table
                        st.write("Average Conditions Comparison:")
                        st.dataframe(comparison_df)
                        
                        # Create bar chart for comparison
                        fig = go.Figure()
                        
                        for col in comparison_cols:
                            # Normalize values for better comparison
                            max_val = max(anomaly_means[col], normal_means[col])
                            normalized_anomaly = anomaly_means[col] / max_val * 100
                            normalized_normal = normal_means[col] / max_val * 100
                            
                            fig.add_trace(go.Bar(
                                x=[col],
                                y=[normalized_anomaly],
                                name='Anomaly Days',
                                marker_color='red',
                                text=[f"{anomaly_means[col]:.1f}"],
                                textposition='outside'
                            ))
                            
                            fig.add_trace(go.Bar(
                                x=[col],
                                y=[normalized_normal],
                                name='Normal Days',
                                marker_color='blue',
                                text=[f"{normal_means[col]:.1f}"],
                                textposition='outside'
                            ))
                        
                        fig.update_layout(
                            title='Conditions on Anomaly Days vs. Normal Days (Normalized)',
                            barmode='group',
                            height=500,
                            yaxis_title='Normalized Value (%)'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Download anomaly data
                        csv = anomaly_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "Download Anomaly Data",
                            csv,
                            "anomaly_data.csv",
                            "text/csv",
                            key='download-anomaly-csv'
                        )
                    else:
                        st.write("No anomalies detected with the current threshold.")
                    
                    # PM10 WHO Exceedance Analysis
                    st.subheader("PM10 WHO Guideline Exceedance Analysis")
                    
                    # Calculate exceedances
                    exceedances = filtered_df[filtered_df['PM10'] > 50]  # WHO guideline
                    exceedance_percentage = len(exceedances) / len(filtered_df) * 100
                    
                    st.write(f"WHO Guideline: 50 µg/m³")
                    st.write(f"Exceedances: {len(exceedances)} out of {len(filtered_df)} days ({exceedance_percentage:.2f}%)")
                    
                    # Plot exceedances over time
                    if len(exceedances) > 0:
                        exceedance_fig = go.Figure()
                        
                        # All data
                        exceedance_fig.add_trace(go.Scatter(
                            x=filtered_df['Date'],
                            y=filtered_df['PM10'],
                            mode='lines+markers',
                            name='PM10',
                            line=dict(color='blue', width=1)
                        ))
                        
                        # Exceedances
                        exceedance_fig.add_trace(go.Scatter(
                            x=exceedances['Date'],
                            y=exceedances['PM10'],
                            mode='markers',
                            name='Exceedances',
                            marker=dict(color='red', size=10)
                        ))
                        
                        # WHO guideline line
                        exceedance_fig.add_shape(
                            type="line",
                            x0=filtered_df['Date'].min(),
                            y0=50,  # WHO guideline
                            x1=filtered_df['Date'].max(),
                            y1=50,
                            line=dict(
                                color="green",
                                width=2,
                                dash="dash",
                            )
                        )
                        
                        exceedance_fig.update_layout(
                            title='PM10 WHO Guideline Exceedances (> 50 µg/m³)',
                            xaxis_title='Date',
                            yaxis_title='PM10 (µg/m³)',
                            height=500
                        )
                        
                        st.plotly_chart(exceedance_fig, use_container_width=True)
                        
                        # Exceedance patterns
                        st.write("Exceedance Patterns:")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # By day of week
                            dow_counts = exceedances['DayName'].value_counts().reset_index()
                            dow_counts.columns = ['Day', 'Count']
                            
                            # Calculate percentage by day
                            total_by_day = filtered_df.groupby('DayName').size().reset_index(name='Total')
                            dow_counts = dow_counts.merge(total_by_day, left_on='Day', right_on='DayName', how='left')
                            dow_counts['Percentage'] = (dow_counts['Count'] / dow_counts['Total'] * 100).round(2)
                            
                            # Order by day of week
                            dow_counts['Day'] = pd.Categorical(
                                dow_counts['Day'],
                                categories=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
                                ordered=True
                            )
                            dow_counts = dow_counts.sort_values('Day')
                            
                            dow_ex_fig = px.bar(
                                dow_counts,
                                x='Day',
                                y='Percentage',
                                title='Percentage of PM10 Exceedances by Day of Week',
                                text=dow_counts['Percentage'].apply(lambda x: f'{x:.1f}%')
                            )
                            
                            dow_ex_fig.update_traces(textposition='outside')
                            
                            st.plotly_chart(dow_ex_fig, use_container_width=True)
                        
                        with col2:
                            # By month
                            month_counts = exceedances['MonthName'].value_counts().reset_index()
                            month_counts.columns = ['Month', 'Count']
                            
                            # Calculate percentage by month
                            total_by_month = filtered_df.groupby('MonthName').size().reset_index(name='Total')
                            month_counts = month_counts.merge(total_by_month, left_on='Month', right_on='MonthName', how='left')
                            month_counts['Percentage'] = (month_counts['Count'] / month_counts['Total'] * 100).round(2)
                            
                            # Order by month
                            month_counts['Month'] = pd.Categorical(
                                month_counts['Month'],
                                categories=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                                ordered=True
                            )
                            month_counts = month_counts.sort_values('Month')
                            
                            month_ex_fig = px.bar(
                                month_counts,
                                x='Month',
                                y='Percentage',
                                title='Percentage of PM10 Exceedances by Month',
                                text=month_counts['Percentage'].apply(lambda x: f'{x:.1f}%')
                            )
                            
                            month_ex_fig.update_traces(textposition='outside')
                            
                            st.plotly_chart(month_ex_fig, use_container_width=True)
                        
                        # Download exceedance data
                        csv = exceedances.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "Download WHO Exceedance Data",
                            csv,
                            "pm10_who_exceedances.csv",
                            "text/csv",
                            key='download-exceedance-csv'
                        )
                    else:
                        st.write("No WHO guideline exceedances found in the filtered data.")                

            if __name__ == "__main__":
                main()
    else:
        st.error('⛔ [ATTENTION] Please confirm your data in the main page to proceed.')
else:
    st.markdown('⛔ [ATTENTION] No file found. Please upload and process file/s in the main page to access this module.')

