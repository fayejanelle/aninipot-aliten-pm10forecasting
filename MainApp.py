import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, time
import io
import openpyxl
from collections import defaultdict
from PIL import Image
from io import BytesIO
import base64
import os
from dotenv import load_dotenv

st.set_page_config(page_title="Main - PM10 Forecasting App", layout="wide")


combined_image = Image.open('data/combined_NZSE_AC_logo.png')
# Create two columns
col1, col2, col3 = st.columns([1.5, 1, 1.5])

with col2:
    st.image(combined_image,  use_container_width=True)
    

# st.title("A Collaborative Capstone Project of NZSE and Auckland Council")
st.markdown(
    "<h1 style='text-align: center;'>A Collaborative Capstone Project of NZSE and Auckland Council</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<h3 style='text-align: center;'>By Faye Janelle A. Aninipot & Janice A. Aliten (GDDA7224C)</h>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='font-size:24px; font-style: italic; text-align: center;'>Supervised by Dr. Sara Zandi (NZSE) & Dr. Louis Boamponsem (Auckland Council)</p>",
    unsafe_allow_html=True
)

# Initialize session state
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'year_data' not in st.session_state:
    st.session_state.year_data = {}
if 'confirmed_data' not in st.session_state:
    st.session_state.confirmed_data = None
if 'data_confirmed' not in st.session_state:
    st.session_state.data_confirmed = False
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
    # st.session_state['uploaded_file'] = 'no'
if 'selected_year_range' not in st.session_state:
    st.session_state.selected_year_range = None
if 'year_selection_confirmed' not in st.session_state:
    st.session_state.year_selection_confirmed = False

# Define your secret password (in production, store this securely!)
load_dotenv()
CORRECT_PASSWORD = os.getenv("STREAMLIT_PASSWORD")

# Use Streamlit session state to store authentication status
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

def is_hourly_time(time_val):
    """Check if time is in hourly format (minutes = 00)"""
    try:
        if isinstance(time_val, (datetime, pd.Timestamp)):
            return time_val.minute == 0
        elif isinstance(time_val, time):
            return time_val.minute == 0
        elif isinstance(time_val, str):
            # Parse time string
            time_obj = pd.to_datetime(time_val, format='%H:%M:%S', errors='coerce')
            if pd.isna(time_obj):
                time_obj = pd.to_datetime(time_val, errors='coerce')
            return time_obj.minute == 0 if not pd.isna(time_obj) else False
        return False
    except:
        return False

def process_excel_files(uploaded_files):
    """Process multiple Excel files and combine them"""
    all_data = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, file in enumerate(uploaded_files):
        status_text.text(f"Processing {file.name}...")
        
        try:
            # Read Excel file
            df = pd.read_excel(file, engine='openpyxl')
            
            # Ensure columns are named correctly
            expected_cols = ['Site', 'Date', 'Time', 'Parameter', 'Value']
            if len(df.columns) >= 5:
                df.columns = expected_cols[:len(df.columns)] + list(df.columns[5:])
            
            # Convert date and time columns
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            
            # Handle time column - Excel sometimes stores times as datetime
            if df['Time'].dtype == 'object':
                df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S', errors='coerce').dt.time
            else:
                # If it's already datetime, extract time
                df['Time'] = pd.to_datetime(df['Time'], errors='coerce').dt.time
            
            # Filter for hourly times only
            hourly_mask = df['Time'].apply(is_hourly_time)
            df_hourly = df[hourly_mask].copy()
            
            # Create datetime column
            df_hourly['DateTime'] = pd.to_datetime(
                df_hourly['Date'].dt.strftime('%Y-%m-%d') + ' ' + 
                df_hourly['Time'].astype(str)
            )
            
            # Add to collection
            all_data.append(df_hourly)
            
            st.info(f"Processed {file.name}: {len(df_hourly)} hourly records out of {len(df)} total")
            
        except Exception as e:
            st.error(f"Error processing {file.name}: {str(e)}")
        
        progress_bar.progress((idx + 1) / len(uploaded_files))
    
    status_text.text("Combining all data...")
    
    if all_data:
        # Combine all dataframes
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Pivot the data
        pivoted = combined_df.pivot_table(
            index=['DateTime', 'Site'],
            columns='Parameter',
            values='Value',
            aggfunc='mean'  # Average if there are duplicates
        ).reset_index()
        
        # Separate date and time
        pivoted['Date'] = pivoted['DateTime'].dt.date
        pivoted['Time'] = pivoted['DateTime'].dt.time
        
        # Reorder columns
        base_cols = ['Date', 'Time']
        param_cols = [col for col in pivoted.columns if col not in ['Date', 'Time', 'DateTime', 'Site']]
        final_cols = base_cols + param_cols
        
        # Add placeholder columns for any missing parameters
        all_possible_cols = ['Date', 'Time', 'AQI', 'PM10', 'PM2.5', 'NO2', 'NO', 'NOx', 
                            'Temp', 'WD', 'WS', 'Total_Pedestrians', 'City_Centre_TVCount', 'TrafficV']
        
        for col in all_possible_cols:
            if col not in pivoted.columns and col not in ['Date', 'Time']:
                pivoted[col] = np.nan
        
        # Select and order columns
        available_cols = [col for col in all_possible_cols if col in pivoted.columns]
        final_df = pivoted[available_cols].copy()
        
        # Sort by datetime
        final_df = final_df.sort_values(['Date', 'Time']).reset_index(drop=True)
        
        # Group by year for selection
        final_df['Year'] = pd.to_datetime(final_df['Date']).dt.year
        year_groups = final_df.groupby('Year')
        
        year_data = {}
        for year, group in year_groups:
            year_data[year] = {
                'data': group.drop('Year', axis=1),
                'count': len(group),
                'start_date': group['Date'].min(),
                'end_date': group['Date'].max()
            }
        
        status_text.text("Processing complete!")
        progress_bar.empty()
        
        return final_df.drop('Year', axis=1), year_data
    
    return None, {}

# Show password input only if not authenticated
if not st.session_state.authenticated:
    st.session_state['uploaded_file'] = 'no' #F
    st.title("🔐 Secure App Access")
    password = st.text_input("Enter password:", type="password")
    if st.button("Login"):
        if password == CORRECT_PASSWORD:
            st.session_state.authenticated = True
            st.rerun()  # This will refresh the page and hide the login section
        else:
            st.error("Incorrect password. Try again.")
else:
    # Show app content if authenticated - login section is completely hidden
    st.header("🌏 Welcome to the PM10 Forecasting App")
    
    # Optional: Add a logout button
    if st.button("Logout"):
        st.session_state.authenticated = False
        st.rerun()

# # Show app content if authenticated
if 'show_instructions' not in st.session_state:
    st.session_state.show_instructions = False

if st.session_state.authenticated:

    # Instructions expander in main page
    with st.expander("📖 Instructions", expanded=st.session_state.show_instructions):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### How to use:
            
            1. **Upload Files**: Upload one or more Excel files with the following format:
               - Site: Location identifier
               - Date: Date of measurement
               - Time: Time of measurement
               - Parameter: Type of measurement (NO, NO2, PM10, etc.)
               - Value: Measured value
            
            2. **Process**: Click "Process Files" to transform the data
               - Only hourly measurements (XX:00:00) will be kept
               - Data will be pivoted with parameters as columns
            
            3. **Select Periods**: Choose which years to include in the final dataset
               - View data availability percentage based on expected hourly data
               - Use slider to select consecutive years
               - Click "Apply Year Selection" to confirm your range
               - Confirm your selection before proceeding
            
            4. **Download**: Export the transformed data as CSV
            """)
            
            st.markdown("""
            ### Output Format:
            The transformed data will have columns:
            - Date, Time
            - AQI, PM10, PM2.5, NO2, NO, NOx
            - Temp, WD, WS
            - Total_Pedestrians, City_Centre_TVCount, TrafficV
            
            Missing parameters will have NaN values.
            """)
        
        with col2:
            st.markdown("""
            ### Data Availability Metrics
            
            The availability percentages are calculated based on:
            - **Expected hours per year**: 8,760 (non-leap) or 8,784 (leap)
            - **Actual date range**: Calculated from your data
            - **Percentage**: (Non-null values / Expected hours) × 100
            
            This gives you a true picture of data completeness for time-series analysis.
            """)
            
            st.markdown("""
            ### Sample Data Structure
            
            **Input (Raw Excel)**:
            ```
            Site | Date     | Time  | Parameter | Value
            QUS  | 7/1/2022 | 1:00  | NO       | 5.0
            QUS  | 7/1/2022 | 1:00  | NO2      | 12.3
            ```
            
            **Output (Time-Series)**:
            ```
            Date       | Time     | NO  | NO2  | ...
            2022-07-01 | 01:00:00 | 5.0 | 12.3 | ...
            ```
            """)

    # File uploader
    st.sidebar.header("Raw Data Uploader")
    new_files = st.sidebar.file_uploader(
        "Choose Excel files to upload",
        type=['xlsx', 'xls'],
        accept_multiple_files=True,
        help="Upload one or more Excel files in the format: Site, Date, Time, Parameter, Value"
    )

    # Update uploaded files list
    if new_files:
        # Add new files to session state
        existing_names = [f.name for f in st.session_state.uploaded_files]
        for file in new_files:
            if file.name not in existing_names:
                st.session_state.uploaded_files.append(file)

    # Display uploaded files with remove option
    if st.session_state.uploaded_files:
        st.header("1. Uploaded Raw Data")
        st.info(f"**✅ {len(st.session_state.uploaded_files)} Files Successfully Uploaded**")
        files_to_remove = []
        
        for idx, file in enumerate(st.session_state.uploaded_files):
            col1, col2 = st.columns([4, 1])
            with col1:
                st.text(f"📄 {file.name}")
            with col2:
                if st.button("Remove", key=f"remove_{idx}"):
                    files_to_remove.append(idx)
        
        # Remove files marked for deletion
        for idx in reversed(files_to_remove):
            st.session_state.uploaded_files.pop(idx)
            # Reset processing state if files are removed
            st.session_state.data_confirmed = False
            st.session_state.year_selection_confirmed = False
            st.rerun()

    if st.session_state.uploaded_files:
        if st.button("Process Files", type="primary"):
            with st.spinner("Processing files..."):
                processed_data, year_data = process_excel_files(st.session_state.uploaded_files)
                
                if processed_data is not None:
                    st.session_state.processed_data = processed_data
                    st.session_state.year_data = year_data
                    st.session_state.data_confirmed = False  # Reset confirmation
                    st.session_state.confirmed_data = None
                    st.session_state.year_selection_confirmed = False  # Reset year selection
                    st.success(f"Successfully processed {len(processed_data)} hourly records")
                    st.session_state['uploaded_file'] = 'yes' #F
                    st.session_state['imputed'] = 'no' #F
    else:
        st.session_state['uploaded_file'] = 'no' #F
        st.info("✅ You now have access to the app content. Please start by uploading your files using the Raw Data Uploader in the app's side bar and follow the instructions above 👆.")

    # Year selection section
    if st.session_state.year_data:
        st.header("2. Select Time Periods")
        
        # Calculate parameter statistics for each year
        parameter_stats = {}
        param_columns = ['AQI', 'PM10', 'PM2.5', 'NO2', 'NO', 'NOx', 'Temp', 'WD', 'WS', 
                         'Total_Pedestrians', 'City_Centre_TVCount', 'TrafficV']
        
        # Function to check if year is leap year
        def is_leap_year(year):
            return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
        
        for year, info in st.session_state.year_data.items():
            # Calculate expected hours for the year
            expected_hours = 8784 if is_leap_year(year) else 8760
            
            # Calculate actual date range in the data
            date_range = pd.to_datetime(info['data']['Date'])
            actual_start = date_range.min()
            actual_end = date_range.max()
            
            # Calculate expected hours for the actual date range
            days_in_range = (actual_end - actual_start).days + 1
            expected_hours_in_range = days_in_range * 24
            
            year_stats = {}
            year_stats['expected_hours'] = expected_hours
            year_stats['expected_hours_in_range'] = expected_hours_in_range
            year_stats['actual_records'] = len(info['data'])
            year_stats['date_range'] = f"{actual_start.strftime('%Y-%m-%d')} to {actual_end.strftime('%Y-%m-%d')}"
            
            for param in param_columns:
                if param in info['data'].columns:
                    non_null_count = info['data'][param].notna().sum()
                    # Calculate percentage based on expected hours in the actual date range
                    percentage = (non_null_count / expected_hours_in_range * 100) if expected_hours_in_range > 0 else 0
                    year_stats[param] = {
                        'count': non_null_count,
                        'percentage': percentage,
                        'expected': expected_hours_in_range
                    }
                else:
                    year_stats[param] = {'count': 0, 'percentage': 0, 'expected': expected_hours_in_range}
            parameter_stats[year] = year_stats
        
        # Display data availability
        st.subheader("Data Availability by Year and Parameter")
        
        years_sorted = sorted(st.session_state.year_data.keys())
        
        st.write("**Parameter Availability (% of expected hourly data):**")
        
        # Create a summary dataframe for visualization
        availability_data = []
        
        for param in param_columns:
            row = {'Parameter': param}
            for year in years_sorted:
                if year in parameter_stats and param in parameter_stats[year]:
                    count = parameter_stats[year][param]['count']
                    pct = parameter_stats[year][param]['percentage']
                    expected = parameter_stats[year][param]['expected']
                    row[f'{year}'] = f"{count:,}/{expected:,} ({pct:.1f}%)"
                else:
                    row[f'{year}'] = "0 (0.0%)"
            availability_data.append(row)
        
        availability_df = pd.DataFrame(availability_data)
        
        # Style the dataframe with frozen first column
        def color_availability(val):
            if isinstance(val, str) and '(' in val:
                pct = float(val.split('(')[1].split('%')[0])
                if pct >= 80:
                    return 'background-color: #4169E1'  # Blue
                elif pct >= 50:
                    return 'background-color: #800080'  # Purple
                elif pct > 0:
                    return 'background-color: #FF8C00'  # Orange
                else:
                    return 'background-color: #696969'  # Gray
            return ''
        
        # Create a styled dataframe with frozen Parameter column
        styled_df = availability_df.style.applymap(
            color_availability, 
            subset=[str(y) for y in years_sorted]
        ).set_sticky(axis="columns")
        
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
        
        # Legend
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown("🟦 **≥80% available**")
        with col2:
            st.markdown("🟪 **50-79% available**")
        with col3:
            st.markdown("🟧 **1-49% available**")
        with col4:
            st.markdown("🔳 **No data**")
        
        st.divider()
        
        # Year selection with slider
        st.subheader("Select Consecutive Years for Time Series")
        
        if len(years_sorted) > 1:
            # Initialize year range if not set
            if st.session_state.selected_year_range is None:
                st.session_state.selected_year_range = (min(years_sorted), max(years_sorted))
            
            # Use a slider for selecting a range of consecutive years
            year_range = st.slider(
                "Select year range:",
                min_value=min(years_sorted),
                max_value=max(years_sorted),
                value=st.session_state.selected_year_range,
                step=1,
                key="year_slider"
            )
            
            # Check if slider has changed
            slider_changed = year_range != st.session_state.selected_year_range
            
            if slider_changed:
                st.session_state.year_selection_confirmed = False
            
            # Show preview of selection
            preview_years = [year for year in years_sorted if year_range[0] <= year <= year_range[1]]
            st.info(f"**Preview**: {year_range[0]} to {year_range[1]} ({len(preview_years)} years)")
            
            # Show total records for preview period
            preview_records = sum(st.session_state.year_data[year]['count'] for year in preview_years)
            st.metric("Total records in preview period", f"{preview_records:,}")
            
            # Apply button
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("Apply Year Selection", type="primary", use_container_width=True, 
                            disabled=st.session_state.year_selection_confirmed and not slider_changed):
                    st.session_state.selected_year_range = year_range
                    st.session_state.year_selection_confirmed = True
                    st.success("✅ Year selection applied!")
            
            # Show selected years only after confirmation
            if st.session_state.year_selection_confirmed:
                selected_years = [year for year in years_sorted if 
                                st.session_state.selected_year_range[0] <= year <= st.session_state.selected_year_range[1]]
                st.info(f"**Selected years confirmed**: {st.session_state.selected_year_range[0]} to {st.session_state.selected_year_range[1]}")
            else:
                selected_years = None
                
        else:
            # Only one year available
            selected_years = years_sorted
            st.session_state.year_selection_confirmed = True
            st.info(f"**Available year**: {years_sorted[0]}")
            st.metric("Total records", f"{st.session_state.year_data[years_sorted[0]]['count']:,}")
        
        # Confirmation section - only show if years are selected
        if st.session_state.year_selection_confirmed and selected_years:
            st.divider()
            st.subheader("Confirm Data Selection")
            
            # Show parameter summary for selected years
            combined_stats = {}
            total_expected_hours = 0
            
            for year in selected_years:
                total_expected_hours += parameter_stats[year]['expected_hours_in_range']
            
            for param in param_columns:
                total_count = 0
                for year in selected_years:
                    if year in parameter_stats and param in parameter_stats[year]:
                        total_count += parameter_stats[year][param]['count']
                
                if total_expected_hours > 0:
                    combined_stats[param] = {
                        'count': total_count,
                        'percentage': (total_count / total_expected_hours * 100),
                        'expected': total_expected_hours
                    }
            
            # Display combined statistics
            st.write("**Parameter availability in selected period:**")
            
            col1, col2 = st.columns(2)
            params_split = len(param_columns) // 2
            
            with col1:
                for param in param_columns[:params_split]:
                    if param in combined_stats:
                        count = combined_stats[param]['count']
                        pct = combined_stats[param]['percentage']
                        expected = combined_stats[param]['expected']
                        if pct >= 80:
                            st.success(f"{param}: {count:,}/{expected:,} ({pct:.1f}%)")
                        elif pct >= 50:
                            st.warning(f"{param}: {count:,}/{expected:,} ({pct:.1f}%)")
                        elif pct > 0:
                            st.info(f"{param}: {count:,}/{expected:,} ({pct:.1f}%)")
                        else:
                            st.error(f"{param}: No data available")
            
            with col2:
                for param in param_columns[params_split:]:
                    if param in combined_stats:
                        count = combined_stats[param]['count']
                        pct = combined_stats[param]['percentage']
                        expected = combined_stats[param]['expected']
                        if pct >= 80:
                            st.success(f"{param}: {count:,}/{expected:,} ({pct:.1f}%)")
                        elif pct >= 50:
                            st.warning(f"{param}: {count:,}/{expected:,} ({pct:.1f}%)")
                        elif pct > 0:
                            st.info(f"{param}: {count:,}/{expected:,} ({pct:.1f}%)")
                        else:
                            st.error(f"{param}: No data available")
            
            # Confirmation button
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                confirm_selection = st.button(
                    "✅ Confirm and Use This Data", 
                    type="primary",
                    use_container_width=True,
                    help="Click to confirm your selection and proceed to download"
                )
            
            # Process confirmed selection
            if confirm_selection:
                selected_data = []
                for year in selected_years:
                    selected_data.append(st.session_state.year_data[year]['data'])
                
                final_data = pd.concat(selected_data, ignore_index=True)
                st.session_state.confirmed_data = final_data
                st.session_state['df'] = final_data #F
                st.session_state.data_confirmed = True
                st.info("✅ Data selection confirmed! Proceed to preview and download below.")

    # Show preview and download section only after confirmation
    if st.session_state.data_confirmed and st.session_state.confirmed_data is not None:
        st.header("3. Preview and Download")
        final_data = st.session_state.confirmed_data
        
        # Show preview
        st.subheader("Data Preview")
        st.write(f"Total records: {len(final_data)}")
        st.dataframe(final_data.head(100))
        
        # Show column info
        with st.expander("Column Information"):
            col_info = pd.DataFrame({
                'Column': final_data.columns,
                'Non-null Count': final_data.count(),
                'Null Count': final_data.isnull().sum(),
                'Data Type': final_data.dtypes
            })
            st.dataframe(col_info)

        st.success("✅ The confirmed data is successfully saved as the original dataset. You may proceed to use the other available modules of this app.")
        # Download section
        st.subheader("Download Confirmed Data (Original Dataset)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # CSV download
            csv_buffer = io.StringIO()
            final_data.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()
            
            st.download_button(
                label="Download as CSV",
                data=csv_data,
                file_name=f"transformed_timeseries_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                type="primary"
            )
        
        # with col2:
        #     # Excel download
        #     excel_buffer = io.BytesIO()
        #     with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
        #         final_data.to_excel(writer, sheet_name='TimeSeries', index=False)
        #     excel_data = excel_buffer.getvalue()
            
        #     st.download_button(
        #         label="Download as Excel",
        #         data=excel_data,
        #         file_name=f"transformed_timeseries_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
        #         mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        #     )
# FAYE 