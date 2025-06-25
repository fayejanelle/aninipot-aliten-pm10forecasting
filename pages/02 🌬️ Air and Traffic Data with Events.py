import streamlit as st
import pandas as pd
import plotly.graph_objects as go

if 'uploaded_file' not in st.session_state:
        st.session_state['uploaded_file'] = 'no'

uploaded_file = st.session_state['uploaded_file']

if(uploaded_file == 'yes'):
    if st.session_state.data_confirmed:
        df = st.session_state['df']
        st.info('⚪ Original dataset is loaded for this page.')
        ### START HERE
        def events(dataframe):
        
            try:
               
                # Use the passed dataframe
                df = dataframe.copy()   
                
                # Import numpy for wind direction calculations
                import numpy as np  
                
                # Ensure Date column exists and is in datetime format
                if 'Date' not in df.columns:
                    st.error("Required column 'Date' is missing from the dataframe")
                    return
                
                # First ensure 'Time' column exists before dropping
                columns_to_drop = ['Time'] if 'Time' in df.columns else []
                
                # Create daily averages
                daily = df.drop(columns=columns_to_drop).groupby('Date').agg('mean').reset_index()
                
                # Ensure Date is in datetime format
                if not pd.api.types.is_datetime64_any_dtype(daily['Date']):
                    daily["Date"] = pd.to_datetime(daily["Date"])
                
                # Add Year column
                daily["Year"] = daily["Date"].dt.year
                
                # Initialize session state events dataframe if not exists
                if 'events_df' not in st.session_state:
                    # Predefined annotated events
                    default_events_data = {
                        "Date": [
                            "2018-01-04", "2018-02-01", "2018-02-20", "2018-04-10", "2018-06-03",
                            "2019-10-22", "2019-12-06", "2020-01-04", "2020-03-26", "2020-04-28",
                            "2021-08-18", "2021-11-10", "2021-12-03"
                        ],
                        "Event": [
                            "North Island Storm", "Cyclone Fehi", "Cyclone Gita", "Auckland Storm", 
                            "Queen's Birthday Storm", "NZICC Fire", "Aussie Dust Arrival", "Orange Sky Event",
                            "COVID Lockdown Start", "COVID Lockdown Ends", "Delta Lockdown Start", 
                            "Lockdown Ease", "Traffic Light System Begins"
                        ],
                        "Type": ["Default"] * 13  # Track event source
                    }
                    
                    # Create events dataframe
                    st.session_state.events_df = pd.DataFrame(default_events_data)
                    st.session_state.events_df["Date"] = pd.to_datetime(st.session_state.events_df["Date"])
                
                # Streamlit UI
                st.title("🌬️ Air and Traffic Data with Events")
                st.markdown("Visualize air and traffic data with customizable metrics and events.")
                
                # Sidebar for controls
                st.sidebar.header("Configuration")
                
                # Get year options
                years = sorted(daily["Year"].unique())
                if not years:
                    st.error("No valid years found in the dataset")
                    return
                    
                # Year selector
                selected_year = st.sidebar.selectbox("Select Year", years)
                
                # Get numeric columns for selection (excluding Date, Year, Event)
                numeric_columns = [col for col in daily.columns if col not in ['Date', 'Year', 'Event'] 
                                and pd.api.types.is_numeric_dtype(daily[col])]
                
                if not numeric_columns:
                    st.error("No numeric columns found in the dataset for visualization")
                    return
                    
                # Variable selectors
                st.sidebar.subheader("Select Variables to Display")
                
                # Default to PM10 if available, otherwise use the first numeric column
                default_primary_index = numeric_columns.index('PM10') if 'PM10' in numeric_columns else 0
                primary_var = st.sidebar.selectbox(
                    "Primary Variable (Left Y-axis)", 
                    numeric_columns, 
                    index=default_primary_index
                )
                
                default_secondary = []
                for col in ['NOx', 'Temp', 'WS']:
                    if col in numeric_columns and col != primary_var:
                        default_secondary.append(col)
                        
                secondary_vars = st.sidebar.multiselect(
                    "Secondary Variables (Right Y-axes)",
                    [col for col in numeric_columns if col != primary_var],
                    default=default_secondary
                )
                
                # Wind direction plot option
                has_wind_direction = 'WD' in daily.columns
                show_wind_plot = False
                if has_wind_direction:
                    show_wind_plot = st.sidebar.checkbox(
                        "Show Wind Direction Plot", 
                        value=True,
                        help="Display wind direction as circular polar time series"
                    )
                
                # Event management
                st.sidebar.subheader("Event Management")
                
                # Event filtering options
                event_types = st.session_state.events_df["Type"].unique().tolist()
                selected_event_types = st.sidebar.multiselect(
                    "Show Event Types",
                    event_types,
                    default=event_types
                )
                
                # Bulk import/export events
                st.sidebar.subheader("Bulk Event Operations")
                
                # Upload events CSV
                uploaded_events_file = st.sidebar.file_uploader(
                    "Upload Events CSV",
                    type=['csv'],
                    help="CSV should have columns: Date, Event (and optionally Type)"
                )
                
                if uploaded_events_file is not None:
                    try:
                        # Read uploaded CSV
                        uploaded_events = pd.read_csv(uploaded_events_file)
                        
                        # Validate required columns
                        required_columns = ['Date', 'Event']
                        missing_columns = [col for col in required_columns if col not in uploaded_events.columns]
                        
                        if missing_columns:
                            st.sidebar.error(f"Missing required columns: {', '.join(missing_columns)}")
                        else:
                            # Process uploaded events
                            uploaded_events['Date'] = pd.to_datetime(uploaded_events['Date'])
                            
                            # Add Type column if not present
                            if 'Type' not in uploaded_events.columns:
                                uploaded_events['Type'] = 'Imported'
                            else:
                                # Ensure imported events are marked as custom types
                                uploaded_events['Type'] = uploaded_events['Type'].fillna('Imported')
                            
                            # Show preview and confirmation
                            st.sidebar.write("**Preview of uploaded events:**")
                            preview_df = uploaded_events.copy()
                            preview_df['Date'] = preview_df['Date'].dt.strftime('%Y-%m-%d')
                            st.sidebar.dataframe(preview_df.head(), hide_index=True)
                            
                            col1, col2 = st.sidebar.columns(2)
                            
                            with col1:
                                if st.button("Import Events", key="import_events"):
                                    # Check for duplicates and handle them
                                    existing_dates = st.session_state.events_df['Date'].dt.strftime('%Y-%m-%d').tolist()
                                    uploaded_dates = uploaded_events['Date'].dt.strftime('%Y-%m-%d').tolist()
                                    
                                    duplicates = set(existing_dates) & set(uploaded_dates)
                                    
                                    if duplicates:
                                        st.sidebar.warning(f"Found {len(duplicates)} duplicate dates. These will be updated.")
                                    
                                    # Remove existing events for duplicate dates
                                    for date_str in duplicates:
                                        date_to_remove = pd.to_datetime(date_str)
                                        st.session_state.events_df = st.session_state.events_df[
                                            st.session_state.events_df['Date'] != date_to_remove
                                        ]
                                    
                                    # Add new events
                                    st.session_state.events_df = pd.concat([
                                        st.session_state.events_df, uploaded_events
                                    ], ignore_index=True)
                                    
                                    # Sort by date
                                    st.session_state.events_df = st.session_state.events_df.sort_values('Date').reset_index(drop=True)
                                    
                                    st.sidebar.success(f"Successfully imported {len(uploaded_events)} events!")
                                    st.rerun()
                            
                            with col2:
                                if st.button("Cancel Import", key="cancel_import"):
                                    st.rerun()
                                    
                    except Exception as e:
                        st.sidebar.error(f"Error reading CSV file: {str(e)}")
                        st.sidebar.info("Please ensure your CSV has the correct format with Date and Event columns.")
                
                # Export events CSV
                if not st.session_state.events_df.empty:
                    # Prepare export data
                    export_df = st.session_state.events_df.copy()
                    export_df['Date'] = export_df['Date'].dt.strftime('%Y-%m-%d')
                    export_df = export_df.sort_values('Date')
                    
                    # Convert to CSV
                    csv_data = export_df.to_csv(index=False)
                    
                    st.sidebar.download_button(
                        label="📥 Download Events CSV",
                        data=csv_data,
                        file_name=f"events_export_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        help="Download all events as CSV file"
                    )
                
                # Custom event input
                st.sidebar.subheader("Add Custom Event")
                with st.sidebar.form("add_event_form"):
                    event_date = st.date_input("Event Date")
                    event_name = st.text_input("Event Description")
                    submitted = st.form_submit_button("Add Event")
                    
                if submitted and event_name:
                    # Check if event already exists for this date
                    event_date_pd = pd.to_datetime(event_date)
                    existing_event = st.session_state.events_df[
                        st.session_state.events_df["Date"] == event_date_pd
                    ]
                    
                    if not existing_event.empty:
                        # Update existing event
                        st.session_state.events_df.loc[
                            st.session_state.events_df["Date"] == event_date_pd, 
                            ["Event", "Type"]
                        ] = [event_name, "Custom"]
                        st.sidebar.success(f"Updated event for {event_date}: {event_name}")
                    else:
                        # Add new event
                        new_event = pd.DataFrame({
                            "Date": [event_date_pd],
                            "Event": [event_name],
                            "Type": ["Custom"]
                        })
                        st.session_state.events_df = pd.concat([
                            st.session_state.events_df, new_event
                        ], ignore_index=True)
                        st.sidebar.success(f"Added: {event_date} - {event_name}")
                
                # Display and manage events
                st.sidebar.subheader("Manage Events")
                
                # Show events for the selected year
                year_events = st.session_state.events_df[
                    st.session_state.events_df["Date"].dt.year == selected_year
                ].copy()
                
                if not year_events.empty:
                    year_events_sorted = year_events.sort_values("Date")
                    
                    for idx, row in year_events_sorted.iterrows():
                        col1, col2, col3 = st.sidebar.columns([2, 2, 1])
                        
                        with col1:
                            st.write(f"{row['Date'].strftime('%Y-%m-%d')}")
                        with col2:
                            st.write(f"{row['Event']} ({row['Type']})")
                        with col3:
                            if st.button("🗑️", key=f"remove_{idx}", help="Remove event"):
                                st.session_state.events_df = st.session_state.events_df.drop(idx).reset_index(drop=True)
                                st.rerun()
                else:
                    st.sidebar.info(f"No events for {selected_year}")
                
                # Clear all custom events button
                if st.sidebar.button("Clear All Custom Events"):
                    st.session_state.events_df = st.session_state.events_df[
                        st.session_state.events_df["Type"] == "Default"
                    ].reset_index(drop=True)
                    st.sidebar.success("All custom events cleared!")
                    st.rerun()
                
                # Filter events based on selected types
                filtered_events_df = st.session_state.events_df[
                    st.session_state.events_df["Type"].isin(selected_event_types)
                ].copy()
                
                # Create Event column in daily data
                daily['Event'] = ""
                
                # Apply events to dataframe using the events dataframe
                if not filtered_events_df.empty:
                    # Create a mapping dictionary from the events dataframe
                    event_mapping = dict(zip(
                        filtered_events_df["Date"].dt.strftime('%Y-%m-%d'),
                        filtered_events_df["Event"]
                    ))
                    daily["Event"] = daily["Date"].dt.strftime('%Y-%m-%d').map(event_mapping).fillna("")
                
                # Filter by year after events are mapped
                subset = daily[daily["Year"] == selected_year].copy()
                
                # Debug info for subset
                # with debug_expander:
                #     st.write("Subset shape:", subset.shape)
                #     st.write("Subset columns:", subset.columns.tolist())
                #     if not subset.empty:
                #         st.write("First few rows of subset:", subset.head(2))
                #         st.write("Events in subset:", subset[subset['Event'] != ''])
                
                # Plot
                fig = go.Figure()
                
                # Primary variable trace
                fig.add_trace(go.Scatter(
                    x=subset["Date"], 
                    y=subset[primary_var],
                    mode="lines+markers",
                    name=f"{primary_var}",
                    line=dict(color="blue"),
                    marker=dict(size=4)
                ))
                
                # Color palette for secondary variables
                colors = ["purple", "orange", "green", "red", "brown", "pink", "gray", "cyan"]
                dash_styles = ["dot", "dash", "dashdot", "solid"]
                
                # Create layout dictionary with initial settings
                layout_dict = {
                    "xaxis_title": "Date",
                    "yaxis": {"title": primary_var, "side": "left"},
                    "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "center", "x": 0.5},
                    "hovermode": "x unified",
                    "height": 700,
                    "template": "plotly_white",
                    "title": f"Air and Traffic with Events – {selected_year}"
                }
                
                # Add secondary variables with different y-axes
                for i, var in enumerate(secondary_vars):
                    color_index = i % len(colors)
                    dash_index = i % len(dash_styles)
                    
                    # Calculate y-axis position (starting from right side)
                    position = 0.95 - (i * 0.1)
                    position = max(position, 0.5)  # Ensure position doesn't go below 0.5
                    
                    axis_number = i + 2  # yaxis2, yaxis3, etc.
                    
                    # Add trace with reference to the correct axis
                    fig.add_trace(go.Scatter(
                        x=subset["Date"], 
                        y=subset[var],
                        yaxis=f"y{axis_number}", 
                        name=f"{var}",
                        line=dict(color=colors[color_index], dash=dash_styles[dash_index])
                    ))
                    
                    # Add the axis configuration to our layout dictionary
                    layout_dict[f"yaxis{axis_number}"] = {
                        "title": var,
                        "overlaying": "y", 
                        "side": "right", 
                        "anchor": "x", 
                        "position": position
                    }
                
                # Update figure layout all at once
                fig.update_layout(**layout_dict)
                
                # Annotated events (using primary variable's value for y-position)
                events_to_plot = subset[subset["Event"] != ""]
                if not events_to_plot.empty:
                    fig.add_trace(go.Scatter(
                        x=events_to_plot["Date"], 
                        y=events_to_plot[primary_var],
                        mode="markers+text",
                        name="Events",
                        text=events_to_plot["Event"],
                        textposition="top center",
                        marker=dict(color="red", size=10, symbol="x")
                    ))
                
                # Display plot
                st.plotly_chart(fig, use_container_width=True)
                
                # Wind Direction Polar Plot
                if show_wind_plot and has_wind_direction:
                    st.subheader("🧭 Wind Direction Analysis")
                    
                    # Create wind direction plot
                    wind_subset = subset.copy()
                    
                    # Convert wind direction to radians for polar plot
                    wind_subset['WD_rad'] = np.radians(wind_subset['WD'])
                    
                    # Create polar plot
                    wind_fig = go.Figure()
                    
                    # Add wind direction trace
                    wind_fig.add_trace(go.Scatterpolar(
                        r=wind_subset.index,  # Use index as radius (time progression)
                        theta=wind_subset['WD'],
                        mode='markers+lines',
                        name='Wind Direction',
                        line=dict(color='darkblue', width=1),
                        marker=dict(
                            size=4,
                            color=wind_subset[primary_var],  # Color by primary variable
                            colorscale='Viridis',
                            colorbar=dict(
                                title=f"{primary_var}",
                                x=1.1
                            ),
                            showscale=True
                        ),
                        hovertemplate=(
                            '<b>Date:</b> %{customdata[0]}<br>'
                            '<b>Wind Direction:</b> %{theta}°<br>'
                            f'<b>{primary_var}:</b> %{{customdata[1]:.2f}}<br>'
                            '<b>Event:</b> %{customdata[2]}<br>'
                            '<extra></extra>'
                        ),
                        customdata=np.column_stack([
                            wind_subset['Date'].dt.strftime('%Y-%m-%d'),
                            wind_subset[primary_var],
                            wind_subset['Event'].fillna('No event')
                        ])
                    ))
                    
                    # Add event markers on wind plot
                    wind_events = wind_subset[wind_subset["Event"] != ""]
                    if not wind_events.empty:
                        wind_fig.add_trace(go.Scatterpolar(
                            r=wind_events.index,
                            theta=wind_events['WD'],
                            mode='markers+text',
                            name='Events',
                            text=wind_events['Event'],
                            textposition='top center',
                            marker=dict(
                                color='red',
                                size=12,
                                symbol='x',
                                line=dict(width=2, color='darkred')
                            ),
                            hovertemplate=(
                                '<b>Event:</b> %{text}<br>'
                                '<b>Date:</b> %{customdata[0]}<br>'
                                '<b>Wind Direction:</b> %{theta}°<br>'
                                '<extra></extra>'
                            ),
                            customdata=wind_events['Date'].dt.strftime('%Y-%m-%d')
                        ))
                    
                    # Update polar plot layout
                    wind_fig.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                title="Time Progression",
                                showticklabels=False,
                                showgrid=True,
                                gridcolor='lightgray'
                            ),
                            angularaxis=dict(
                                direction="clockwise",
                                rotation=90,  # Rotate to put North at top
                                tickmode="array",
                                tickvals=[0, 45, 90, 135, 180, 225, 270, 315],
                                ticktext=["N", "NE", "E", "SE", "S", "SW", "W", "NW"],
                                showgrid=True,
                                gridcolor='lightgray'
                            ),
                            bgcolor='white'
                        ),
                        title=f"Wind Direction Pattern with Events - {selected_year}",
                        showlegend=True,
                        height=600,
                        template="plotly_white"
                    )
                    
                    st.plotly_chart(wind_fig, use_container_width=True)
                    
                    # Wind direction statistics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Mean Wind Direction", 
                            f"{wind_subset['WD'].mean():.1f}°",
                            help="Circular mean of wind direction"
                        )
                    
                    with col2:
                        # Calculate predominant direction
                        wind_bins = pd.cut(wind_subset['WD'], bins=8, labels=['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])
                        predominant_dir = wind_bins.mode().iloc[0] if not wind_bins.mode().empty else "N/A"
                        st.metric(
                            "Predominant Direction", 
                            predominant_dir,
                            help="Most frequent wind direction sector"
                        )
                    
                    with col3:
                        # Calculate variability (circular standard deviation approximation)
                        wind_std = wind_subset['WD'].std()
                        st.metric(
                            "Direction Variability", 
                            f"{wind_std:.1f}°",
                            help="Standard deviation of wind direction"
                        )
                    
                    # Wind direction distribution
                    st.subheader("Wind Direction Distribution")
                    
                    # Create wind rose-like histogram
                    wind_hist_fig = go.Figure()
                    
                    # Create 16 direction bins
                    wind_bins = np.arange(0, 361, 22.5)  # 16 bins of 22.5 degrees each
                    wind_labels = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
                                   'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
                    
                    # Calculate frequencies for each bin
                    wind_counts, _ = np.histogram(wind_subset['WD'], bins=wind_bins)
                    
                    # Calculate the center angles for each bin
                    bin_centers = (wind_bins[:-1] + wind_bins[1:]) / 2
                    
                    wind_hist_fig.add_trace(go.Barpolar(
                        r=wind_counts,
                        theta=bin_centers,
                        width=22.5,  # Width of each bar in degrees
                        name='Frequency',
                        marker_color='lightblue',
                        marker_line_color='darkblue',
                        marker_line_width=1,
                        opacity=0.8,
                        hovertemplate='<b>Direction:</b> %{customdata}<br><b>Frequency:</b> %{r}<extra></extra>',
                        customdata=wind_labels
                    ))
                    
                    wind_hist_fig.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                title="Frequency",
                                showticklabels=True,
                                showgrid=True,
                                gridcolor='lightgray'
                            ),
                            angularaxis=dict(
                                direction="clockwise",
                                rotation=90,  # North at top
                                tickmode="array",
                                tickvals=[0, 45, 90, 135, 180, 225, 270, 315],
                                ticktext=["N", "NE", "E", "SE", "S", "SW", "W", "NW"],
                                showgrid=True,
                                gridcolor='lightgray'
                            ),
                            bgcolor='white'
                        ),
                        title="Wind Direction Frequency Distribution",
                        showlegend=False,
                        height=500,
                        template="plotly_white"
                    )
                    
                    st.plotly_chart(wind_hist_fig, use_container_width=True)
                
                # Show event table
                if not filtered_events_df.empty:
                    st.subheader("Event Details")
                    
                    # Filter events for the selected year
                    year_events_display = filtered_events_df[
                        filtered_events_df["Date"].dt.year == selected_year
                    ].copy()
                    
                    if not year_events_display.empty:
                        # Format for display
                        year_events_display["Date"] = year_events_display["Date"].dt.strftime("%Y-%m-%d")
                        year_events_display = year_events_display.sort_values("Date")
                        year_events_display = year_events_display.rename(columns={"Event": "Event Description"})
                        
                        st.dataframe(year_events_display, hide_index=True)
                    else:
                        st.info(f"No events recorded for {selected_year}")
                
                # Display events dataframe summary
                with st.expander("📊 Events Summary"):
                    st.write(f"**Total Events**: {len(st.session_state.events_df)}")
                    event_type_counts = st.session_state.events_df["Type"].value_counts()
                    for event_type, count in event_type_counts.items():
                        st.write(f"**{event_type} Events**: {count}")
                    
                    # Show all events in a table
                    display_events = st.session_state.events_df.copy()
                    display_events["Date"] = display_events["Date"].dt.strftime("%Y-%m-%d")
                    display_events = display_events.sort_values("Date")
                    st.dataframe(display_events, hide_index=True)
                    
                    # Export button in main area as well
                    if not st.session_state.events_df.empty:
                        export_df = st.session_state.events_df.copy()
                        export_df['Date'] = export_df['Date'].dt.strftime('%Y-%m-%d')
                        export_df = export_df.sort_values('Date')
                        csv_data = export_df.to_csv(index=False)
                        
                        st.download_button(
                            label="📥 Download Complete Events List",
                            data=csv_data,
                            file_name=f"all_events_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            help="Download complete events database as CSV"
                        )
                        
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.write("Please check if your dataframe has the correct format.")
                import traceback
                st.code(traceback.format_exc())

        # Example usage
        if __name__ == "__main__":
            events(df)

    else:
        st.error('⛔ [ATTENTION] Please confirm your data in the main page to proceed.')
else:
    st.markdown('⛔ [ATTENTION] No file found. Please upload and process file/s in the main page to access this module.')

