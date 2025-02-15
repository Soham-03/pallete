import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import folium_static

# Define the haversine_distance function
def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points in kilometers using Haversine formula"""
    R = 6371  # Earth's radius in kilometers
    
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return R * c

def create_map(df):
    center_lat = df['latitude'].mean()
    center_lon = df['longitude'].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=6)
    
    # Add route line
    coordinates = [[row['latitude'], row['longitude']] for index, row in df.iterrows()]
    folium.PolyLine(coordinates, weight=2, color='blue', opacity=0.8).add_to(m)
    
    # Add start and end markers
    folium.Marker(
        [df['latitude'].iloc[0], df['longitude'].iloc[0]],
        popup='Start',
        icon=folium.Icon(color='green')
    ).add_to(m)
    
    folium.Marker(
        [df['latitude'].iloc[-1], df['longitude'].iloc[-1]],
        popup='End',
        icon=folium.Icon(color='red')
    ).add_to(m)
    
    return m

def load_and_process_data():
    try:
        # Load the CSV file
        df = pd.read_csv('on going trip data.csv')
        
        # Handle timestamp conversion with error checking
        def safe_parse_timestamp(ts):
            try:
                return pd.to_datetime(ts)
            except:
                return None

        # Convert timestamp column safely
        df['timestamp'] = df['timestamp'].apply(safe_parse_timestamp)
        
        # Remove any rows with invalid timestamps
        df = df.dropna(subset=['timestamp'])
        
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        # Calculate distances
        df['distance_km'] = 0.0
        for i in range(1, len(df)):
            df.iloc[i, df.columns.get_loc('distance_km')] = haversine_distance(
                df.iloc[i-1]['latitude'], df.iloc[i-1]['longitude'],
                df.iloc[i]['latitude'], df.iloc[i]['longitude']
            )
        
        df['cumulative_distance'] = df['distance_km'].cumsum()
        
        # Calculate time differences for speed
        df['time_diff'] = df['timestamp'].diff().dt.total_seconds() / 3600  # in hours
        df['speed_kmh'] = df['distance_km'] / df['time_diff']
        
        # Remove infinite or unrealistic speeds
        df.loc[df['speed_kmh'] > 150, 'speed_kmh'] = None
        
        return df
        
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        return None

def main():
    st.title("Ongoing Trips Analysis Dashboard")
    
    # Load and process data
    df = load_and_process_data()
    
    if df is None or len(df) == 0:
        st.error("No valid data available for analysis")
        return
        
    # Display data overview
    st.subheader("Data Overview")
    st.write(f"Number of valid records: {len(df)}")
    st.write(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Journey Overview
    st.header("1. Journey Overview")
    col1, col2, col3 = st.columns(3)
    
    # Calculate metrics
    duration_hours = (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 3600
    total_distance = df['cumulative_distance'].max()
    avg_speed = total_distance / duration_hours if duration_hours > 0 else 0
    
    with col1:
        st.metric("Total Duration", f"{duration_hours:.1f} hours")
    with col2:
        st.metric("Total Distance", f"{total_distance:.1f} km")
    with col3:
        st.metric("Average Speed", f"{avg_speed:.1f} km/h")
    
    # Route Visualization
    st.header("2. Route Visualization")
    st.write("Interactive map showing pallet movements:")
    m = create_map(df)
    folium_static(m)
    
    # Speed Analysis
    st.header("3. Speed Analysis")
    valid_speeds = df[df['speed_kmh'].notna()]
    if not valid_speeds.empty:
        fig_speed = px.line(valid_speeds, x='timestamp', y='speed_kmh', 
                           title='Speed over Time')
        st.plotly_chart(fig_speed)
    
    # Pallet Movement Analysis
    st.header("4. Pallet Movement Patterns")
    
    # Time of Day Analysis
    st.subheader("Activity by Hour")
    df['hour'] = df['timestamp'].dt.hour
    hourly_counts = df.groupby('hour').size().reset_index(name='count')
    fig_hourly = px.bar(hourly_counts, x='hour', y='count',
                       title='Pallet Activity by Hour of Day',
                       labels={'hour': 'Hour of Day', 'count': 'Number of Records'})
    st.plotly_chart(fig_hourly)
    
    # Movement Patterns
    st.header("5. Key Insights")
    
    # Calculate key statistics
    avg_speed = valid_speeds['speed_kmh'].mean()
    max_speed = valid_speeds['speed_kmh'].max()
    peak_hour = hourly_counts.loc[hourly_counts['count'].idxmax(), 'hour']
    
    st.markdown(f"""
    ### Movement Statistics:
    - Average Speed: {avg_speed:.1f} km/h
    - Maximum Speed: {max_speed:.1f} km/h
    - Peak Activity Hour: {peak_hour:02d}:00
    - Total Movement Distance: {total_distance:.1f} km
    
    ### Analysis Insights:
    1. **Speed Patterns**
       - Average movement speed indicates {'normal' if 30 < avg_speed < 90 else 'unusual'} transportation patterns
       - Speed variations suggest {'consistent' if valid_speeds['speed_kmh'].std() < 20 else 'variable'} movement
    
    2. **Activity Patterns**
       - Peak activity at {peak_hour:02d}:00 suggests {'optimal' if 10 <= peak_hour <= 14 else 'sub-optimal'} timing
       - Movement patterns align with {'standard' if 6 <= peak_hour <= 18 else 'non-standard'} delivery hours
    
    3. **Efficiency Indicators**
       - Distance covered indicates {'efficient' if total_distance < 500 else 'long-distance'} route planning
       - Time utilization shows {'good' if duration_hours < 12 else 'extended'} delivery duration
    """)

    # Download option
    if st.button("Download Analysis Report"):
        # Prepare analysis data
        analysis_data = pd.DataFrame({
            'Metric': ['Total Duration (hours)', 'Total Distance (km)', 'Average Speed (km/h)',
                      'Maximum Speed (km/h)', 'Peak Activity Hour', 'Number of Records'],
            'Value': [duration_hours, total_distance, avg_speed,
                     max_speed, peak_hour, len(df)]
        })
        
        csv = analysis_data.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="pallet_movement_analysis.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()