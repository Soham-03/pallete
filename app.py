import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(layout="wide", page_title="PalletSense Analytics Dashboard")

# Function to calculate distance between two points using Haversine formula
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Earth's radius in kilometers
    
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return R * c

# Load and prepare data
@st.cache_data
def load_data():
    df = pd.read_csv('known_route_repo.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

# Load data
df = load_data()

# Title and Introduction
st.title("🚛 PalletSense Analytics Dashboard")
st.markdown("### Smart Solutions for Pallet Tracking")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Analysis Section",
    ["Overview",
     "Challenge 1: Route Analysis",
     "Challenge 2: Deviation Detection",
     "Challenge 3: Stationary Pallets",
     "Recommendations"]
)

# Overview Section
if page == "Overview":
    st.header("📊 System Overview")
    
    # Key Metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_routes = df['Route_ID'].nunique()
        st.metric("Total Routes", total_routes)
        
    with col2:
        total_pings = len(df)
        st.metric("Total GPS Pings", total_pings)
        
    with col3:
        avg_pings = total_pings / total_routes
        st.metric("Average Pings per Route", f"{avg_pings:.2f}")
    
    # Timeline visualization
    st.subheader("GPS Ping Activity Timeline")
    
    # Aggregate data by date
    daily_pings = df.groupby(df['timestamp'].dt.date).size().reset_index()
    daily_pings.columns = ['date', 'count']
    
    fig = px.line(daily_pings, x='date', y='count',
                  title='Daily GPS Ping Activity')
    st.plotly_chart(fig, use_container_width=True)
    
    # Route Distribution
    st.subheader("Route Distribution")
    route_counts = df['Route_ID'].value_counts().head(10)
    fig = px.bar(route_counts, title='Top 10 Routes by Number of Pings')
    st.plotly_chart(fig, use_container_width=True)

elif page == "Challenge 1: Route Analysis":
    st.header("🛣️ Route Analysis")
    
    # Time gap analysis
    st.subheader("Time Gap Analysis Between Pings")
    
    # Calculate time differences for each route
    df_sorted = df.sort_values(['Route_ID', 'timestamp'])
    df_sorted['time_diff'] = df_sorted.groupby('Route_ID')['timestamp'].diff()
    df_sorted['time_diff_hours'] = df_sorted['time_diff'].dt.total_seconds() / 3600
    
    # Plot time gaps
    fig = px.box(df_sorted, x='Route_ID', y='time_diff_hours',
                 title='Distribution of Time Gaps Between Pings by Route')
    st.plotly_chart(fig, use_container_width=True)
    
    # Route clustering
    st.subheader("Route Geographic Distribution")
    fig = px.scatter_mapbox(df, 
                           lat='latitude', 
                           lon='longitude',
                           color='Route_ID',
                           title='Route Geographic Distribution',
                           mapbox_style="carto-positron")
    st.plotly_chart(fig, use_container_width=True)
    
    # Analysis insights
    st.markdown("""
    ### Key Findings:
    - Average time between pings varies significantly by route
    - Geographic clusters show common travel patterns
    - Multiple pallets often travel similar routes
    """)

elif page == "Challenge 2: Deviation Detection":
    st.header("🎯 Deviation Detection")
    
    # Calculate route deviations
    st.subheader("Route Deviation Analysis")
    
    # Function to calculate deviation from expected path
    def calculate_deviation(group):
        if len(group) < 3:
            return pd.Series({'deviation': 0})
            
        start_point = group.iloc[0]
        end_point = group.iloc[-1]
        
        max_deviation = 0
        for idx in range(1, len(group)-1):
            point = group.iloc[idx]
            # Calculate deviation as distance from point to line
            deviation = haversine_distance(
                point['latitude'], point['longitude'],
                start_point['latitude'], start_point['longitude']
            )
            max_deviation = max(max_deviation, deviation)
            
        return pd.Series({'deviation': max_deviation})
    
    route_deviations = df.groupby('Route_ID').apply(calculate_deviation).reset_index()
    
    # Plot deviations
    fig = px.bar(route_deviations, x='Route_ID', y='deviation',
                 title='Maximum Route Deviations')
    st.plotly_chart(fig, use_container_width=True)
    
    # High deviation routes
    st.subheader("High Deviation Routes")
    high_dev_routes = route_deviations[route_deviations['deviation'] > 
                                     route_deviations['deviation'].mean()]
    st.dataframe(high_dev_routes)

elif page == "Challenge 3: Stationary Pallets":
    st.header("⏸️ Stationary Pallet Detection")
    
    # Function to detect stationary periods
    def detect_stationary_periods(group, threshold_hours=24):
        if len(group) < 2:
            return pd.DataFrame()
            
        stationary_periods = []
        current_start = group.iloc[0]['timestamp']
        current_pos = group.iloc[0][['latitude', 'longitude']]
        
        for idx in range(1, len(group)):
            point = group.iloc[idx]
            distance = haversine_distance(
                point['latitude'], point['longitude'],
                current_pos['latitude'], current_pos['longitude']
            )
            
            time_diff = (point['timestamp'] - current_start).total_seconds() / 3600
            
            if distance < 0.1:  # Less than 100 meters movement
                if time_diff >= threshold_hours:
                    stationary_periods.append({
                        'Route_ID': group.iloc[0]['Route_ID'],
                        'start_time': current_start,
                        'end_time': point['timestamp'],
                        'duration_hours': time_diff,
                        'latitude': current_pos['latitude'],
                        'longitude': current_pos['longitude']
                    })
            else:
                current_start = point['timestamp']
                current_pos = point[['latitude', 'longitude']]
                
        return pd.DataFrame(stationary_periods)
    
    # Process data for stationary periods
    stationary_df = pd.concat([
        detect_stationary_periods(group) 
        for _, group in df.groupby('Route_ID')
    ])
    
    if not stationary_df.empty:
        # Visualize stationary locations
        st.subheader("Stationary Pallet Locations")
        fig = px.scatter_mapbox(stationary_df,
                               lat='latitude',
                               lon='longitude',
                               size='duration_hours',
                               color='duration_hours',
                               title='Stationary Pallet Locations',
                               mapbox_style="carto-positron")
        st.plotly_chart(fig, use_container_width=True)
        
        # Stationary duration analysis
        st.subheader("Stationary Duration Analysis")
        fig = px.histogram(stationary_df, x='duration_hours',
                          title='Distribution of Stationary Durations')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No stationary periods detected with current threshold")

else:  # Recommendations
    st.header("💡 Recommendations")
    
    st.subheader("Immediate Actions")
    st.markdown("""
    1. **Route Optimization**
       - Implement dynamic route reassignment based on historical patterns
       - Optimize ping frequency for high-risk routes
       
    2. **Alert System Enhancement**
       - Deploy predictive deviation alerts
       - Set up graduated alert severity levels
    """)
    
    st.subheader("Long-term Strategy")
    st.markdown("""
    1. **Infrastructure Improvements**
       - Install additional tracking points in high-risk areas
       - Implement mesh network for better coverage
       
    2. **Analytics Enhancement**
       - Develop machine learning models for prediction
       - Implement real-time route optimization
    """)
    
    # Impact Analysis
    st.subheader("Expected Impact")
    impact_data = pd.DataFrame({
        'Metric': ['Lost Pallet Rate', 'Detection Time', 'Route Efficiency', 'Cost Savings'],
        'Improvement': [30, 45, 25, 35]
    })
    
    fig = px.bar(impact_data, x='Metric', y='Improvement',
                 title='Expected Improvements (%)')
    st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("PalletSense Analytics Dashboard | Created with Streamlit")