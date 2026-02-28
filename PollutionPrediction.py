import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

# Set Page Configuration
st.set_page_config(page_title="Air Quality Predictor", page_icon="🌍", layout="wide")

# -----------------------------------------------------------------------------
# 1. Caching Data & Model to ensure the app stays fast
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv('city_day.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    return df

@st.cache_resource
def train_model(df):
    # Prepare data
    df_clean = df.dropna(subset=['AQI']).copy()
    df_clean['Year'] = df_clean['Date'].dt.year
    df_clean['Month'] = df_clean['Date'].dt.month
    
    # Drop target and leaky columns
    X = df_clean.drop(['AQI', 'Date', 'AQI_Bucket'], axis=1, errors='ignore')
    y = df_clean['AQI']
    
    # Identify columns
    categorical_cols = ['City']
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Preprocessing pipelines
    numerical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer([
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])
    
    # Model Pipeline (Random Forest)
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(n_estimators=50, max_depth=15, random_state=42, n_jobs=-1)) # Slightly tuned for speed in UI
    ])
    
    # Train
    pipeline.fit(X, y)
    return pipeline, numerical_cols

# -----------------------------------------------------------------------------
# 2. Helper function to categorize AQI
# -----------------------------------------------------------------------------
def get_aqi_category_and_color(aqi):
    if aqi <= 50:
        return "Good", "#00E400" # Green
    elif aqi <= 100:
        return "Satisfactory", "#FFFF00" # Yellow
    elif aqi <= 200:
        return "Moderate", "#FF7E00" # Orange
    elif aqi <= 300:
        return "Poor", "#FF0000" # Red
    elif aqi <= 400:
        return "Very Poor", "#8F3F97" # Purple
    else:
        return "Severe", "#7E0023" # Maroon

# -----------------------------------------------------------------------------
# 3. Main Application Build
# -----------------------------------------------------------------------------
def main():
    st.title("🌍 Smart Air Quality Predictor (India)")
    st.markdown("Predict the **Air Quality Index (AQI)** based on real-time environmental pollutants and explore historical trends.")

    # Load data and model
    with st.spinner("Loading Data & Training Model..."):
        df = load_data()
        model, num_cols = train_model(df)

    # UI: Sidebar for User Inputs
    st.sidebar.header("🔬 Input Parameters")
    st.sidebar.markdown("Adjust the pollutant levels below to simulate air quality:")

    # City & Date inputs
    cities = sorted(df['City'].dropna().unique().tolist())
    selected_city = st.sidebar.selectbox("Select City", cities)
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        selected_year = st.selectbox("Year", list(range(2020, 2041)))
    with col2:
        selected_month = st.selectbox("Month", list(range(1, 13)))

    # Get median values for the selected city to use as slider defaults
    city_data = df[df['City'] == selected_city]
    
    input_data = {'City': selected_city, 'Year': selected_year, 'Month': selected_month}
    
    st.sidebar.subheader("Pollutant Concentrations (µg/m³)")
    # Generate dynamic sliders for all numerical features
    for col in num_cols:
        if col not in ['Year', 'Month']:
            # Safe defaults just in case a city lacks data for a specific pollutant
            min_val = float(df[col].min()) if not pd.isna(df[col].min()) else 0.0
            max_val = float(df[col].quantile(0.95)) if not pd.isna(df[col].quantile(0.95)) else 100.0
            default_val = float(city_data[col].median()) if not pd.isna(city_data[col].median()) else (max_val/2)
            
            # Create slider
            val = st.sidebar.slider(col, min_value=0.0, max_value=max(100.0, max_val * 1.5), value=default_val, step=0.1)
            input_data[col] = val

    # UI: Main Body Tabs
    tab1, tab2 = st.tabs(["🎯 AQI Prediction", "📈 Historical Trends"])

    with tab1:
        st.subheader(f"Prediction for {selected_city}")
        
        # Convert user inputs to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Predict
        predicted_aqi = model.predict(input_df)[0]
        category, color = get_aqi_category_and_color(predicted_aqi)

        # Beautiful display of the result using HTML/CSS
        st.markdown(
            f"""
            <div style="background-color: {color}; padding: 20px; border-radius: 10px; text-align: center; color: {'white' if category in ['Poor', 'Very Poor', 'Severe'] else 'black'};">
                <h2 style="margin:0;">Predicted AQI: {predicted_aqi:.2f}</h2>
                <h3 style="margin:0;">Category: {category}</h3>
            </div>
            <br>
            """,
            unsafe_allow_html=True
        )

        st.info("💡 **Tip:** Try increasing PM2.5 or NO2 in the sidebar to see how drastically it degrades the air quality index!")
        
        # Display the input data back to the user
        st.write("**Your Input Breakdown:**")
        st.dataframe(input_df.drop(['City', 'Year', 'Month'], axis=1).style.format("{:.2f}"))

    with tab2:
        st.subheader(f"Historical AQI Trends for {selected_city}")
        
        if not city_data.empty:
            # Group by month/year for a cleaner chart
            trend_data = city_data.dropna(subset=['AQI']).groupby('Date')['AQI'].mean().reset_index()
            
            # Plotly interactive time-series chart
            fig = px.line(trend_data, x='Date', y='AQI', 
                          title=f"AQI Over Time in {selected_city}",
                          labels={'AQI': 'Air Quality Index', 'Date': 'Date'},
                          color_discrete_sequence=['#1f77b4'])
            
            # Add a horizontal line for the "Safe" limit (AQI = 100)
            fig.add_hline(y=100, line_dash="dash", line_color="green", annotation_text="Safe Limit")
            fig.update_layout(xaxis_title="Timeline", yaxis_title="Daily Average AQI", hovermode="x unified")
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No historical data available for the selected city.")

if __name__ == "__main__":

    main()
