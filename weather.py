# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # Make sure this is installed
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Set page configuration
st.set_page_config(
    page_title="Kolkata Weather Forecast",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    # Add title and description
    st.title("🌤️ Kolkata Weather Forecasting Model")
    st.markdown("""
    This app uses machine learning to predict temperatures in Kolkata using:
    - **Linear Regression**
    - **Decision Tree Regressor**
    """)

    # Sidebar for user inputs
    with st.sidebar:
        st.header("⚙️ Settings")
        test_size = st.slider("Test Set Size (%)", 10, 40, 20)
        max_depth = st.slider("Decision Tree Max Depth", 2, 10, 5)
        upload_file = st.file_uploader("Upload CSV Data", type=["csv"])

    # Load and preprocess data
    @st.cache_data
    def load_data(file):
        try:
            df = pd.read_csv(file)
            
            # Convert and extract date features
            df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
            df['day_of_year'] = df['Date'].dt.dayofyear
            df['month'] = df['Date'].dt.month
            
            # Handle missing values and create features
            df['preciptype'] = df['preciptype'].fillna('none')
            df['precipitation_flag'] = np.where(df['precip'] > 0, 1, 0)
            
            return df
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            return None

    # Main app logic
    if upload_file is not None:
        df = load_data(upload_file)
        if df is not None:
            try:
                # Rest of your code remains the same...
                # [Keep all the existing code here without changes]

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.stop()
    else:
        st.info("👈 Please upload a CSV file to get started")

if __name__ == "__main__":
    main()
