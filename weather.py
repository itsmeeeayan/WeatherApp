# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import random
import time

# Attempt to import matplotlib, and if not installed, show an error message.
try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    st.error("Matplotlib is not installed. Please install it by running 'pip install matplotlib' in your environment.")
    st.stop()

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
    st.title("🌤️ Kolkata Weather Forecast")
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
            # Expecting the CSV to have a 'Date' column and a 'temp' column for target temperature,
            # plus other weather-related features and a categorical 'conditions' column.
            df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
            df['day_of_year'] = df['Date'].dt.dayofyear
            df['month'] = df['Date'].dt.month
            # Fill missing categorical data
            if 'conditions' in df.columns:
                df['conditions'] = df['conditions'].fillna('none')
            else:
                # Create a default conditions column if it doesn't exist
                df['conditions'] = 'none'
            df['precipitation_flag'] = np.where(df['precip'] > 0, 1, 0)
            return df
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            return None

    if upload_file is not None:
        df = load_data(upload_file)
        if df is not None:
            try:
                # Feature selection: adjust features as needed
                features = [
                    'day_of_year', 'humidity', 'sealevelpressure', 'precip',
                    'windspeed', 'cloudcover', 'month', 'precipitation_flag'
                ]
                target = 'temp'
    
                # Preprocessor for categorical variable 'conditions'
                preprocessor = ColumnTransformer(
                    transformers=[('cat', OneHotEncoder(), ['conditions'])],
                    remainder='passthrough'
                )
    
                # Combine numerical features and categorical conditions
                X = df[features + ['conditions']]
                y = df[target]
    
                # Split data into training and testing sets
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size/100, random_state=42
                )
    
                # Process the input features
                X_train_processed = preprocessor.fit_transform(X_train)
                X_test_processed = preprocessor.transform(X_test)
    
                # ==============================
                # Linear Regression Model
                # ==============================
                st.markdown("### Training Linear Regression Model...")
                lr_model = LinearRegression()
                lr_model.fit(X_train_processed, y_train)
                lr_predictions = lr_model.predict(X_test_processed)
    
                lr_mae = mean_absolute_error(y_test, lr_predictions)
                lr_mse = mean_squared_error(y_test, lr_predictions)
                lr_r2  = r2_score(y_test, lr_predictions)
    
                st.write("**Linear Regression Metrics:**")
                st.write(f"MAE: {lr_mae:.2f}°F")
                st.write(f"MSE: {lr_mse:.2f}")
                st.write(f"R² Score: {lr_r2:.2f}")
    
                # ==============================
                # Decision Tree Model
                # ==============================
                st.markdown("### Training Decision Tree Model...")
                dt_model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
                dt_model.fit(X_train_processed, y_train)
                dt_predictions = dt_model.predict(X_test_processed)
    
                dt_mae = mean_absolute_error(y_test, dt_predictions)
                dt_mse = mean_squared_error(y_test, dt_predictions)
                dt_r2  = r2_score(y_test, dt_predictions)
    
                st.write("**Decision Tree Metrics:**")
                st.write(f"MAE: {dt_mae:.2f}°F")
                st.write(f"MSE: {dt_mse:.2f}")
                st.write(f"R² Score: {dt_r2:.2f}")
    
                # ==============================
                # Visualization of Predictions
                # ==============================
                st.markdown("### Predictions Visualization")
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
                # Plot for Linear Regression Predictions
                ax1.scatter(y_test.values, lr_predictions, alpha=0.5)
                ax1.plot([y.min(), y.max()], [y.min(), y.max()], 'k--')
                ax1.set_xlabel('Actual Temperature')
                ax1.set_ylabel('Predicted Temperature')
                ax1.set_title('Linear Regression Predictions')
    
                # Plot for Decision Tree Predictions
                ax2.scatter(y_test.values, dt_predictions, alpha=0.5, color='green')
                ax2.plot([y.min(), y.max()], [y.min(), y.max()], 'k--')
                ax2.set_xlabel('Actual Temperature')
                ax2.set_ylabel('Predicted Temperature')
                ax2.set_title('Decision Tree Predictions')
    
                st.pyplot(fig, use_container_width=True)
    
                # ==============================
                # Feature Importance for Decision Tree
                # ==============================
                st.markdown("### Decision Tree Feature Importance")
                # Get feature names from OneHotEncoder and combine with numerical feature names
                cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(['conditions']).tolist()
                feature_names = cat_feature_names + features
                importances = dt_model.feature_importances_
    
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importances
                }).sort_values('Importance', ascending=False)
    
                st.bar_chart(importance_df.set_index('Feature'))
    
                # ==============================
                # Prediction Interface Demo
                # ==============================
                st.markdown("### Make Your Own Temperature Prediction")
                with st.form("prediction_form"):
                    st.write("Enter weather parameters:")
    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        date = st.date_input("Date")
                        humidity_input = st.number_input("Humidity (%)", 0, 100, 60)
                    with col2:
                        pressure_input = st.number_input("Sea Level Pressure (hPa)", 980, 1040, 1010)
                        precip_input = st.number_input("Precipitation (mm)", 0.0, 50.0, 0.0)
                    with col3:
                        wind_speed_input = st.number_input("Wind Speed (km/h)", 0.0, 50.0, 10.0)
                        cloud_cover_input = st.number_input("Cloud Cover (%)", 0, 100, 30)
    
                    # For conditions, use the unique values from the dataset
                    condition = st.selectbox("Weather Condition", df['conditions'].unique())
    
                    submitted = st.form_submit_button("Predict Temperature")
                    if submitted:
                        input_data = pd.DataFrame([{
                            'day_of_year': pd.Timestamp(date).dayofyear,
                            'month': pd.Timestamp(date).month,
                            'humidity': humidity_input,
                            'sealevelpressure': pressure_input,
                            'precip': precip_input,
                            'windspeed': wind_speed_input,
                            'cloudcover': cloud_cover_input,
                            'precipitation_flag': 1 if precip_input > 0 else 0,
                            'conditions': condition
                        }])
    
                        processed_input = preprocessor.transform(input_data)
    
                        # Predict using both models
                        lr_pred_value = lr_model.predict(processed_input)[0]
                        dt_pred_value = dt_model.predict(processed_input)[0]
    
                        st.markdown("#### Prediction Results")
                        col_lr, col_dt = st.columns(2)
                        with col_lr:
                            st.metric("Linear Regression Prediction", f"{lr_pred_value:.1f}°F")
                        with col_dt:
                            st.metric("Decision Tree Prediction", f"{dt_pred_value:.1f}°F")
    
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.stop()
    else:
        st.info("👈 Please upload a CSV file to get started")
    
if __name__ == "__main__":
    main()
