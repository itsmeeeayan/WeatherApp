# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
                # Feature selection
                features = [
                    'day_of_year', 'humidity', 'sealevelpressure', 'precip',
                    'windspeed', 'cloudcover', 'month', 'precipitation_flag'
                ]
                target = 'temp'
                
                # Preprocessing
                preprocessor = ColumnTransformer(
                    transformers=[('cat', OneHotEncoder(), ['conditions'])],
                    remainder='passthrough'
                )
                
                X = df[features + ['conditions']]
                y = df[target]
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size/100, random_state=42
                )
                
                # Process data
                X_train_processed = preprocessor.fit_transform(X_train)
                X_test_processed = preprocessor.transform(X_test)
                
                # Train models
                lr_model = LinearRegression()
                lr_model.fit(X_train_processed, y_train)
                
                dt_model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
                dt_model.fit(X_train_processed, y_train)
                
                # Make predictions
                lr_pred = lr_model.predict(X_test_processed)
                dt_pred = dt_model.predict(X_test_processed)
                
                # Calculate metrics
                lr_mae = mean_absolute_error(y_test, lr_pred)
                lr_mse = mean_squared_error(y_test, lr_pred)
                lr_r2 = r2_score(y_test, lr_pred)
                
                dt_mae = mean_absolute_error(y_test, dt_pred)
                dt_mse = mean_squared_error(y_test, dt_pred)
                dt_r2 = r2_score(y_test, dt_pred)
                
                # Display results
                st.header("📊 Model Performance")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Linear Regression")
                    st.metric("MAE", f"{lr_mae:.2f}°F")
                    st.metric("MSE", f"{lr_mse:.2f}")
                    st.metric("R² Score", f"{lr_r2:.2f}")
                    
                with col2:
                    st.subheader("Decision Tree")
                    st.metric("MAE", f"{dt_mae:.2f}°F")
                    st.metric("MSE", f"{dt_mse:.2f}")
                    st.metric("R² Score", f"{dt_r2:.2f}")
                
                # Visualization
                st.header("📈 Predictions Visualization")
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
                
                # Actual vs Predicted plot
                ax1.plot(y_test.values, label='Actual')
                ax1.plot(lr_pred, label='Linear Regression')
                ax1.plot(dt_pred, label='Decision Tree')
                ax1.set_xlabel("Data Points")
                ax1.set_ylabel("Temperature (°F)")
                ax1.set_title("Actual vs Predicted Temperatures")
                ax1.legend()
                
                # Residual plot
                ax2.scatter(lr_pred, lr_pred - y_test, alpha=0.5, label='Linear Regression')
                ax2.scatter(dt_pred, dt_pred - y_test, alpha=0.5, color='green', label='Decision Tree')
                ax2.axhline(y=0, color='r', linestyle='--')
                ax2.set_xlabel("Predicted Values")
                ax2.set_ylabel("Residuals")
                ax2.set_title("Residual Analysis")
                ax2.legend()
                
                st.pyplot(fig)
                
                # Feature importance
                st.header("🔍 Feature Importance (Decision Tree)")
                feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(['conditions']).tolist() + features
                importances = dt_model.feature_importances_
                
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importances
                }).sort_values('Importance', ascending=False)
                
                st.bar_chart(importance_df.set_index('Feature'))
                
                # Prediction interface
                st.header("🔮 Make Prediction")
                with st.form("prediction_form"):
                    st.write("Enter weather parameters:")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        date = st.date_input("Date")
                        humidity = st.number_input("Humidity (%)", 0, 100, 60)
                        
                    with col2:
                        pressure = st.number_input("Sea Level Pressure (hPa)", 980, 1040, 1010)
                        precip = st.number_input("Precipitation (mm)", 0.0, 50.0, 0.0)
                        
                    with col3:
                        wind_speed = st.number_input("Wind Speed (km/h)", 0.0, 50.0, 10.0)
                        cloud_cover = st.number_input("Cloud Cover (%)", 0, 100, 30)
                        
                    condition = st.selectbox("Weather Condition", df['conditions'].unique())
                    
                    if st.form_submit_button("Predict Temperature"):
                        # Create input DataFrame
                        input_data = pd.DataFrame([{
                            'day_of_year': pd.Timestamp(date).dayofyear,
                            'month': pd.Timestamp(date).month,
                            'humidity': humidity,
                            'sealevelpressure': pressure,
                            'precip': precip,
                            'windspeed': wind_speed,
                            'cloudcover': cloud_cover,
                            'precipitation_flag': 1 if precip > 0 else 0,
                            'conditions': condition
                        }])
                        
                        # Preprocess input
                        processed_input = preprocessor.transform(input_data)
                        
                        # Make predictions
                        lr_pred = lr_model.predict(processed_input)[0]
                        dt_pred = dt_model.predict(processed_input)[0]
                        
                        # Display results
                        st.subheader("Prediction Results")
                        col1, col2 = st.columns(2)
                        col1.metric("Linear Regression Prediction", f"{lr_pred:.1f}°F")
                        col2.metric("Decision Tree Prediction", f"{dt_pred:.1f}°F")

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.stop()
    else:
        st.info("👈 Please upload a CSV file to get started")

if __name__ == "__main__":
    main()
