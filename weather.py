import streamlit as st
import pandas as pd
import numpy as np
import random
import time

# Attempt to import matplotlib; if not installed, display an error and stop the app.
try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    st.error("Matplotlib is not installed. Please install it using: pip install matplotlib")
    st.stop()

# -----------------------------------
# Data Preparation and Model Training
# -----------------------------------

st.title("Weather Forecast Web App using Machine Learning")

st.write("""
This app uses a machine learning model to forecast temperature based on weather parameters.
It trains both a Linear Regression model and a Decision Tree model on a sample dataset and displays the results.
""")

# Generate sample weather data
data = {
    'date': pd.date_range(start='2020-01-01', periods=365),
    'temperature': np.random.randint(-10, 40, 365),
    'humidity': np.random.randint(20, 100, 365),
    'pressure': np.random.randint(980, 1040, 365),
    'precipitation': np.random.rand(365) * 10,
    'wind_speed': np.random.rand(365) * 30
}

df = pd.DataFrame(data)
df['date'] = pd.to_datetime(df['date'])
df['day_of_year'] = df['date'].dt.dayofyear

st.write("### Sample Data")
st.write(df.head())

# Feature Engineering
X = df[['day_of_year', 'humidity', 'pressure', 'precipitation', 'wind_speed']]
y = df['temperature']

# Split data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

st.write("Data is split into training (80%) and testing (20%) sets.")

# ------------------------------
# Linear Regression Model
# ------------------------------

st.subheader("Training Linear Regression Model")

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)

lr_mae = mean_absolute_error(y_test, lr_predictions)
lr_mse = mean_squared_error(y_test, lr_predictions)
lr_r2  = r2_score(y_test, lr_predictions)

st.write("**Linear Regression Metrics:**")
st.write(f"MAE: {lr_mae:.2f}")
st.write(f"MSE: {lr_mse:.2f}")
st.write(f"R² Score: {lr_r2:.2f}")

# ------------------------------
# Decision Tree Model
# ------------------------------

st.subheader("Training Decision Tree Model")

from sklearn.tree import DecisionTreeRegressor

dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)
dt_predictions = dt_model.predict(X_test)

dt_mae = mean_absolute_error(y_test, dt_predictions)
dt_mse = mean_squared_error(y_test, dt_predictions)
dt_r2  = r2_score(y_test, dt_predictions)

st.write("**Decision Tree Metrics:**")
st.write(f"MAE: {dt_mae:.2f}")
st.write(f"MSE: {dt_mse:.2f}")
st.write(f"R² Score: {dt_r2:.2f}")

# ------------------------------
# Visualization of Predictions
# ------------------------------

st.subheader("Model Predictions vs Actual Temperature")

fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Plot for Linear Regression
ax[0].scatter(y_test, lr_predictions, alpha=0.5)
ax[0].plot([y.min(), y.max()], [y.min(), y.max()], 'k--')
ax[0].set_xlabel('Actual Temperature')
ax[0].set_ylabel('Predicted Temperature')
ax[0].set_title('Linear Regression Predictions')

# Plot for Decision Tree
ax[1].scatter(y_test, dt_predictions, alpha=0.5, color='green')
ax[1].plot([y.min(), y.max()], [y.min(), y.max()], 'k--')
ax[1].set_xlabel('Actual Temperature')
ax[1].set_ylabel('Predicted Temperature')
ax[1].set_title('Decision Tree Predictions')

st.pyplot(fig, use_container_width=True)

# Feature Importance for Decision Tree
st.subheader("Decision Tree Feature Importance")
feature_importance = pd.Series(
    dt_model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)
st.write(feature_importance)

# ------------------------------
# Optional: Weather Forecast Prediction Demo
# ------------------------------

st.subheader("Weather Forecast Prediction Demo")

st.write("Enter weather parameters to get a temperature forecast:")

day_of_year = st.slider("Day of Year", 1, 365, 200)
humidity = st.slider("Humidity (%)", 0, 100, 50)
pressure = st.slider("Pressure (hPa)", 980, 1040, 1013)
precipitation = st.slider("Precipitation (mm)", 0.0, 10.0, 2.0)
wind_speed = st.slider("Wind Speed (km/h)", 0.0, 30.0, 10.0)

input_data = pd.DataFrame({
    'day_of_year': [day_of_year],
    'humidity': [humidity],
    'pressure': [pressure],
    'precipitation': [precipitation],
    'wind_speed': [wind_speed]
})

model_choice = st.selectbox("Select Model for Prediction", ["Linear Regression", "Decision Tree"])

if st.button("Predict Temperature"):
    if model_choice == "Linear Regression":
        prediction = lr_model.predict(input_data)[0]
    else:
        prediction = dt_model.predict(input_data)[0]
    st.success(f"Predicted Temperature: {prediction:.2f} °C")

# -----------------------------------
# Q-learning for a Weather Forecast Web App (Demonstration)
# -----------------------------------
# Here we include a separate simple example demonstrating Q-learning in another context.
# For this example, we provide the code from the weather forecast project only if required.
# (If you don't want this, simply ignore or remove this section.)

st.markdown("---")
st.header("(Optional) Q-learning Demo for Weather Forecasting")
st.write("This section is reserved for potential Q-learning based approaches if required.")

# (Code for Q-learning might be added here if needed; however, the above code covers ML-based forecasting.)
# In this demo, we focus on the regression and decision tree forecasts.

# -----------------------------------
# End of App
# -----------------------------------
