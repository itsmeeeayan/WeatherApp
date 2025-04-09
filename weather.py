import streamlit as st
import numpy as np
import time

# -------------------------------
# Dummy Machine Learning Model
# -------------------------------
# In a real-world app, you might load a pre-trained ML model using joblib or pickle:
#   from joblib import load
#   model = load('weather_model.joblib')
#
# For this demo, we create a dummy prediction function.
def predict_weather(temp, humidity, wind_speed):
    """
    Dummy ML function to simulate weather forecast.
    It takes in temperature (°C), humidity (%), and wind speed (km/h) and
    returns a forecast string along with a confidence score.
    """
    # Here we compute a dummy "score" using an arbitrary formula.
    score = 0.3 * temp + 0.4 * (100 - humidity) + 0.3 * wind_speed

    # Use some simple rules to decide a forecast.
    if humidity > 70:
        forecast = "Rainy"
    elif temp > 25 and humidity < 50:
        forecast = "Sunny"
    elif 15 <= temp <= 25:
        forecast = "Cloudy"
    else:
        forecast = "Mixed"

    # For demonstration, we return both the forecast and the dummy score.
    return forecast, score

# -------------------------------
# Weather Forecast Streamlit App
# -------------------------------
st.title("Weather Forecast Web App")
st.write("""
This app uses a simple (dummy) machine learning model to predict the weather forecast based on user inputs.
Enter the values for temperature, humidity, and wind speed to get a forecast!
""")

# Sidebar for inputs:
st.sidebar.header("Input Weather Parameters")
temperature = st.sidebar.slider("Temperature (°C)", -10, 40, 20)
humidity = st.sidebar.slider("Humidity (%)", 0, 100, 50)
wind_speed = st.sidebar.slider("Wind Speed (km/h)", 0, 50, 10)

# Button to generate a forecast:
if st.button("Predict Weather"):
    with st.spinner("Predicting..."):
        # Simulate computation delay
        time.sleep(1)
        forecast, score = predict_weather(temperature, humidity, wind_speed)
    
    st.success("Prediction complete!")
    st.subheader("Forecast Results")
    st.write(f"**Forecast:** {forecast}")
    st.write(f"**Confidence Score:** {score:.2f}")

# -------------------------------
# Optional: Display Weather Forecast Animation
# -------------------------------
# Here we simulate an animation of a weather forecast "evolution" 
# by showing a sequence of images that represent weather conditions.
st.markdown("---")
st.header("Weather Forecast Animation (Demo)")

st.write("""
Below is a demo animation (images) that you can run to see a simulated forecast update.
For a real application, you could replace these images with weather icons or maps that match your forecast.
""")

animation_speed = st.slider("Animation Speed (seconds per frame)", 0.1, 2.0, 1.0)

# Dummy image paths (update with your own image files if available)
# For example, create image files named "sunny.png", "cloudy.png", "rainy.png", etc.
weather_images = {
    "Sunny": "sunny.png",
    "Cloudy": "cloudy.png",
    "Rainy": "rainy.png",
    "Mixed": "mixed.png"
}

# Button to start the animation:
if st.button("Start Weather Animation"):
    forecast_images = []
    # Create a simple sequence by repeating the predicted condition.
    # In a more complex app you might have multiple frames.
    if forecast in weather_images:
        forecast_images = [weather_images[forecast]] * 5
    else:
        forecast_images = [weather_images["Mixed"]] * 5

    placeholder = st.empty()
    for img_file in forecast_images:
        placeholder.image(img_file, use_container_width=True)
        time.sleep(animation_speed)
    st.success("Animation complete!")
