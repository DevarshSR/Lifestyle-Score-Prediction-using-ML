import streamlit as st
import pandas as pd
import joblib

# ==========================
# 🔹 Load Saved Models
# ==========================
score_model = joblib.load('score.pkl')        # Regression model
category_model = joblib.load('category.pkl')  # Classification model
le = joblib.load('labelencoder.pkl')          # Label encoder

# ==========================
# 🔹 Page Configuration
# ==========================
st.set_page_config(page_title="Lifestyle Health Predictor", page_icon="💪", layout="centered")
st.title("🏋️‍♂️ Lifestyle Health Predictor")
st.write("Enter your daily lifestyle details below to predict your **Lifestyle Score** and **Lifestyle Category**.")

st.markdown("---")

# ==========================
# 🔹 User Inputs
# ==========================
col1, col2 = st.columns(2)

with col1:
    Sleep_hours = st.slider('🛏️ Hours of Sleep', 3.0, 10.0, 7.0)
    Workout_days_per_week = st.slider('🏃‍♂️ Workout Days per Week', 0, 7, 3)
    Workout_intensity = st.slider('🔥 Workout Intensity (1=Low, 3=High)', 1, 3, 2)
    Steps_per_day = st.slider('🚶 Steps per Day', 2000, 15000, 8000)
    Water_intake_L = st.slider('💧 Water Intake (Litres)', 1.5, 6.0, 3.0)

with col2:
    Stress_level = st.slider('😫 Stress Level (1-10)', 1.0, 10.0, 4.0)
    Screen_time_hrs = st.slider('📱 Screen Time (hrs)', 1.0, 16.0, 6.0)
    Junk_food_per_week = st.slider('🍔 Junk Food per Week', 0, 15, 3)


# ==========================
# 🔹 Create Input DataFrame
# ==========================
user_df = pd.DataFrame([{
    'Sleep_hours': Sleep_hours,
    'Workout_days_per_week': Workout_days_per_week,
    'Workout_intensity': Workout_intensity,
    'Steps_per_day': Steps_per_day,
    'Water_intake_L': Water_intake_L,
    'Stress_level': Stress_level,
    'Junk_food_per_week': Junk_food_per_week,
    'Screen_time_hrs': Screen_time_hrs
}])

st.markdown("---")

# ==========================
# 🔹 Predict Button
# ==========================
if st.button("🚀 Predict My Lifestyle"):
    # Predict Lifestyle Score
    score_pred = score_model.predict(user_df)[0]

    # Predict Category
    category_pred = category_model.predict(user_df)[0]
    category_label = le.inverse_transform([category_pred])[0]

    # Display Predictions
    st.subheader(f"🧠 **Predicted Lifestyle Score:** {round(score_pred, 2)} / 100")
    st.subheader(f"🏆 **Lifestyle Category:** {category_label}")

    # Feedback Message
    if score_pred >= 70:
        st.success("💪 Excellent lifestyle! Keep it up — you’re on the right path!")
    elif score_pred >= 40:
        st.warning("👌 Average lifestyle — you’re doing okay, but there’s room to improve!")
    else:
        st.error("⚠️ Unhealthy lifestyle — time to make some positive changes!")

    st.markdown("---")
    st.caption("💡 Tip: Improve your sleep, reduce stress, and stay hydrated for a better score!")

else:
    st.info("👆 Adjust your sliders and click **Predict My Lifestyle** to see your results.")
