import streamlit as st
import pickle
import pandas as pd
import numpy as np
from datetime import datetime

# Load the trained model
with open('models/xgb_energy_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.set_page_config(page_title="Energy Forecast App", layout="wide")
st.title("⚡ Energy Consumption Forecasting App")

st.markdown("### 🧠 Input recent consumption data and forecast the next hour's energy usage")

# Input: last 168 hourly values (or at least 24)
recent_input = st.text_area(
    "Enter the last 168 hourly energy values (comma-separated)",
    placeholder="e.g. 1510, 1525, 1530, ...",
    height=150
)

if recent_input:
    try:
        values = [float(v.strip()) for v in recent_input.split(",") if v.strip() != ""]
        values = np.array(values)

        if len(values) < 168:
            st.warning("Please enter at least 168 values (for lag_168).")
        else:
            # Get the most recent value (current = t)
            t_minus_1 = values[-1]
            t_minus_24 = values[-24]
            t_minus_168 = values[-168]
            rolling_3 = values[-3:].mean()
            rolling_24 = values[-24:].mean()
            rolling_std_24 = values[-24:].std()

            # Calendar features from next time step (t + 1)
            next_hour = (datetime.now().hour + 1) % 24
            next_day = datetime.now().weekday() if next_hour > datetime.now().hour else (datetime.now().weekday() + 1) % 7
            next_month = datetime.now().month
            is_weekend = 1 if next_day >= 5 else 0

            # Construct input features
            input_df = pd.DataFrame([{
                'lag_1': t_minus_1,
                'lag_24': t_minus_24,
                'lag_168': t_minus_168,
                'rolling_3': rolling_24,
                'rolling_24': rolling_24,
                'rolling_std_24': rolling_std_24,
                'hour': next_hour,
                'dayofweek': next_day,
                'month': next_month,
                'is_weekend': is_weekend
            }])

            st.subheader("🔍 Preview of Model Input")
            st.dataframe(input_df.T, use_container_width=True)

            if st.button("🔮 Predict Next Hour's Energy Usage"):
                prediction = model.predict(input_df)[0]
                st.success(f"Forecasted Energy Consumption: **{prediction:.2f} MW**")

    except Exception as e:
        st.error(f"Something went wrong: {e}")
