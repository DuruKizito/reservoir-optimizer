# dashboard.py
# A Streamlit dashboard for real-time reservoir optimization using ML and RL
# This dashboard allows users to upload production and reservoir data, visualize sensor feeds,
# make predictions using machine learning, and train a reinforcement learning agent for optimal control.

print("Dashboard started")
import streamlit as st
from models.bhp_estimator import BHP_Estimator
from models.rl_agent import RLAgent
from models.reservoir_model import ReservoirMLModel
from streams.mqtt_client import MQTTSensorClient
from utils.visualizer import (
    plot_pressure_contour,
    plot_saturation_contour,
    plot_reward_curve,
    display_live_metrics
)
from utils.parser import (
    parse_production_csv,
    parse_reservoir_properties,
    parse_eclipse_output
)
import numpy as np
import pandas as pd
import os
import logging
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

WELL_SAVE_PATH = "data/last_uploaded_wells.csv"

st.set_page_config(page_title="Real-Time Reservoir Optimization", layout="wide")

st.title("\U0001F501 Real-Time Reservoir Optimization Dashboard")

# --- Upload well file and production data per well (move above tabs) ---
st.sidebar.header("\U0001F6E2️ Well Overlay Options")
show_wells = st.sidebar.checkbox("Show Wells on Contours", value=True, key="show_wells_main")
uploaded_well_file = st.sidebar.file_uploader("Upload Well Coordinates (.csv)", type=["csv"], key="well_coords_uploader")

well_coords = []
well_names = []
if uploaded_well_file:
    try:
        df_wells = pd.read_csv(uploaded_well_file)
        required_columns = {'well_name', 'x', 'y'}
        if required_columns.issubset(set(df_wells.columns)):
            well_coords = df_wells.to_dict(orient='records')
            well_names = list(df_wells['well_name'])
            st.sidebar.success(f"{len(well_coords)} wells loaded")
            os.makedirs("data", exist_ok=True)
            df_wells.to_csv(WELL_SAVE_PATH, index=False)
            st.sidebar.info(f"Auto-saved to: {WELL_SAVE_PATH}")
        else:
            st.sidebar.error("CSV must include: well_name, x, y")
    except Exception as e:
        st.sidebar.error(f"Error reading file: {e}")
elif os.path.exists(WELL_SAVE_PATH):
    df_wells = pd.read_csv(WELL_SAVE_PATH)
    well_coords = df_wells.to_dict(orient='records')
    well_names = list(df_wells['well_name'])
    st.sidebar.info("Using last uploaded wells from saved file.")

# ---- Download Template CSV ----
sample_df = pd.DataFrame({
    'well_name': ['Well-A', 'Well-B'],
    'x': [5, 10],
    'y': [7, 3]
})
csv_bytes = sample_df.to_csv(index=False).encode()
st.sidebar.download_button(
    label="\U0001F4E5 Download Well Template",
    data=csv_bytes,
    file_name="well_template.csv",
    mime='text/csv'
)

# --- Upload production data per well ---
st.sidebar.header("\U0001F4C8 Production Data Per Well")
if 'prod_data_per_well' not in st.session_state:
    st.session_state['prod_data_per_well'] = {}
for well in well_names:
    prod_file = st.sidebar.file_uploader(f"Upload Production Data for {well}", type=["csv"], key=f"prod_{well}")
    if prod_file:
        try:
            df_prod = pd.read_csv(prod_file)
            required_prod_cols = {'well_name', 'time', 'rate', 'BHP pressure', 'THP Pressure'}
            if required_prod_cols.issubset(set(df_prod.columns)):
                st.session_state['prod_data_per_well'][well] = df_prod
                st.sidebar.success(f"Production data loaded for {well}")
            else:
                st.sidebar.error(f"{well}: CSV must include: {', '.join(required_prod_cols)}")
        except Exception as e:
            st.sidebar.error(f"{well}: Error reading file: {e}")
prod_data_per_well = st.session_state['prod_data_per_well']

tabs = st.tabs(["\U0001F4C1 Upload & Parse", "\U0001F4C8 Live Sensor Feed", "\U0001F9E0 ML Prediction", "\U0001F916 RL Training", "\U0001F4CA Visualization"])

# Shared instances
rl_agent = RLAgent()
ml_model = ReservoirMLModel(grid_size=(10, 10, 10))
bhp_estimator = BHP_Estimator()

# --- 📁 Upload & Parse ---
with tabs[0]:
    st.header("📁 Upload Production & Reservoir Files")
    prod_file = st.file_uploader("Upload Production CSV", type=["csv"])
    resv_file = st.file_uploader("Upload Rock & Fluid Properties", type=["csv", "json"])
    eclipse_file = st.file_uploader("Upload Eclipse Output", type=["csv", "tsv"])

    if prod_file:
        with open("data/uploads/prod.csv", "wb") as f:
            f.write(prod_file.read())
        df_prod = parse_production_csv("data/uploads/prod.csv")
        st.success("Production data parsed!")
        st.write(df_prod.head())

    if resv_file:
        with open("data/uploads/resv.csv", "wb") as f:
            f.write(resv_file.read())
        resv_data = parse_reservoir_properties("data/uploads/resv.csv")
        st.success("Reservoir properties loaded.")

    if eclipse_file:
        with open("data/uploads/eclipse.csv", "wb") as f:
            f.write(eclipse_file.read())
        eclipse_data = parse_eclipse_output("data/uploads/eclipse.csv")
        st.success("Eclipse data parsed.")
        st.write(eclipse_data.head())

# --- 📈 Live Sensor Feed ---
with tabs[1]:
    st.header("📡 Live MQTT Sensor Stream")
    live_data = MQTTSensorClient.read_live_feed()
    display_live_metrics(live_data)
    st.json(live_data)

# --- 🧠 ML Prediction ---
with tabs[2]:
    st.header("🧠 Predict Pressure, Saturation & Production Rate (ML)")
    if resv_file:
        st.write("📌 Using uploaded rock/fluid data.")
        pressure_pred, sat_pred = ml_model.predict(
            grid={"cartDims": [10, 10, 10]},
            rock=resv_data['rock'],
            fluid=resv_data['fluid'],
            schedule={}
        )
        st.success("Prediction Complete!")
        plot_pressure_contour(pressure_pred)
        plot_saturation_contour(sat_pred)
        # --- ML Production Prediction per well ---
        st.subheader("Predicted Production Rate per Well")
        for well, df_prod in prod_data_per_well.items():
            # Dummy prediction: use rate column as-is, replace with ML model output as needed
            st.write(f"**{well}**")
            st.line_chart(df_prod[['time', 'rate']].set_index('time'))
            # Decline curve (simple exponential fit as placeholder)
            if len(df_prod) > 1:
                def decline_curve(t, qi, D):
                    return qi * np.exp(-D * t)
                try:
                    popt, _ = curve_fit(decline_curve, df_prod['time'], df_prod['rate'], maxfev=10000)
                    t_fit = np.linspace(df_prod['time'].min(), df_prod['time'].max(), 100)
                    rate_fit = decline_curve(t_fit, *popt)
                    fig, ax = plt.subplots()
                    ax.plot(df_prod['time'], df_prod['rate'], 'o', label='Actual')
                    ax.plot(t_fit, rate_fit, '-', label='Decline Curve')
                    ax.set_xlabel('Time (days)')
                    ax.set_ylabel('Rate (STB/day)')
                    ax.set_title(f'Decline Curve - {well}')
                    ax.legend()
                    st.pyplot(fig)
                except Exception as e:
                    st.warning(f"Decline curve fit failed for {well}: {e}")
        # --- Overall reservoir production (sum of all wells) ---
        if prod_data_per_well:
            st.subheader("Overall Reservoir Production Rate")
            all_prod = pd.concat(prod_data_per_well.values())
            all_prod_grouped = all_prod.groupby('time')['rate'].sum().reset_index()
            st.line_chart(all_prod_grouped.set_index('time'))
    else:
        st.warning("Upload rock/fluid property file first.")

# --- 🤖 RL Training ---
with tabs[3]:
    st.header("🎯 Reinforcement Learning Control")
    if st.button("▶️ Train RL Agent (10 Episodes)"):
        rewards = rl_agent.train(10)
        st.session_state['rewards'] = st.session_state.get('rewards', []) + rewards
        st.success("Training completed.")
    if st.button("💾 Save Agent"):
        checkpoint = rl_agent.save()
        st.success(f"Checkpoint saved: {checkpoint}")
    if st.button("📂 Load Last Checkpoint"):
        rl_agent.load()
        st.success("Checkpoint loaded.")

    st.subheader("📊 Episode Rewards")
    plot_reward_curve(st.session_state.get('rewards', []))

# --- 📊 Visualization ---
with tabs[4]:
    st.header("📊 Visualize BHP or Eclipse Data")
    if prod_file:
        st.subheader("🔍 BHP Estimation")
        results = bhp_estimator.process_well_data(
            well_data=df_prod,
            depth_column='depth',
            density_column='density',
            pressure_column='pressure'
        )
        st.dataframe(results)
    elif eclipse_file:
        st.subheader("🛢️ Eclipse Output Preview")
        st.write(eclipse_data.head())
    else:
        st.info("Upload data to view results.")

# --- Display predictions (with toggle) ---
pressure_pred = None
saturation_pred = None
if st.button("📈 Show Predictions"):
    # Use ml_model and previously loaded data
    if 'resv_data' in locals():
        grid = {"cartDims": [10, 10, 10]}
        rock = resv_data.get('rock')
        fluid = resv_data.get('fluid')
        pressure_pred, saturation_pred = ml_model.predict(grid, rock, fluid, {})
        # ...
    else:
        st.warning("Upload rock/fluid property file first.")

    if show_wells and well_coords and pressure_pred is not None:
        plot_pressure_contour(pressure_pred, well_coords=well_coords)
        plot_saturation_contour(saturation_pred, well_coords=well_coords)
    elif pressure_pred is not None:
        plot_pressure_contour(pressure_pred)
        plot_saturation_contour(saturation_pred)
    else:
        st.info("No predictions to display yet. Run a prediction first.")

# Only plot if pressure_pred is defined
if pressure_pred is not None:
    if show_wells and well_coords:
        plot_pressure_contour(pressure_pred, well_coords=well_coords)
        plot_saturation_contour(saturation_pred, well_coords=well_coords)
    else:
        plot_pressure_contour(pressure_pred)
        plot_saturation_contour(saturation_pred)
else:
    st.info("No predictions to display yet. Run a prediction first.")

with st.expander("🩺 Logs / Monitoring"):
    st.code(open("logs/training.log").read() if os.path.exists("logs/training.log") else "No logs yet.")

# Ensure logs directory exists before configuring logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(filename='logs/training.log', level=logging.INFO)

# Make sure episode and reward are defined in your RL training loop before logging
# Example:
# for episode in range(num_episodes):
#     reward = ... # get reward from training
#     logging.info(f"Episode {episode} | Reward: {reward}")