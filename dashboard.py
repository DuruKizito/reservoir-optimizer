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

WELL_SAVE_PATH = "data/last_uploaded_wells.csv"

st.set_page_config(page_title="Real-Time Reservoir Optimization", layout="wide")

st.title("🔄 Real-Time Reservoir Optimization Dashboard")
tabs = st.tabs(["📁 Upload & Parse", "📈 Live Sensor Feed", "🧠 ML Prediction", "🤖 RL Training", "📊 Visualization"])

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
    st.header("🧠 Predict Pressure & Saturation (ML)")
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

# --- Upload well file ---
st.sidebar.header("🛢️ Well Overlay Options")
show_wells = st.sidebar.checkbox("Show Wells on Contours", value=True, key="show_wells_main")
uploaded_well_file = st.sidebar.file_uploader("Upload Well Coordinates (.csv)", type=["csv"])

# --- Parse wells ---
well_coords = []
if uploaded_well_file:
    try:
        df_wells = pd.read_csv(uploaded_well_file)
        required_columns = {'well_name', 'x', 'y'}
        if required_columns.issubset(set(df_wells.columns)):
            well_coords = df_wells.to_dict(orient='records')
            st.sidebar.success(f"{len(well_coords)} wells loaded")
        else:
            st.sidebar.error("CSV must include: well_name, x, y")
    except Exception as e:
        st.sidebar.error(f"Error reading file: {e}")

# ---- Download Template CSV ----
sample_df = pd.DataFrame({
    'well_name': ['Well-A', 'Well-B'],
    'x': [5, 10],
    'y': [7, 3]
})
csv_bytes = sample_df.to_csv(index=False).encode()
st.sidebar.download_button(
    label="📥 Download Well Template",
    data=csv_bytes,
    file_name="well_template.csv",
    mime='text/csv'
)

# ---- Upload Well Coordinates ----
uploaded_well_file = st.sidebar.file_uploader("Upload Well Coordinates (.csv)", type=["csv"])

well_coords = []
if uploaded_well_file:
    try:
        df_wells = pd.read_csv(uploaded_well_file)
        required_columns = {'well_name', 'x', 'y'}
        if required_columns.issubset(df_wells.columns):
            well_coords = df_wells.to_dict(orient='records')
            st.sidebar.success(f"{len(well_coords)} wells loaded")

            # Save to /data/
            os.makedirs("data", exist_ok=True)
            df_wells.to_csv(WELL_SAVE_PATH, index=False)
            st.sidebar.info(f"Auto-saved to: {WELL_SAVE_PATH}")
        else:
            st.sidebar.error("CSV must contain columns: well_name, x, y")
    except Exception as e:
        st.sidebar.error(f"Error reading file: {e}")
elif os.path.exists(WELL_SAVE_PATH):
    df_wells = pd.read_csv(WELL_SAVE_PATH)
    well_coords = df_wells.to_dict(orient='records')
    st.sidebar.info("Using last uploaded wells from saved file.")

# === Notifications ===
if show_wells:
    if well_coords:
        st.success(f"✅ {len(well_coords)} wells loaded and overlaid.")
    else:
        st.warning("⚠ 'Show Wells' is enabled, but no well data was loaded.")
else:
    st.info("ℹ Well overlay is disabled.")

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


logging.basicConfig(filename='logs/training.log', level=logging.INFO)

# Make sure episode and reward are defined in your RL training loop before logging
# Example:
# for episode in range(num_episodes):
#     reward = ... # get reward from training
#     logging.info(f"Episode {episode} | Reward: {reward}")