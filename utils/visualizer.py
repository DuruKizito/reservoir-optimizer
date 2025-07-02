# visualizer.py
import numpy as np
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import streamlit as st


def plot_pressure_contour(pressure_field: np.ndarray, well_coords=None, title="Pressure Contour (Mid-Z Layer)"):
    """
    Plot a 2D contour from the center layer of a 3D pressure field with optional well trajectories.
    """
    nz = pressure_field.shape[2]
    mid_layer = pressure_field[:, :, nz // 2]

    fig = go.Figure()

    # Contour layer
    fig.add_trace(go.Contour(
        z=mid_layer,
        contours=dict(coloring='heatmap'),
        colorbar=dict(title='Pressure'),
        showscale=True
    ))

    # Optional wells
    if well_coords:
        x_vals = [w['x'] for w in well_coords]
        y_vals = [w['y'] for w in well_coords]
        fig.add_trace(go.Scatter(
            x=x_vals, y=y_vals,
            mode='markers+text',
            text=[w['well_name'] for w in well_coords],
            name='Wells',
            marker=dict(size=8, color='black', symbol='cross'),
            textposition='top center'
        ))

    fig.update_layout(title=title, xaxis_title="X", yaxis_title="Y")
    st.plotly_chart(fig, use_container_width=True)


def plot_saturation_contour(saturation_field: np.ndarray, well_coords=None, title="Saturation Contour (Mid-Z Layer)"):
    """
    Plot a 2D contour from the center layer of a 3D saturation field with optional well trajectories.
    """
    nz = saturation_field.shape[2]
    mid_layer = saturation_field[:, :, nz // 2]

    fig = go.Figure()

    fig.add_trace(go.Contour(
        z=mid_layer,
        contours=dict(coloring='heatmap'),
        colorbar=dict(title='Saturation'),
        showscale=True
    ))

    if well_coords:
        x_vals = [w['x'] for w in well_coords]
        y_vals = [w['y'] for w in well_coords]
        fig.add_trace(go.Scatter(
            x=x_vals, y=y_vals,
            mode='markers+text',
            text=[w['well_name'] for w in well_coords],
            name='Wells',
            marker=dict(size=8, color='black', symbol='cross'),
            textposition='top center'
        ))

    fig.update_layout(title=title, xaxis_title="X", yaxis_title="Y")
    st.plotly_chart(fig, use_container_width=True)


def plot_reward_curve(reward_history: list, title="RL Episode Rewards"):
    """
    Plot reward progression over training episodes.
    """
    if not reward_history:
        st.warning("No reward data to plot.")
        return

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(1, len(reward_history) + 1)),
        y=reward_history,
        mode='lines+markers',
        name='Episode Reward'
    ))
    fig.update_layout(title=title, xaxis_title="Episode", yaxis_title="Reward")
    st.plotly_chart(fig, use_container_width=True)


def plot_time_series_predictions(time_steps: list, values: list, ylabel: str = "Pressure", title="Time Series Prediction"):
    """
    Plot time-series prediction data.
    """
    if not values:
        st.warning("No time-series prediction data.")
        return

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time_steps,
        y=values,
        mode='lines+markers',
        name=ylabel
    ))
    fig.update_layout(title=title, xaxis_title="Time (days)", yaxis_title=ylabel)
    st.plotly_chart(fig, use_container_width=True)


def display_live_metrics(sensor_data: dict):
    """
    Show live sensor metrics in Streamlit.
    """
    if not sensor_data:
        st.info("Waiting for live sensor data...")
        return

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Live Pressure", f"{sensor_data.get('pressure', 'N/A')} psi")
    with col2:
        st.metric("Saturation", f"{sensor_data.get('saturation', 'N/A')}")

