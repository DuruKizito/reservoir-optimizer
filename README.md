# README.md
# ⛽ Real-Time Reservoir Optimization System

A full-stack platform for hydrocarbon field optimization using:

- 📡 Live sensor data (MQTT/WebSocket)
- 🧠 Machine learning + reinforcement learning (Ray RLlib)
- 📊 Interactive dashboards (Streamlit + Plotly)
- 🔁 Real-time prediction & RL decision streaming

## 🚀 Features

- Upload Eclipse, Production, and Reservoir data
- Predict Pressure/Saturation with ML
- RL-based optimization agent with live rewards
- Live streaming of RL decisions and sensor feedback

## 📦 Tech Stack

- Python, Streamlit, Ray RLlib, MQTT, WebSocket
- Plotly, Scikit-Learn, DEAP, Pandas, Docker

## 📂 Project Structure

```bash
dashboard.py              # Main Streamlit app
models/                   # ML/RL logic
streams/                  # MQTT + WebSocket clients/servers
utils/                    # Parsers + visualizers
tests/                    # Unit tests (pytest)
data/, logs/              # Runtime storage
Dockerfile, run.sh, README.md

