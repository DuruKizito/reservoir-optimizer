# README.md
# ⛽ Real-Time Reservoir Optimization System

A full-stack platform for hydrocarbon field optimization using:

- 📡 Live sensor data (MQTT/WebSocket)
- 🧠 Machine learning + reinforcement learning (Stable Baselines3)
- 📊 Interactive dashboards (Streamlit + Plotly)
- 🔁 Real-time prediction & RL decision streaming

## 🚀 Features

- Upload Eclipse, Production, and Reservoir data
- Predict Pressure/Saturation with ML
- RL-based optimization agent with live rewards (Stable Baselines3 PPO)
- Live streaming of RL decisions and sensor feedback
- Dockerized for easy deployment

## 📦 Tech Stack

- Python, Streamlit, Stable Baselines3 (PPO), MQTT, WebSocket
- Plotly, Scikit-Learn, DEAP, Pandas, Docker
- Gym, Gymnasium, Shimmy, OpenCV

## 📂 Project Structure

```bash
dashboard.py              # Main Streamlit app
models/                   # ML/RL logic (Stable Baselines3 PPO agent)
streams/                  # MQTT + WebSocket clients/servers
utils/                    # Parsers + visualizers
requirements.txt          # Python dependencies
Dockerfile, run.sh, README.md
logs/, data/              # Runtime storage
```

## 🐳 Quickstart (Docker)

```bash
docker build -t reservoir-optimizer .
docker run --rm -p 8501:8501 reservoir-optimizer
```

Then open http://localhost:8501 in your browser.

## 📝 Notes
- RLlib is no longer used; RL agent is implemented with Stable Baselines3 for better compatibility and easier deployment.
- All dependencies are installed via `requirements.txt` and Dockerfile.
- For local development, create a virtual environment and run `pip install -r requirements.txt`.

