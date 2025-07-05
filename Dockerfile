# Use official Python image with a supported version (3.9 recommended for RLlib 2.x)
FROM python:3.9-bullseye

# Set work directory
WORKDIR /app

# Install system dependencies and update GPG keys
RUN set -eux; \
    apt-get update --allow-releaseinfo-change || true; \
    apt-get install -y --no-install-recommends ca-certificates gnupg dirmngr wget; \
    for key in \
    0E98404D386FA1D9 \
    6ED0E7B82643E131 \
    F8D2585B8783D481 \
    54404762BBB6E853 \
    BDE6D2B9216EC7A8 \
    ; do \
    gpg --keyserver keyserver.ubuntu.com --recv-keys "$key" || \
    gpg --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys "$key" || true; \
    gpg --export "$key" | apt-key add - || true; \
    done; \
    apt-get update --allow-releaseinfo-change; \
    apt-get install -y build-essential git libgl1-mesa-glx && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt ./
RUN pip install --upgrade pip setuptools wheel \
    && pip install torch==2.3.0+cpu torchvision==0.18.0+cpu torchaudio==2.3.0+cpu --index-url https://download.pytorch.org/whl/cpu \
    && pip install stable-baselines3[extra] streamlit \
    && pip install -r requirements.txt

# Copy the rest of the code
COPY . .

# Expose Streamlit default port
EXPOSE 8501

# Default command: run the Streamlit dashboard
CMD ["streamlit", "run", "dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
