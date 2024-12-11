FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    ffmpeg \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install requirements
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p model_cache offload

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV MODEL_CACHE_DIR=/app/model_cache
ENV HUGGING_FACE_HUB_CACHE=/app/model_cache

RUN mkdir -p /app/vector_cache
VOLUME /app/vector_cache
# Expose port
EXPOSE 8501

# Run the application
CMD ["streamlit", "run", "app.py"]