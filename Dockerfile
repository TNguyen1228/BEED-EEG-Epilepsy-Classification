FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependencies and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and config
COPY config/ ./config/
COPY src/ ./src/
COPY api/ ./api/
COPY BEED_Data.csv .

# Train model during build or expect mounted model
RUN python -m src.models.train

# Expose port
EXPOSE 8000

# Run API service
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]

