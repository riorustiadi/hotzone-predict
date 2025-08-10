FROM python:3.13-slim

WORKDIR /app

# Install system dependencies for LightGBM
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY tuner.py .

# Copy data folder (akan ada setelah upload via FTP)
COPY data ./data

# Create vault directory in container
RUN mkdir -p vault

# Run the script
CMD ["python", "tuner.py"]