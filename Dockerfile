# Use an official lightweight Python image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# --- FIX: Install curl for healthcheck ---
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
# ---------------------------------------

# Copy the requirements file first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Set the command to run the application
CMD ["python", "src/api.py"]