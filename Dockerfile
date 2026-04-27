# Use a lightweight Python 3.12 image
FROM python:3.12-slim

# Set environment variables to ensure Python output is logged correctly
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /app

# Install system dependencies for build-essential (needed for some pip packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

# Expose the FastAPI port
EXPOSE 8000

# Run the app. We use 0.0.0.0 so it's accessible outside the container
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]