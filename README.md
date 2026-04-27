# 🏥 End-to-End Diabetes MLOps System

This project is a production-grade MLOps pipeline designed to predict diabetes while maintaining a high standard of data versioning, experiment tracking, and automated deployment. It features a "closed-loop" retraining mechanism to handle data drift.

---

## 🏗️ 1. System Architecture

The system is built on four pillars of MLOps:

* **Data Versioning (DVC):** Tracks the state of the 768-row baseline and 131-row new data.
* **Experiment Tracking (MLflow):** Logs every training run, parameter, and model version.
* **Model Serving (FastAPI):** A high-performance REST API for real-time predictions.
* **Containerization (Docker):** Standardizes the environment to ensure "it works on any machine".

---

## 📉 2. The Drift Story (Detection & Recovery)

In this project, we simulated a real-world data science crisis:

1. **Detection:** A new batch of data was introduced that caused a performance drop.
2. **Versioning:** We used `dvc checkout --force` to instantly revert to our last "Golden State" when needed.
3. **Retraining:** We executed `src/retrain.py` to merge the new data, update the baseline, and register **Version 4** of our model—successfully adapting to the new data distribution.

---

## 🚀 3. Installation & Setup

### Step 1: Environment Preparation

```powershell
# Create and activate a virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install all dependencies (Python 3.12 optimized)
pip install -r requirements.txt
```

### Step 2: Data Initialization (DVC)

```powershell
# Pull the actual data files managed by DVC pointers
dvc pull
```

---

## 🛠️ 4. Development & Training

### Step 3: Run Initial Training

```powershell
# Trains the model, scales features, and logs to MLflow
python src/train.py
```

### Step 4: Manage the Model Registry

```powershell
# Promotes the best performing model to 'Staging' for deployment
python src/model_manager.py
```

### Step 5: Monitor for Drift

```powershell
# Compares current data vs. baseline and generates an HTML report
python src/monitoring.py
```

---

## 🐳 5. Production Deployment (Docker)

We use Docker to ensure the API runs in a consistent Linux environment, regardless of the host OS.

### Step 6: Build the Docker Image

```powershell
# Build the image and tag it as 'diabetes-mlops-app'
docker build -t diabetes-mlops-app .
```

### Step 7: Run the Containerized API

```powershell
# Run the container, mapping port 8000 and mounting local model files
# We use a volume mount (-v) so the container sees our MLflow artifacts
docker run -p 8000:8000 -v ${PWD}:/app diabetes-mlops-app
```

* **API UI:** http://localhost:8000/docs
* **Health Check:** http://localhost:8000/

---

## 🧪 6. Final Verification (Example Request)

Once the Docker container is running, test the endpoint with:

```json
{
  "Pregnancies": 2,
  "Glucose": 135,
  "BloodPressure": 80,
  "SkinThickness": 35,
  "Insulin": 150,
  "BMI": 28.5,
  "DiabetesPedigreeFunction": 0.6,
  "Age": 45
}
```

---

## 🛠️ Tech Stack

* **Core:** Python 3.12, Scikit-Learn, Pandas
* **Tracking:** MLflow (Registry + UI)
* **Data Ops:** DVC (Data Version Control)
* **Serving:** FastAPI & Uvicorn
* **Infrastructure:** Docker & GitHub Actions
