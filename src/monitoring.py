import pandas as pd
import sys
import os
import mlflow.sklearn
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPreset

def check_model_and_data_drift(reference_path, current_path, model_name="diabetes_classifier"):
    # 1. Load Data
    reference = pd.read_csv(reference_path)
    current = pd.read_csv(current_path)

    # 2. Load the Staged Model from MLflow
    try:
        model = mlflow.sklearn.load_model(f"models:/{model_name}/Staging")
    except Exception as e:
        print(f"Could not load staging model: {e}")
        return

    # 3. Add Predictions and Rename Columns for Evidently
    # reference['Outcome'] is the ground truth. Evidently expects 'target'.
    reference['target'] = reference['Outcome']
    current['target'] = current['Outcome']
    
    # Use the model to generate the 'prediction' column
    reference['prediction'] = model.predict(reference.drop(["Outcome", "target"], axis=1))
    current['prediction'] = model.predict(current.drop(["Outcome", "target"], axis=1))

    # 4. Create the Report
    # We include both Data Drift (inputs) and Classification (model performance)
    drift_report = Report(metrics=[
        DataDriftPreset(),
        ClassificationPreset()
    ])
    
    drift_report.run(reference_data=reference, current_data=current)
    
    # 5. Save results
    report_path = "model_and_data_drift_report.html"
    drift_report.save_html(report_path)
    print(f"✅ Success! Integrated report generated at: {os.path.abspath(report_path)}")

if __name__ == "__main__":
    ref_path = "data/raw/diabetes.csv"
    # To see drift, current_data should ideally be new data from your API logs
    current_path = "data/new/current_data.csv" 
    
    if os.path.exists(current_path):
        check_model_and_data_drift(ref_path, current_path)
    else:
        print("Waiting for new data in 'data/new/current_data.csv' to check model drift.")