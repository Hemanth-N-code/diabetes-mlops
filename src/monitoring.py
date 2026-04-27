import pandas as pd
import sys
import os
import evidently
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
print(f"Evidently version: {evidently.__version__}")
# 1. Robust Import Logic
try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset
    print("Successfully imported from evidently.report")
except ImportError:
    try:
        from evidently import Report
        from evidently.metric_presets import DataDriftPreset
        print("Successfully imported from evidently (alternate structure)")
    except ImportError:
        print("ERROR: Evidently is not fully installed. Run: pip install evidently")
        sys.exit(1)

def check_drift(reference_data_path, current_data_path):
    # Verify files exist to avoid Pandas errors
    if not os.path.exists(reference_data_path):
        print(f"Error: Reference file not found at {reference_data_path}")
        return

    reference = pd.read_csv(reference_data_path)
    current = pd.read_csv(current_data_path)
    
    # Initialize and run the report
    drift_report = Report(metrics=[DataDriftPreset()])
    drift_report.run(reference_data=reference, current_data=current)
    
    # Save results
    report_path = "drift_report.html"
    drift_report.save_html(report_path)
    print(f"✅ Success! Drift report generated at: {os.path.abspath(report_path)}")

if __name__ == "__main__":
    # Ensure this path matches where your DVC data is stored
    # If your file is in data/raw/diabetes.csv, change the path below!
    ref_path = "data/raw/diabetes.csv" 
    check_drift(ref_path, ref_path)