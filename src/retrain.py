import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

def retrain_model(ref_path, cur_path, n_estimators=100, max_depth=5):
    # 1. Load and Combine Data
    df_ref = pd.read_csv(ref_path)
    df_cur = pd.read_csv(cur_path)
    combined_df = pd.concat([df_ref, df_cur], ignore_index=True)
    
    # 2. Save the updated dataset to disk for DVC versioning
    combined_df.to_csv(ref_path, index=False)
    print(f"💾 Dataset updated at {ref_path} (Total rows: {len(combined_df)})")

    # 3. Prepare Data
    X = combined_df.drop("Outcome", axis=1)
    y = combined_df["Outcome"]

    # Use the same split logic as your train code
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=60
    )

    # 4. Start MLflow Run
    with mlflow.start_run(run_name="Retraining_After_Drift"):
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
        model.fit(X_train, y_train)
        
        # 5. Evaluate
        predictions = model.predict(X_test)
        acc = accuracy_score(y_test, predictions)

        # 6. Log to MLflow
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("data_source", "combined_drifted_data")
        mlflow.log_metric("accuracy", acc)
        
        mlflow.sklearn.log_model(model, "diabetes_model")
        print(f"✅ Retrained model accuracy: {acc}")

if __name__ == "__main__":
    # Ensure these paths are correct for your VS Code structure
    ref_file = "data/raw/diabetes.csv"
    cur_file = "data/new/current_data.csv"
    
    if os.path.exists(cur_file):
        retrain_model(ref_file, cur_file)
    else:
        print(f"Error: {cur_file} not found. Cannot retrain.")