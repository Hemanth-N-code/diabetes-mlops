import mlflow
from mlflow.tracking import MlflowClient

def manage_model(model_name="diabetes_classifier"):
    client = MlflowClient()
    
    # Search for the best run
    runs = mlflow.search_runs(order_by=["metrics.accuracy DESC"])
    best_run_id = runs.iloc[0].run_id
    
    # Register the best model
    model_uri = f"runs:/{best_run_id}/diabetes_model"
    mv = mlflow.register_model(model_uri, model_name)
    
    # Transition to Staging
    client.transition_model_version_stage(
        name=model_name,
        version=mv.version,
        stage="Staging"
    )
    print(f"Model version {mv.version} moved to Staging.")

if __name__ == "__main__":
    manage_model()