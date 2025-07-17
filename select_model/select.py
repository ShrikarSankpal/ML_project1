import mlflow
from mlflow.tracking import MlflowClient
import os
from paths import BEST_MODEL_DIR, EXPERIMENT_NAME

def select_best_model():
    client = MlflowClient()
    
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise ValueError(f"No experiment named '{EXPERIMENT_NAME}' found.")
    
    experiment_id = experiment.experiment_id
    print(f"Searching best model from experiment: {EXPERIMENT_NAME} (ID: {experiment_id})")

    # Search for runs sorted by highest r2_score
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        order_by=["metrics.r2_score DESC"],
        max_results=1,
    )

    if not runs:
        raise ValueError("No runs found in the experiment.")
    
    best_run = runs[0]
    best_run_id = best_run.info.run_id
    print(f"Best run ID: {best_run_id}")
    print(f"R2 Score: {best_run.data.metrics['r2_score']}")

    # Download the model from artifact store
    model_uri = f"runs:/{best_run_id}/model"
    local_path = mlflow.artifacts.download_artifacts(artifact_uri=model_uri, dst_path=BEST_MODEL_DIR)
    print(f"Downloaded best model to: {local_path}")

    # Optionally write the run_id to a file for serving
    with open(os.path.join(BEST_MODEL_DIR, "run_id.txt"), "w") as f:
        f.write(best_run_id)

if __name__ == "__main__":
    select_best_model()
