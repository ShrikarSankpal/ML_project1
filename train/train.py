import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
import itertools

def load_data():
    data = fetch_california_housing()
    return data.data, data.target

def get_hyperparameter_combinations():
    param_grid = {
        "n_estimators": [50, 100],
        "max_depth": [5, 10],
        "min_samples_split": [2, 5]
    }
    return list(itertools.product(
        param_grid["n_estimators"],
        param_grid["max_depth"],
        param_grid["min_samples_split"]
    ))

def train_and_log(X_train, X_test, y_train, y_test, experiment_name):
    mlflow.set_experiment(experiment_name)
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id

    print(f"Experiment: {experiment_name} (ID: {experiment_id})")

    for n_estimators, max_depth, min_samples_split in get_hyperparameter_combinations():
        run_name = f"RF_{n_estimators}_{max_depth}_{min_samples_split}"
        with mlflow.start_run(run_name=run_name) as run:
            run_id = run.info.run_id
            print(f"Run: {run_name} (ID: {run_id})")

            mlflow.set_tag("model", "RandomForestRegressor")
            mlflow.log_params({
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "min_samples_split": min_samples_split
            })

            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                random_state=42
            )
            model.fit(X_train, y_train)

            preds = model.predict(X_test)
            mse = mean_squared_error(y_test, preds)
            r2 = r2_score(y_test, preds)

            mlflow.log_metric("mse", mse)
            mlflow.log_metric("r2_score", r2)
            mlflow.sklearn.log_model(model, artifact_path="model")

            print(f"Logged run {run_id} | MSE: {mse:.2f} | R2: {r2:.2f}")

if __name__ == "__main__":
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    train_and_log(X_train, X_test, y_train, y_test, experiment_name="house_price_prediction")
