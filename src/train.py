import yaml
import mlflow
import mlflow.sklearn
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Charger la configuration
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Configurer MLflow
mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
mlflow.set_experiment(config["mlflow"]["experiment_name"])

with mlflow.start_run():
    # 1. Charger les données
    data = fetch_california_housing(as_frame=True)
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config["model"]["test_size"], random_state=config["project"]["random_state"]
    )

    # 2. Paramètres
    n_estimators = config["model"]["n_estimators"]
    max_depth = config["model"]["max_depth"]

    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)

    # 3. Entraînement
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=config["project"]["random_state"])
    model.fit(X_train, y_train)

    # 4. Évaluation
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2", r2)

    # 5. Sauvegarde MLflow
    mlflow.sklearn.log_model(model, "model")
    print(f"Entraînement terminé ! MSE: {mse:.4f}, R2: {r2:.4f}")
