import os
import json
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
except ImportError:
    MLFLOW_AVAILABLE = False

from src.data.data_loader import load_and_split_data, load_config
from src.features.feature_engineering import build_preprocessing_pipeline
from src.models.evaluate import evaluate_model
from src.utils.logger import get_logger

logger = get_logger("Trainer")

def train_pipeline(config_path: str = "config/config.yaml") -> Pipeline:
    """
    End-to-end training pipeline with scikit-learn & MLflow tracking.
    """
    config = load_config(config_path)
    
    # 1. Load & Split Data
    X_train, X_test, y_train, y_test = load_and_split_data(config_path)
    rf_params = config["model"]["params"]
    
    # 2. Build Full Pipeline (Preprocessor with PSD & Spatial Features + Classifier)
    preprocessor = build_preprocessing_pipeline(include_psd=True, include_spatial=True)
    rf_classifier = RandomForestClassifier(**rf_params)
    
    full_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", rf_classifier)
    ])
    
    # 3. Setup MLflow if available
    run_context = None
    if MLFLOW_AVAILABLE:
        mlflow_cfg = config.get("mlflow", {})
        if "tracking_uri" in mlflow_cfg:
            mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
        mlflow.set_experiment(mlflow_cfg.get("experiment_name", "BEED_Default_Experiment"))
        run_context = mlflow.start_run(run_name="RandomForest_Baseline")
        mlflow.log_params(rf_params)
        mlflow.log_param("test_size", config["data"]["test_size"])
        mlflow.log_param("num_features", len(config["data"]["feature_columns"]))
        logger.info("MLflow Tracking active (Run ID: %s)", run_context.info.run_id)
    else:
        logger.warning("MLflow is not installed. Training will proceed and save model locally.")
    
    try:
        # 4. Train Model
        logger.info("Fitting Random Forest Pipeline on training set...")
        full_pipeline.fit(X_train, y_train)
        
        if rf_params.get("oob_score", False):
            oob_acc = full_pipeline.named_steps["classifier"].oob_score_
            logger.info("OOB Accuracy Score: %.4f", oob_acc)
            if MLFLOW_AVAILABLE:
                mlflow.log_metric("oob_score", oob_acc)
        
        # 5. Evaluate
        logger.info("Evaluating model on test set...")
        labels_dict = {int(k): v for k, v in config["labels"].items()}
        metrics = evaluate_model(full_pipeline, X_test, y_test, labels_dict=labels_dict)
        
        # 6. Save Model Artifacts locally
        save_path = config["model"]["saved_model_path"]
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(full_pipeline, save_path)
        logger.info("Model pipeline saved locally to %s", save_path)
        
        metrics_path = config["model"].get("metrics_output_path", "models/metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4)
        logger.info("Evaluation metrics saved to %s", metrics_path)
            
        # 7. Log to MLflow if available
        if MLFLOW_AVAILABLE:
            mlflow.log_metric("accuracy", metrics["accuracy"])
            mlflow.log_metric("f1_macro", metrics["f1_macro"])
            mlflow.log_metric("f1_weighted", metrics["f1_weighted"])
            mlflow.log_metric("precision_macro", metrics["precision_macro"])
            try:
                mlflow.sklearn.log_model(
                    sk_model=full_pipeline,
                    artifact_path="model_pipeline",
                    registered_model_name="BEED_EEG_RandomForest",
                    serialization_format="cloudpickle"
                )
            except TypeError:
                # In case older/newer mlflow versions expect skops_trusted_types
                mlflow.sklearn.log_model(
                    sk_model=full_pipeline,
                    artifact_path="model_pipeline",
                    registered_model_name="BEED_EEG_RandomForest",
                    skops_trusted_types=["src.features.feature_engineering.EEGSpectralAndSpatialFeatureExtractor"]
                )
            mlflow.log_artifact(metrics_path, artifact_path="evaluation")
            
        logger.info("Pipeline training execution finished successfully.")
        return full_pipeline

    finally:
        if MLFLOW_AVAILABLE and run_context:
            mlflow.end_run()

if __name__ == "__main__":
    train_pipeline()

