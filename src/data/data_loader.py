import os
from typing import Tuple, Dict, Any
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from src.utils.logger import get_logger

logger = get_logger("DataLoader")

def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Loads configuration file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config

def validate_data(df: pd.DataFrame, config: Dict[str, Any]) -> bool:
    """
    Validates dataset against schema constraints:
    - Checks presence of required features and target.
    - Checks missing values.
    - Checks valid class labels.
    """
    feature_cols = config["data"]["feature_columns"]
    target_col = config["data"]["target_column"]
    
    # Check column presence
    missing_cols = [col for col in feature_cols + [target_col] if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in dataset: {missing_cols}")
    
    # Check null values
    null_counts = df[feature_cols + [target_col]].isnull().sum().sum()
    if null_counts > 0:
        raise ValueError(f"Dataset contains {null_counts} missing/null values.")
    
    # Check target classes
    unique_labels = sorted(df[target_col].unique())
    expected_labels = sorted([int(k) for k in config["labels"].keys()])
    if unique_labels != expected_labels:
        raise ValueError(f"Invalid target labels: found {unique_labels}, expected {expected_labels}")
    
    logger.info("Data validation passed successfully: %d rows, %d features", len(df), len(feature_cols))
    return True

def load_and_split_data(config_path: str = "config/config.yaml") -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Loads raw BEED EEG dataset, validates it, and performs stratified split.
    """
    config = load_config(config_path)
    raw_path = config["data"]["raw_path"]
    
    logger.info("Loading raw dataset from %s", raw_path)
    df = pd.read_csv(raw_path)
    
    validate_data(df, config)
    
    feature_cols = config["data"]["feature_columns"]
    target_col = config["data"]["target_column"]
    
    X = df[feature_cols]
    y = df[target_col]
    
    test_size = config["data"]["test_size"]
    random_state = config["data"]["random_state"]
    stratify = y if config["data"].get("stratify", True) else None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )
    
    logger.info("Data split completed: Train shape=%s, Test shape=%s", X_train.shape, X_test.shape)
    return X_train, X_test, y_train, y_test

