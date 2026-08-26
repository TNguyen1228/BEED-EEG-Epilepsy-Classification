from typing import Dict, Any
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from src.utils.logger import get_logger

logger = get_logger("Evaluation")

def evaluate_model(model, X_test, y_test, labels_dict: Dict[int, str] = None) -> Dict[str, Any]:
    """
    Computes comprehensive evaluation metrics for multi-class classification.
    """
    y_pred = model.predict(X_test)
    
    accuracy = float(accuracy_score(y_test, y_pred))
    f1_macro = float(f1_score(y_test, y_pred, average="macro"))
    f1_weighted = float(f1_score(y_test, y_pred, average="weighted"))
    precision_macro = float(precision_score(y_test, y_pred, average="macro"))
    recall_macro = float(recall_score(y_test, y_pred, average="macro"))
    
    target_names = [labels_dict[i] for i in sorted(labels_dict.keys())] if labels_dict else None
    report_dict = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
    report_str = classification_report(y_test, y_pred, target_names=target_names)
    cm = confusion_matrix(y_test, y_pred).tolist()
    
    metrics = {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "confusion_matrix": cm,
        "classification_report": report_dict
    }
    
    logger.info("Evaluation Summary:\n%s", report_str)
    logger.info("Accuracy: %.4f | F1-Macro: %.4f", accuracy, f1_macro)
    
    return metrics

