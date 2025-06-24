"""
This is a boilerplate pipeline 'model_predict'
generated using Kedro 0.19.5
"""
import pandas as pd
import logging
from typing import Dict, Tuple, Any
import numpy as np  
import pickle



logger = logging.getLogger(__name__)

def model_predict(
        X: pd.DataFrame,
        model: Any,
        selected_features: list
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Predict using the trained model and return predictions and summary."""

    # Use only the selected features
    X_input = X[selected_features]

    # Make predictions
    y_pred = model.predict(X_input)

    # Create output DataFrame
    predictions = X.copy()
    predictions["y_pred"] = y_pred

    # Summarize predictions
    describe_serving = predictions.describe().to_dict()

    logger.info("✅ Predictions created successfully.")
    logger.info(f"# of predictions: {len(y_pred)}")

    return predictions, describe_serving