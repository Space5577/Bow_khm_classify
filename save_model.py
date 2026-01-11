# save_model.py
import os
import joblib
from logs import logger

def save_model(model, model_name, use_pca=False):
    """
    Save the trained model to a folder.
    Automatically separates models into 'with_pca' or 'without_pca'.
    
    Args:
        model: trained sklearn model
        model_name: str, e.g., "SVM", "LogisticRegression"
        use_pca: bool, whether the model was trained with PCA

    Returns:
        path: str, full path where the model is saved
    """
    try:
        # Base folder
        base_dir = "classifiers_without_pca"
        sub_dir = "with_pca" if use_pca else "without_pca"

        # Full folder path for the model
        model_dir = os.path.join(base_dir, sub_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)

        # File path
        path = os.path.join(model_dir, "model.pkl")

        # Save the model
        joblib.dump(model, path)

        logger.info(f"Model saved successfully | {path}")
        return path

    except Exception as e:
        logger.error(f"Failed to save model: {e}")
        raise
