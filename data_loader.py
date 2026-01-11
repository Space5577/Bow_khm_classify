# data_loader.py
from datasets import load_dataset
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from logs import logger
import numpy as np

def load_khmer_dataset():
    """
    Load the Khmer News Classification dataset.
    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test, label_encoder
    """
    try:
        logger.info("Loading Khmer dataset...")

        # Load dataset
        dataset = load_dataset("CADT-IDRI/Khmer_News_classification")
        train_df = dataset["train"].to_pandas()[["content", "label"]]
        val_df   = dataset["validation"].to_pandas()[["content", "label"]]
        test_df  = dataset["test"].to_pandas()[["content", "label"]]

        # Encode labels
        le = LabelEncoder()
        y_train_enc = le.fit_transform(train_df["label"])
        y_val_enc   = le.transform(val_df["label"])
        y_test_enc  = le.transform(test_df["label"])

        # Convert to proper types
        X_train = train_df["content"].tolist()
        y_train = np.array(y_train_enc).ravel()
        X_val   = val_df["content"].tolist()
        y_val   = np.array(y_val_enc).ravel()
        X_test  = test_df["content"].tolist()
        y_test  = np.array(y_test_enc).ravel()

        logger.info(
            f"Dataset loaded successfully | "
            f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}"
        )

        return X_train, y_train, X_val, y_val, X_test, y_test, le

    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise


if __name__ == "__main__":
    X_train, y_train, X_val, y_val, X_test, y_test, le = load_khmer_dataset()
    print(f"Train size: {len(X_train)}, Val size: {len(X_val)}, Test size: {len(X_test)}")
    print(f"Sample y_train: {y_train[:5]}")
