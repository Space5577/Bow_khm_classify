# testing.py
import os
import json
import joblib
import numpy as np
from logs import logger
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# -----------------------------
# Label configuration
# -----------------------------
LABEL_NAMES = [
    "economic",
    "entertainment",
    "life",
    "politic",
    "sport",
    "technology"
]

NUM_CLASSES = len(LABEL_NAMES)

# Folder to save test metrics
METRIC_TEST_DIR = "metric_test"
os.makedirs(METRIC_TEST_DIR, exist_ok=True)


def test_model(model_path, X_test, y_test):
    """Load a trained model and evaluate it on the TEST set."""
    try:
        logger.info(f"Testing model: {model_path}")
        model = joblib.load(model_path)

        # Predictions
        y_pred = model.predict(X_test)

        # Accuracy
        acc = accuracy_score(y_test, y_pred)

        # AUC
        auc = None
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)
            try:
                auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")
            except Exception as e:
                logger.warning(f"AUC computation failed: {e}")

        # Classification report
        class_report = classification_report(y_test, y_pred, target_names=LABEL_NAMES, output_dict=True)

        # Print results
        print("\n====================================")
        print(f"Model: {os.path.basename(os.path.dirname(model_path))}")
        print(f"Test Accuracy: {acc:.4f}")
        if auc is not None:
            print(f"Test AUC: {auc:.4f}")
        print("\nClassification Report:")
        for label in LABEL_NAMES:
            stats = class_report[label]
            print(f"{label}: Precision={stats['precision']:.4f}, Recall={stats['recall']:.4f}, F1={stats['f1-score']:.4f}")

        logger.info(f"Testing completed | Accuracy={acc:.4f} | AUC={auc}")

        return acc, auc, class_report

    except Exception as e:
        logger.error(f"Testing failed for {model_path}: {e}")
        raise


def test_all_models(classifiers_dir, X_test_dict, y_test):
    """
    Test all trained models in classifiers_dir.
    Save metrics for each model into metric_test folder.
    """
    results = {}

    for model_name in os.listdir(classifiers_dir):
        model_folder = os.path.join(classifiers_dir, model_name)
        if not os.path.isdir(model_folder):
            continue

        model_path = os.path.join(model_folder, "model.pkl")
        metrics_path = os.path.join(model_folder, "metrics.json")

        if not os.path.exists(model_path):
            continue

        # Check if PCA was used
        use_pca = True
        if os.path.exists(metrics_path):
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)
            use_pca = metrics.get("use_pca", True)

        # Select correct test matrix
        if use_pca and "pca" in X_test_dict:
            X_test_selected = X_test_dict["pca"]
            suffix = "_PCA"
        else:
            X_test_selected = X_test_dict["tfidf"]
            suffix = "_TFIDF"

        acc, auc, class_report = test_model(model_path, X_test_selected, y_test)

        # Save metrics to file
        metric_file_path = os.path.join(METRIC_TEST_DIR, f"{model_name}{suffix}_metrics.json")
        with open(metric_file_path, "w", encoding="utf-8") as f:
            json.dump({
                "test_accuracy": acc,
                "test_auc": auc,
                "classification_report": class_report,
                "use_pca": use_pca
            }, f, indent=4, ensure_ascii=False)

        results[model_name + suffix] = {
            "test_accuracy": acc,
            "test_auc": auc,
            "classification_report": class_report,
            "use_pca": use_pca
        }

    return results


# -----------------------------
# Run tests if script executed directly
# -----------------------------
if __name__ == "__main__":
    from data_loader import load_khmer_dataset
    from bow_tfidf import build_bow_matrix, compute_tfidf
    from tokenizer import khmer_tokenizer, build_vocabulary
    from PCA import apply_svd

    logger.info("=== Starting test_all_models ===")

    # Load dataset
    X_train, y_train, X_val, y_val, X_test, y_test, label_encoder = load_khmer_dataset()

    # Build vocab & BoW
    vocab = build_vocabulary(X_train, khmer_tokenizer, min_freq=1)
    X_test_bow = build_bow_matrix(X_test, vocab, khmer_tokenizer)
    X_test_tfidf = compute_tfidf(X_test_bow)

    # Optionally apply PCA
    X_train_bow = build_bow_matrix(X_train, vocab, khmer_tokenizer)
    X_train_tfidf = compute_tfidf(X_train_bow)
    X_train_pca, X_test_pca = apply_svd(X_train_tfidf, X_test_tfidf, n_components=100)

    X_test_dict = {
        "tfidf": X_test_tfidf,
        "pca": X_test_pca
    }

    classifiers_dir = "classifiers"
    results = test_all_models(classifiers_dir, X_test_dict, y_test)

    print("\n=== FINAL TEST RESULTS ===")
    for model_name, metrics in results.items():
        print(f"{model_name} | Accuracy: {metrics['test_accuracy']:.4f} | AUC: {metrics['test_auc']}")
