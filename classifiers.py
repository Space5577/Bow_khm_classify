import json
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, classification_report
from sklearn.preprocessing import label_binarize
from save_model import save_model
from logs import logger

def train_and_evaluate(
    X_train, y_train,
    X_eval, y_eval,
    label_encoder=None,
    model_name="LogisticRegression",
    use_pca=False,
    X_test=None,
    y_test=None
):
    """
    Train a model and evaluate on train / validation / optional test set.
    Fully compatible with LinearSVC (SVM), LogisticRegression, RandomForest.
    Handles multiclass AUC properly for SVM.
    """
    try:
        logger.info(f"Training {model_name} | PCA applied: {use_pca}")

        # Convert labels to numpy
        y_train = np.asarray(y_train, dtype=np.int64).ravel()
        y_eval  = np.asarray(y_eval, dtype=np.int64).ravel()

        # -----------------------------
        # Model selection
        # -----------------------------
        if model_name == "LogisticRegression":
            model = LogisticRegression(max_iter=5000, solver="saga", n_jobs=-1)
            supports_proba = True

        elif model_name == "RandomForest":
            model = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
            supports_proba = True

        elif model_name == "SVM":
            model = LinearSVC(C=1.0, max_iter=5000)
            supports_proba = False

        else:
            raise ValueError(f"Unknown model: {model_name}")

        # -----------------------------
        # Train
        # -----------------------------
        model.fit(X_train, y_train)

        # -----------------------------
        # Predictions and accuracy
        # -----------------------------
        y_eval_pred = model.predict(X_eval)
        acc = accuracy_score(y_eval, y_eval_pred)

        # -----------------------------
        # Compute AUC
        # -----------------------------
        if supports_proba:
            y_eval_score = model.predict_proba(X_eval)
            if len(np.unique(y_eval)) == 2:
                auc = roc_auc_score(y_eval, y_eval_score[:, 1])
            else:
                auc = roc_auc_score(y_eval, y_eval_score, multi_class="ovr", average="macro")

            train_loss = log_loss(y_train, model.predict_proba(X_train))
            val_loss   = log_loss(y_eval, y_eval_score)

        else:
            # LinearSVC: use decision_function
            scores = model.decision_function(X_eval)
            n_classes = scores.shape[1] if scores.ndim > 1 else 1

            if n_classes == 1:
                # Binary case
                auc = roc_auc_score(y_eval, scores)
            else:
                # Multiclass case: convert y to one-hot
                y_eval_bin = label_binarize(y_eval, classes=np.arange(n_classes))
                auc = roc_auc_score(y_eval_bin, scores, multi_class="ovr", average="macro")

            train_loss = None
            val_loss = None

        metrics = {
            "accuracy": acc,
            "auc": auc,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "use_pca": use_pca
        }

        # -----------------------------
        # Optional test set
        # -----------------------------
        if X_test is not None and y_test is not None:
            y_test = np.asarray(y_test, dtype=np.int64).ravel()
            y_test_pred = model.predict(X_test)
            metrics["test_accuracy"] = accuracy_score(y_test, y_test_pred)

        # -----------------------------
        # Save model & metrics
        # -----------------------------
        model_path = save_model(model, model_name, use_pca)
        metrics_path = model_path.replace("model.pkl", "metrics.json")

        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4)

        # -----------------------------
        # Print report
        # -----------------------------
        print(f"\n=== {model_name} ===")
        print(f"Accuracy: {acc:.4f} | AUC: {auc:.4f}")

        if label_encoder is not None:
            target_names = [str(c) for c in label_encoder.classes_]
            print(classification_report(y_eval, y_eval_pred, target_names=target_names))

        logger.info(f"{model_name} training completed | Metrics saved")
        return model, metrics

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
