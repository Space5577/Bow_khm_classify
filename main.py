from data_loader import load_khmer_dataset
from tokenizer import khmer_tokenizer, build_vocabulary
from bow_tfidf import build_bow_matrix, compute_tfidf
from PCA import apply_svd
from classifiers import train_and_evaluate
from logs import logger

# -----------------------------
# Configuration
# -----------------------------
USE_PCA = False            # True → run PCA models
N_COMPONENTS = 5000        # PCA components
MIN_FREQ = 1
MODEL_LIST = ["LogisticRegression", "RandomForest", "SVM"]


def main():
    logger.info("=== Starting full pipeline ===")

    # -----------------------------
    # 1. Load Data
    # -----------------------------
    X_train, y_train, X_val, y_val, X_test, y_test, label_encoder = load_khmer_dataset()

    # -----------------------------
    # 2. Vocabulary (TRAIN ONLY)
    # -----------------------------
    vocab = build_vocabulary(
        X_train,
        khmer_tokenizer,
        min_freq=MIN_FREQ
    )

    # -----------------------------
    # 3. BoW (SPARSE)
    # -----------------------------
    X_train_bow = build_bow_matrix(X_train, vocab, khmer_tokenizer)
    X_val_bow   = build_bow_matrix(X_val, vocab, khmer_tokenizer)
    X_test_bow  = build_bow_matrix(X_test, vocab, khmer_tokenizer)

    # -----------------------------
    # 4. TF-IDF (SPARSE)
    # -----------------------------
    X_train_tfidf = compute_tfidf(X_train_bow)
    X_val_tfidf   = compute_tfidf(X_val_bow)
    X_test_tfidf  = compute_tfidf(X_test_bow)

    logger.info("TF-IDF matrices ready (sparse)")

    # =====================================================
    # 5. Train WITHOUT PCA (SVM FAST PATH)
    # =====================================================
    logger.info("=== Training models WITHOUT PCA ===")

    for model_name in MODEL_LIST:
        logger.info(f"Training {model_name} without PCA")

        train_and_evaluate(
            X_train=X_train_tfidf,
            y_train=y_train,
            X_eval=X_val_tfidf,
            y_eval=y_val,
            label_encoder=label_encoder,
            model_name=model_name,
            use_pca=False,
            X_test=X_test_tfidf,
            y_test=y_test
        )

    # =====================================================
    # 6. Train WITH PCA (OPTIONAL)
    # =====================================================
    # if USE_PCA:
    #     logger.info(f"=== Applying SVD (n_components={N_COMPONENTS}) ===")

    #     X_train_pca, X_val_pca, svd_model = apply_svd(
    #         X_train_tfidf,
    #         X_val_tfidf,
    #         n_components=N_COMPONENTS,
    #         return_model=True
    #     )

    #     X_test_pca = svd_model.transform(X_test_tfidf)

    #     logger.info("=== Training models WITH PCA ===")

    #     for model_name in MODEL_LIST:
    #         logger.info(f"Training {model_name} with PCA")

    #         train_and_evaluate(
    #             X_train=X_train_pca,
    #             y_train=y_train,
    #             X_eval=X_val_pca,
    #             y_eval=y_val,
    #             label_encoder=label_encoder,
    #             model_name=model_name,
    #             use_pca=True,
    #             X_test=X_test_pca,
    #             y_test=y_test
    #         )

    logger.info("=== Pipeline completed successfully ===")


if __name__ == "__main__":
    main()
