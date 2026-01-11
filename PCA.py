# pca.py
from sklearn.decomposition import TruncatedSVD
from logs import logger

def apply_svd(X_train, X_eval=None, n_components=5000, return_model=False):
    """
    Fit PCA (TruncatedSVD) on X_train and optionally transform X_eval.
    """
    try:
        logger.info(f"Applying Truncated SVD | Components: {n_components}")
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        X_train_reduced = svd.fit_transform(X_train)

        X_eval_reduced = None
        if X_eval is not None:
            X_eval_reduced = svd.transform(X_eval)

        logger.info("Dimensionality reduction completed")

        if return_model:
            return X_train_reduced, X_eval_reduced, svd
        return X_train_reduced, X_eval_reduced

    except Exception as e:
        logger.error(f"Failed SVD transformation: {e}")
        raise


# if __name__ == "__main__":
#     import numpy as np
#     X_train = np.random.rand(100, 500)
#     X_test  = np.random.rand(20, 500)
#     X_train_r, X_test_r = apply_svd(X_train, X_test, n_components=50)
#     print(X_train_r.shape, X_test_r.shape)
