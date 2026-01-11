# bow_tfidf.py
import numpy as np
import json
from logs import logger

def load_vocab(vocab_path="vocab.json"):
    """Load vocabulary from JSON file (do NOT add extra <UNK>)"""
    try:
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        logger.info(f"Vocabulary loaded | Size: {len(vocab)}")
        return vocab
    except Exception as e:
        logger.error(f"Failed to load vocab.json: {e}")
        raise

def build_bow_matrix(corpus, vocab, tokenizer):
    """Build BoW matrix; unknown tokens go to <UNK>"""
    try:
        logger.info("Building BoW matrix...")
        X = np.zeros((len(corpus), len(vocab)), dtype=int)
        unk_idx = vocab["<UNK>"]
        for i, doc in enumerate(corpus):
            tokens = tokenizer(doc)
            for token in tokens:
                idx = vocab.get(token, unk_idx)  # fallback to <UNK>
                X[i, idx] += 1
        logger.info("BoW matrix built successfully")
        return X
    except Exception as e:
        logger.error(f"Failed to build BoW matrix: {e}")
        raise

def compute_tfidf(bow_matrix):
    try:
        logger.info("Computing TF-IDF matrix...")
        N = bow_matrix.shape[0]
        df = np.count_nonzero(bow_matrix, axis=0)
        idf = np.log((N + 1) / (df + 1)) + 1
        tf = bow_matrix / (bow_matrix.sum(axis=1, keepdims=True) + 1e-9)
        tfidf = tf * idf
        logger.info("TF-IDF matrix computed successfully")
        return tfidf
    except Exception as e:
        logger.error(f"Failed to compute TF-IDF: {e}")
        raise
