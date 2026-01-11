# tokenizer_build_vocab.py
import json
from collections import Counter
from logs import logger

def khmer_tokenizer(text):
    return text.strip().split()

def build_vocabulary(corpus, tokenizer, min_freq=1, save_path="vocab.json"):
    try:
        logger.info("Building vocabulary...")
        counter = Counter()
        for doc in corpus:
            tokens = tokenizer(doc)
            counter.update(tokens)

        # Only keep tokens with min frequency
        vocab = {word: idx for idx, (word, freq) in enumerate(counter.items()) if freq >= min_freq}

        # Add <UNK> token at the end
        vocab["<UNK>"] = len(vocab)

        # Save vocab
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(vocab, f, ensure_ascii=False, indent=4)

        logger.info(f"Vocabulary built | Size: {len(vocab)} | Saved to {save_path}")
        return vocab

    except Exception as e:
        logger.error(f"Failed to build vocabulary: {e}")
        raise

if __name__ == "__main__":
    sample_corpus = ["ភ្នំពេញ ជា រាជធានី", "កម្ពុជា មាន ប្រជាជន"]
    vocab = build_vocabulary(sample_corpus, khmer_tokenizer, min_freq=1)
    print(vocab)
