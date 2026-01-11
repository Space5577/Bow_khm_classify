Step of doing this project
Raw Text
   ↓ tokenizer
Tokens
   ↓ build_vocabulary
Word → Index map (vocab)
   ↓ build_bow_matrix
Word counts (BoW)
   ↓ compute_tfidf
Word importance (TF-IDF)
   ↓ TruncatedSVD (optional)
Word-combination features (components)
   ↓ Classifier
Predicted label
