import numpy as np
from collections import defaultdict

# === Tokenizer avec embeddings ===
class TokenizerWithEmbedding:
    def __init__(self, embedding_dim=4):
        self.word_to_id = defaultdict(lambda: len(self.word_to_id))
        self.id_to_word = {}
        self.embedding_dim = embedding_dim
        self.embeddings = {}

    def tokenize(self, text):
        words = text.lower().split()
        tokens = [self.word_to_id[word] for word in words]
        self.id_to_word = {v: k for k, v in self.word_to_id.items()}
        return tokens

    def create_embeddings(self):
        for word_id in self.id_to_word.keys():
            self.embeddings[word_id] = np.random.randn(self.embedding_dim)

    def get_embedding(self, token_id):
        return self.embeddings.get(token_id)