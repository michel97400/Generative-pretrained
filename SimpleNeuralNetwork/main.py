from NeuralNetwork import SimpleNeuralNetwork
import numpy as np

import sys
import os
# Ajouter le dossier parent au path Python
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Tokenizer.Tokenizer_word import TokenizerWithEmbedding


# === Données ===
phrases = [
    "Le chat mange",
    "Le chien dort"
]

# === Tokenisation et vocabulaire ===
tokenizer = TokenizerWithEmbedding(embedding_dim=4)
for phrase in phrases:
    tokenizer.tokenize(phrase)
tokenizer.create_embeddings()

# === Création des paires (input_emb, target_onehot) ===
training_data = []
vocab_size = len(tokenizer.word_to_id)

for phrase in phrases:
    tokens = tokenizer.tokenize(phrase)
    for i in range(len(tokens) - 1):
        inp_emb = tokenizer.get_embedding(tokens[i])
        target_onehot = np.zeros(vocab_size)
        target_onehot[tokens[i+1]] = 1
        training_data.append((inp_emb, target_onehot))

# === Création du réseau ===
net = SimpleNeuralNetwork(n_inputs=4, n_neurons=vocab_size)

# === Entraînement ===
for epoch in range(1000):
    total_error = 0
    for emb, target in training_data:
        outputs = net.train(emb, target, learning_rate=0.05)
        total_error += sum(abs(target - outputs))
    if epoch % 100 == 0:
        print(f"Epoch {epoch} - Erreur: {total_error:.4f}")

# === Génération ===
start_word = "le"
start_token = tokenizer.word_to_id[start_word]
current_emb = tokenizer.get_embedding(start_token)

generated = [start_word]
for _ in range(5):
    outputs = net.forward(current_emb)
    next_token = int(np.argmax(outputs))
    generated.append(tokenizer.id_to_word[next_token])
    current_emb = tokenizer.get_embedding(next_token)

print("\nPhrase générée:", " ".join(generated))