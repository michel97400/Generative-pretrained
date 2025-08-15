from NeuroneMacCullochwithFloat import McCullochPittsFloat
import numpy as np

# Créer le neurone
neuron_relu = McCullochPittsFloat(n_inputs=3)

# Exemple d'activation avec des entrées en float
inputs = [5.0, 5.5, 2.2]
target = 2.0

print("État initial:")
print(f"Poids: {neuron_relu.weights}")
print(f"Bias: {neuron_relu.bias}\n")

for epoch in range(10):
    relu_out = neuron_relu.train(inputs, target=target, learning_rate=0.01, dropout_rate=0)
    error = target - relu_out
    print(f"Epoch {epoch + 1}/{10}")
    print(f"Sortie: {relu_out:.3f}")
    print(f"Erreur: {error:.3f}\n")

print("État final:")
print(f"Poids: {neuron_relu.weights}")
print(f"Bias: {neuron_relu.bias}")