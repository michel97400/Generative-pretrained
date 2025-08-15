from NeuroneMacCullochwithFloat import McCullochPittsFloat
import numpy as np

# Créer le neurone
neuron_relu = McCullochPittsFloat(n_inputs=3)

epochs = 10

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    
    # Exemple d'entrées
    inputs = [5.0, 5.5, 2.2]
    print(f"Entrées: {inputs}")
    
    # Activation ReLU
    relu_out = neuron_relu.train(inputs, target=1.0, learning_rate=0.01)
    print(f"Sortie ReLU: {relu_out:.1f}")