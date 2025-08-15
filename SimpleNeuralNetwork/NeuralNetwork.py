import numpy as np
import sys
import os
# Ajouter le dossier parent au path Python
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Architecture_withFloat.NeuroneMacCullochwithFloat import McCullochPittsFloat

class SimpleNeuralNetwork:
    def __init__(self, n_inputs, n_neurons):
        self.neurons = [McCullochPittsFloat(n_inputs) for _ in range(n_neurons)]
    
    def forward(self, inputs):
        return [neuron.activate(inputs) for neuron in self.neurons]
    
    def train(self, inputs, targets, learning_rate=0.01, dropout_rate=0.1):
        outputs = []
        for neuron, target in zip(self.neurons, targets):
            output = neuron.train(inputs, target, learning_rate, dropout_rate=dropout_rate)
            outputs.append(output)
        return outputs