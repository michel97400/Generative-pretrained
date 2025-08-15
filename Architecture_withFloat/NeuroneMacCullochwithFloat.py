import numpy as np
import math
import matplotlib.pyplot as plt

class McCullochPittsFloat:
    np.random.seed(42)  # Au début du fichier
    """
    Version modifiée du neurone de McCulloch-Pitts avec:
    - Entrées en floats (pas seulement 0 ou 1)
    - Fonction d'activation (nécessaire pour gérer les floats)
    """
    
    def __init__(self, n_inputs):
        """
        Initialise le neurone
        
        Args:
            weights: liste des poids (floats)
            bias: biais (float)
        """
        self.weights = np.random.randn(n_inputs) * np.sqrt(1/n_inputs)
        self.bias = np.random.randn() # Remplace le threshold
    
    
    
    def relu_function(self, x):
        return np.maximum(0.0, x - self.bias)  # np.maximum applique le relu élément par élément

    
    def activate(self, inputs, dropout_rate=0.2):
        """
        Calcule la sortie du neurone
        
        Args:
            inputs: liste des entrées (floats)
            
        Returns:
            sortie après application de la fonction d'activation
        """
        if np.random.random() < dropout_rate:
            return 0.0  # Neurone temporairement inactif
    
        inputs = np.array(inputs, dtype=float)
        weighted_sum = np.dot(inputs, self.weights)
        return self.relu_function(weighted_sum)
    
    def train(self, inputs, target, learning_rate=0.01, dropout_rate=0.2):
        """
        Entraîne le neurone avec une entrée et une cible
        
        Args:
            inputs: liste des entrées (floats)
            target: valeur cible (float)
            learning_rate: taux d'apprentissage
        """
        inputs = np.array(inputs, dtype=float)
        output = self.activate(inputs, dropout_rate=dropout_rate)
        
        # Calcul de l'erreur
        error = target - output
        
        # Mise à jour des poids et du biais
        self.weights += learning_rate * error * inputs
        self.bias -= learning_rate * error
        return output
    
    def __str__(self):
        return f"Neurone Float: poids={self.weights}, bias={self.bias}"




