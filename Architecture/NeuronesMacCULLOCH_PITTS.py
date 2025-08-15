import numpy as np

class McCullochPittsNeuron:
    """
    Implémentation du neurone de McCulloch-Pitts (1943)
    - Entrées binaires (0 ou 1)
    - Poids fixes (pas d'apprentissage)
    - Fonction de seuil binaire
    """
    
    def __init__(self, weights, threshold):
        """
        Initialise le neurone
        
        Args:
            weights: liste des poids pour chaque entrée
            threshold: seuil d'activation
        """
        self.weights = np.array(weights)
        self.threshold = threshold
    
    def activate(self, inputs):
        """
        Calcule la sortie du neurone
        
        Args:
            inputs: liste des entrées binaires (0 ou 1)
            
        Returns:
            1 si la somme pondérée >= seuil, sinon 0
        """
        inputs = np.array(inputs)
        
        # Vérification que les entrées sont binaires
        if not all(x in [0, 1] for x in inputs):
            raise ValueError("Les entrées doivent être binaires (0 ou 1)")
        
        # Calcul de la somme pondérée
        weighted_sum = np.dot(inputs, self.weights)
        
        # Application de la fonction seuil
        return 1 if weighted_sum >= self.threshold else 0
    
    def __str__(self):
        return f"Neurone MP: poids={self.weights}, seuil={self.threshold}"


