from NeuralNetwork import SimpleNeuralNetwork
import numpy as np

# Fonction de normalisation
def normalize_data(data, min_val, max_val):
    return (data - min_val) / (max_val - min_val)

# Données d'entraînement étendues
training_data = [
    ([20.0, 65.0, 10.0], [22.0, 15.0]),
    ([22.0, 70.0, 15.0], [23.0, 16.0]),
    ([19.0, 80.0, 20.0], [20.0, 14.0]),
    ([21.0, 75.0, 12.0], [22.5, 15.5]),
    ([23.0, 60.0, 8.0], [24.0, 17.0]),
    ([18.0, 85.0, 25.0], [19.0, 13.0])
]

# Normalisation des données
temp_min, temp_max = 0, 40  # Plage de température
humidity_min, humidity_max = 0, 100  # Plage d'humidité
wind_min, wind_max = 0, 50  # Plage de vent

# Normaliser les données d'entraînement
normalized_data = []
for inputs, targets in training_data:
    norm_inputs = [
        normalize_data(inputs[0], temp_min, temp_max),
        normalize_data(inputs[1], humidity_min, humidity_max),
        normalize_data(inputs[2], wind_min, wind_max)
    ]
    norm_targets = [
        normalize_data(targets[0], temp_min, temp_max),
        normalize_data(targets[1], temp_min, temp_max)
    ]
    normalized_data.append((norm_inputs, norm_targets))

# Créer et entraîner le réseau
network = SimpleNeuralNetwork(n_inputs=3, n_neurons=2)

print("Entraînement du modèle...")
for epoch in range(1000):  # Plus d'epochs
    total_error = 0
    for inputs, targets in normalized_data:
        outputs = network.train(inputs, targets, learning_rate=0.1)  # Learning rate plus élevé
        error = sum(abs(t-o) for t, o in zip(targets, outputs))
        total_error += error
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Erreur moyenne = {total_error/len(normalized_data):.4f}")

# Test avec nouvelles données
new_data = [21.0, 68.0, 12.0]
norm_new_data = [
    normalize_data(new_data[0], temp_min, temp_max),
    normalize_data(new_data[1], humidity_min, humidity_max),
    normalize_data(new_data[2], wind_min, wind_max)
]

predictions = network.forward(norm_new_data)
# Dénormaliser les prédictions
pred_max = predictions[0] * (temp_max - temp_min) + temp_min
pred_min = predictions[1] * (temp_max - temp_min) + temp_min

print("\nTest du modèle:")
print(f"Entrées: Température hier={new_data[0]}°C, Humidité={new_data[1]}%, Vent={new_data[2]}km/h")
print(f"Prédictions: Temp max={pred_max:.1f}°C, Temp min={pred_min:.1f}°C")