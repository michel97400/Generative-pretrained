from NeuronesMacCULLOCH_PITTS import McCullochPittsNeuron

# Implementer les différents neurone de MacCUlloch-Pitts
# et les tester avec des exemples de portes logiques

print("=== NEURONE DE McCULLOCH-PITTS ===\n")

# Exemple 1: Porte logique AND
print("1. Porte logique AND")
neuron_and = McCullochPittsNeuron(weights=[1, 1], threshold=2)
print(f"Neurone AND: {neuron_and}")


# Exemple 2: Porte logique OR
print("2. Porte logique OR")
neuron_or = McCullochPittsNeuron(weights=[1, 1], threshold=1)
print(f"Neurone OR: {neuron_or}")


# Exemple 3: Neurone avec inhibition (NOT)
print("3. Neurone avec inhibition")
# Un neurone qui s'active sauf si l'entrée inhibitrice est active
neuron_inhibition = McCullochPittsNeuron(weights=[2, -3], threshold=1)
print(f"Neurone avec inhibition: {neuron_inhibition}")
print("(entrée excitatrice: poids=2, entrée inhibitrice: poids=-3)")


# Exemple 4: Réseau simple de neurones
print("4. Réseau simple de 2 neurones")
print("Neurone 1 (AND) → Neurone 2 (NOT)")

neuron1 = McCullochPittsNeuron(weights=[1, 1], threshold=2)  # AND
neuron2 = McCullochPittsNeuron(weights=[-1], threshold=0)    # NOT

print(f"neuron1 : {neuron1}")
print(f"neuron2 : {neuron2}")



# Test des portes logiques
print("\n=== TESTS ===")
print("AND(1, 1):", neuron_and.activate([1, 1]))
print("AND(1, 0):", neuron_and.activate([1, 0]))
print("OR(0, 1):", neuron_or.activate([0, 1]))
print("OR(0, 0):", neuron_or.activate([0, 0]))
print("NOT(0):", neuron2.activate([0]))
print("NOT(1):", neuron2.activate([1]))
print("Inhibition(1, 0):", neuron_inhibition.activate([1, 0]))
print("Inhibition(1, 1):", neuron_inhibition.activate([1, 1]))