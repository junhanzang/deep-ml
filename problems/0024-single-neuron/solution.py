import math

def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> (list[float], float):
	# Your code here
	# a*w+b
	probabilities = []
	for x in features:
		z = sum(x_i * w_i for x_i, w_i in zip(x, weights)) + bias
		y_pred = 1 / (1 + math.exp(-z))
		probabilities.append(y_pred)

	mse = sum((pred - label) ** 2 for pred, label in zip(probabilities, labels)) / len(labels)
	return probabilities, mse