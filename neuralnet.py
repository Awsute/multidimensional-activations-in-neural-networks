import numpy as np

class Model:

    def __new__(self):
        return self

class Node:
    activation = lambda x: np.array([])
    def __new__(self):
        return self

class Network:
    input_layer = []
    hidden_layers = []
    def __new__(self):
        return self
