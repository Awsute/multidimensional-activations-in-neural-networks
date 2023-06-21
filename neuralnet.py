import numpy as np

class Model:
    input_layer = None
    hidden_layers = [None]
    def __new__(self):
        return self
    def forward(self, inputs):
        return

class Layer:
    def __new__(self):
        return self
class Node:
    activation = lambda x: np.array([])
    def __new__(self):
        return self
    def forward(self, inputs):
        return self.activation(inputs)

