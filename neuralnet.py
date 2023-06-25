import numpy as np

class Model:
    layers = [None]
    def __init__(self, layers):
        self.layers = layers
    
    def forward(self, inputs):
        prev_outputs = inputs
        for i in range(len(self.layers)):
            prev_outputs = self.layers[i].forward(prev_outputs)

            if i+1 < len(self.layers):
                intermediate_list = [[0]*len(self.layers[i].nodes)]*len(self.layers[i+1].nodes)
                
                for o in range(len(prev_outputs)):
                    for j in range(len(prev_outputs[o])):
                        intermediate_list[j][o] = prev_outputs[o][j]
                
                prev_outputs = intermediate_list
                print(intermediate_list)

        return prev_outputs

class Layer:
    nodes = [None]
    def __init__(self, nodes):
        self.nodes = nodes

    def forward(self, input):
        outputs = []
        for i in range(len(self.nodes)):
            outputs.append(self.nodes[i].forward(input[i]))
        return outputs

class Node:
    #each node has an input list and a coefficient list
    #output of node is a list
    activation = lambda x, a: np.array([])
    coef = [1]
    def __init__(self, activation, coef):
        self.activation = activation
        self.coef = coef
    
    def forward(self, input):
        return self.activation(input, self.coef)


model = Model([
    Layer(
        [Node(lambda x, a: [a[0]*x[0], a[0]-x[0]], [0.5])]
    ),
        Layer(
        [Node(lambda x, a: [a[0]*x[0], x[0]], [0.1]), Node(lambda x, a: [a[0]*x[0],x[0]], [-0.1])]
    ),
    Layer(
        [Node(lambda x, a: [a[0]*x[0]+x[1]], [0.1]), Node(lambda x, a: [a[0]*x[0]+x[1]], [-0.1])]
    )
])

print(model.forward([[0.63]]))