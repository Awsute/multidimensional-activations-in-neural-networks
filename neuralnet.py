#Code for a neural network where each input of a neuron is a separate parameter in the activation function of that neuron
#The output length of each neuron is a list of the same length as the number of neurons in the next layer
#Each output of a single neuron is an input to a specific neuron in the next layer
#I am guessing this will allow for a more complex network that can solve problems in a much different manner as compared to a traditional neural network of the same dimensions

import math
import random
class Activation:
    function = lambda x, a: []
    derivative_x = lambda x, a: []
    derivative_a = lambda x, a: []
    
    def __init__(self, function, derivative_x, derivative_a):
        self.function = function
        self.derivative_x = derivative_x
        self.derivative_a = derivative_a

class Model:
    layers = [None]
    def __init__(self, layers):
        self.layers = layers
    

    #full forward propogation algorithm
    def forward(self, inputs):
        prev_outputs = inputs
        for i in range(len(self.layers)):
            prev_outputs = self.layers[i].forward(prev_outputs)

            #if not last layer then transpose list of outputs
            if i+1 < len(self.layers):
                intermediate_list = [[0]*len(self.layers[i].nodes)]*len(self.layers[i+1].nodes)
                
                for o in range(len(prev_outputs)):
                    for j in range(len(prev_outputs[o])):
                        intermediate_list[j][o] = prev_outputs[o][j]
                
                prev_outputs = intermediate_list
                print(intermediate_list)

        return prev_outputs

    def fit(self, xs, ys, epochs):
        for i in range(epochs):
            print

class Layer:
    nodes = [None]
    def __init__(self, nodes):
        self.nodes = nodes

    #forward propogation helper function
    def forward(self, input):
        outputs = []
        for i in range(len(self.nodes)):
            outputs.append(self.nodes[i].forward(input[i]))
        return outputs

#just a helper class, do not need to use but can be used
class Node:
    #each node has an input list and a coefficient list
    #output of node is a list
    activation = None
    coef = [1]
    def __init__(self, activation, coef):
        self.activation = activation
        self.coef = coef
    #forward propogation helper helper function
    def forward(self, input):
        return self.activation.function(input, self.coef)

def node_hyper_relu(input_size):
    def relu(x, c):
        if x < c:
            return 0
        else:
            return x
    
    def d_relu(x, c):
        if x < c:
            return 0
        else:
            return 1
    
    def da(x, a):
        output = []
        for i in range(input_size):
            output.append(relu(x,a[3*i+1]))
            output.append(a[3*i]*relu(-x,-a[3*i+1])+a[3*i+2])
            output.append(1)
        return output
    
    
    fn = lambda x, a: [(a[3*i]*relu(x[i], a[3*i+1])+a[3*i+2]) for i in range(input_size)]
    dfdx = lambda x, a: [(a[3*i]*d_relu(x[i], a[3*i+1])) for i in range(input_size)]
    dfda = lambda x, a: da(x, a)
    a = [random.random() for i in range(3*input_size)]
    return Node(Activation(fn,dfdx, dfda), a)
    