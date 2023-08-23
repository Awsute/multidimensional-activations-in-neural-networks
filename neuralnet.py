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
            for o in range(len(xs)):
                d_cost = 2*(self.forward(xs[o])-ys[o])
                #cycle thru each node/function
                #in each function fetch derivative of a single output with respect to each parameter
                #for each function before fetch derivative with respect to each input
                #use a sum for multiple inputs
                for j in range(len(self.layers)):
                    
                    for k in range(j,len(self.layers)):
                        
                        print()
                
                

            print()

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

#just a helper class, no need to use but can be used
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

    