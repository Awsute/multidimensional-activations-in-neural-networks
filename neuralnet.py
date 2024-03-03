#Code for a neural network where each input of a neuron is a separate parameter in the activation function of that neuron
#The output length of each neuron is a list of the same length as the number of neurons in the next layer
#Each output of a single neuron is an input to a specific neuron in the next layer
#I am guessing this will allow for a more complex network that can solve problems in a much different manner as compared to a traditional neural network of the same dimensions


import math
import random
from enum import Enum





def foldl(init, func, lis):
    if lis != []:
        return foldl(func(init,lis[0]), func, lis[1:])
    else:
        return init

def foldr(init, func, lis):
    if lis != []:
        return foldr(func(init,lis[-1:][0]), func, lis[:-1])
    else:
        return init

def transpose(lis):
    intermediate_list = [[0]*len(lis)]*len(lis[0])
    for i in range(len(lis)):
        for o in range(len(lis[o])):
            intermediate_list[o][j] = lis[j][o]
    return intermediate_list


class Activation:
    function = lambda x, a: []
    derivative_x = lambda x, a: []
    derivative_a = lambda x, a: []
    
    def __init__(self, function, derivative_x, derivative_a):
        self.function = function
        self.derivative_x = derivative_x
        self.derivative_a = derivative_a
    
def sigmoid(x):
    return 1/(1+math.e**(-x))

def d_sigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))

def tanh(x):
    return math.tanh(x)

def d_tanh(x):
    return (1/math.cosh(x))**2


#'activation_name' : lambda ins, outs: Node(
#    activation = Activation(
#        lambda x,a: f(x,a),
#        lambda x,a: df/dx,
#        lambda x,a: df/da
#    ),
#    []
#),

activations = {
    'linear_sum' : lambda ins, outs: Node(
        activation = Activation(
            lambda x,a: [sum([x[i]*a[i+1] for i in range(ins)])+a[0] for o in range(outs)],
            lambda x,a: [[a[i+1] for o in range(outs)] for i in range(ins)],
            lambda x,a: [[1 for o in range(outs)]]+[[x[i] for o in range(outs)] for i in range(ins)]
        ),
        coef = [random.uniform(-1,1) for i in range(ins+1)]
    ),
}


        

#just a helper class, no need to use but can be used
class Node:
    #each node has an input list and a coefficient list
    #output of node is a list
    activation = None
    activation_type = ''

    coef = [1]
    def __init__(self, activation_type = '', activation = None, coef = [1]):
        self.activation = activation
        self.coef = coef
        self.activation_type = activation_type
    
    def empty(self, num_coefs):
        return Node(None, [random.random() for x in range(num_coefs)])
    
    #forward propogation helper helper function
    def forward(self, input):
        return self.activation.function(input, self.coef)
    

class Layer:
    nodes = [None]
    
    def __init__(self, num_nodes = 0, activation_type = '', nodes = [None]):
        if nodes[0] != None:
            self.nodes = nodes
        else:
            self.nodes = [Node(activation_type) for i in range(num_nodes)]
    
    def compile(self,ins,outs):
        for i in range(len(self.nodes)):
            self.nodes[i] = activations[self.nodes[i].activation_type](ins, outs)
        
    
    #forward propogation helper function
    def forward(self, input):
        outputs = []
        for i in range(len(self.nodes)):
            outputs.append(self.nodes[i].forward(input[i]))
        return outputs


class Model:
    layers = [None]
    
    def __init__(self, layers):
        self.layers = layers
    
    def compile(self):
        self.layers[0].compile(1, len(self.layers[1].nodes))
        for i in range(1,len(self.layers)-1):
            self.layers[i].compile(len(self.layers[i-1].nodes), len(self.layers[i+1].nodes))
        self.layers[-1].compile(len(self.layers[-2].nodes),1)


    #full forward propogation algorithm
    def forward(self, inputs):
        state = []
        prev_outputs = inputs
        for i in range(len(self.layers)):
            state.append(prev_outputs)
            prev_outputs = self.layers[i].forward(prev_outputs)
            #if not last layer then transpose list of outputs
            if i+1 < len(self.layers):
                intermediate_list = [[0]*len(self.layers[i].nodes)]*len(self.layers[i+1].nodes)
                
                for o in range(len(prev_outputs)):
                    for j in range(len(prev_outputs[o])):
                        intermediate_list[j][o] = prev_outputs[o][j]
                prev_outputs = intermediate_list

        return prev_outputs, state
    
    def fit(self, xs, ys, epochs, dloss, learning_rate):

        for i in range(epochs):

            for o in range(len(xs)):
                out, state = self.forward(xs[o])
                d_cost = dloss(out,ys[o])
                
                
                #Get the chained derivative to a certain parameter
                def derivative_param(i,j,a,p):
                    if i == len(self.layers):
                        return 1
                    node = self.layers[i].nodes[j]
                    if p:
                        #[[do1/dx1,do2/dx1],[do1/dx_n,do2/dx_n,do_n/dx_n]]
                        d_a = node.activation.derivative_x(state[i][j],node.coef)[a]

                        
                    else:
                        #[[do1/da1,do2/da1],[do1/da_n,do2/da_n,do_n/da_n]]
                        d_a = node.activation.derivative_a(state[i][j],node.coef)[a]

                    
                    chain = sum([(d_a[x]*derivative_param(i+1, x, j, True)) for x in range(len(d_a))])
                    return chain
                
                #cycle thru each node/function
                #in each function fetch derivative of a single output with respect to each parameter
                #for each function before fetch derivative with respect to each input
                #use a sum for multiple inputs

                
                for j in range(len(self.layers)):
                    #iterate thru each layer
                    #j will be index of current layer
                    
                    
                    
                    for k in range(len(self.layers[j].nodes)):
                        #for each node in current layer
                        
                        node = self.layers[j].nodes[k]
                        for a in range(len(node.coef)):
                            node.coef[a] += d_cost*learning_rate*derivative_param(j,k,a,False)





##2 inputs, 1 output, 2 coefficients
#a0 = Activation(
#    lambda x,a: [tanh(a[0]*x[0])+tanh(a[1]*x[1])],
#    lambda x,a: [[a[0]*d_tanh(a[0]*x[0])], [a[1]*d_tanh(a[1]*x[1])]],
#    lambda x,a: [[x[0]*d_tanh(a[0]*x[0])], [x[1]*d_tanh(a[1]*x[1])]],
#)
#a01 = Activation(
#    lambda x,a: [a[0]*x[0]+a[1]*x[1]],
#    lambda x,a: [[a[0]], [a[1]]],
#    lambda x,a: [[x[0]], [x[1]]],
#)
#
##2 inputs, 2 outputs, 2 coefficients
#a1 = Activation(
#    lambda x,a: [tanh(a[0]*x[0])+a[1]*x[1], a[0]*x[0]+tanh(a[1]*x[1])],
#    lambda x,a: [[a[0]*d_tanh(a[0]*x[0]), a[1]], [a[0], a[1]*d_tanh(a[1]*x[1])]],
#    lambda x,a: [[x[0]*d_tanh(a[0]*x[0]), x[1]], [x[0], x[1]*d_tanh(a[1]*x[1])]],
#)
#a11 = Activation(
#    lambda x,a: [a[0]*x[0]+a[1]*x[1], a[0]*x[0]+a[1]*x[1]],
#    lambda x,a: [[a[0], a[1]], [a[0], a[1]]],
#    lambda x,a: [[x[0], x[1]], [x[0], x[1]]],
#)
#
##1 input, 2 outputs, 2 coefficients
#a2 = Activation(
#    lambda x,a: [a[0]*x[0]+a[1], a[0]*x[0]+a[1]],
#    lambda x,a: [[a[0], a[0]]],
#    lambda x,a: [[x[0], x[0]], [1, 1]],
#)
#
#
##1 input, 1 output, 2 coefficients
#a3 = Activation(
#    lambda x,a: [x[0]*a[0]+a[1]*x[0]],
#    lambda x,a: [[a[0]+a[1]]],
#    lambda x,a: [[x[0]], [x[0]]],
#)

model = Model([
    Layer(1, 'linear_sum'),
    Layer(2, 'linear_sum'),
    Layer(3, 'linear_sum'),
    Layer(1, 'linear_sum')
])

model.compile()

out, state = model.forward([[1]])


print(out)


x = [[[0]], [[1]], [[2]], [[3]], [[10]], [[-1]]]
y_c = [[[0]], [[1]], [[2]], [[3]], [[10]], [[-1]]]
learning_rate = 0.001
epochs = 1000

model.fit(x, y_c, epochs, lambda y,ys: 2*(ys[0][0]-y[0][0]), learning_rate)

out, state = model.forward([[0]])
print(f'(x,y) = ({0}, {out[0][0]})')


out, state = model.forward([[1]])
print(f'(x,y) = ({1}, {out[0][0]})')


out, state = model.forward([[2]])
print(f'(x,y) = ({2}, {out[0][0]})')


out, state = model.forward([[3]])
print(f'(x,y) = ({3}, {out[0][0]})')

out, state = model.forward([[10]])
print(f'(x,y) = ({10}, {out[0][0]})')

out, state = model.forward([[-1]])
print(f'(x,y) = ({-1}, {out[0][0]})')

