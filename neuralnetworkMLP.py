"""
Perceptron - Matematica Computacional

Bruno Porto de Ascencao - 120053812
Lucca Gandra Ventura - 120101720
Pedro Henrique de Jesus Teixeira - 
"""
import dataset_analisys
import time
import random
import numpy as np
class Perceptron():
    def __init__(self, weights_i=[[random.uniform(-1,1),random.uniform(-1,1),random.uniform(-1,1)],[random.uniform(-1,1),random.uniform(-1,1),random.uniform(-1,1)]],weights_h=[random.uniform(-1,1),random.uniform(-1,1),random.uniform(-1,1)], bias=1, learning_rate=1, epochs=400, hidden_layer_dimensions=3, input_dimension= 2):
        """
        Funcao de atividade 
            Pesos
            Bias
            
            A = w1x1+w2x2+b
        
        Funcao de Ativacao
            Transforma a saida da funcao de atividade em 0 ou 1
        """
        
        self.weights_i = weights_i
        self.weights_h = weights_h
        self.bias = bias
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.hidden_layer_dimensions = hidden_layer_dimensions
        self.inputs_dimension = input_dimension
        self.type_activation = "relu"
        

    def activation_f(self,activity_h):
        # if self.type_activation == "relu":
        #     if activity_h < 0:
        #         activation = 0
        #     else :
        #         activation = activity_h            

        return 1.0/(1 + np.exp(-activity_h))

    def activation_d(self,activity):
        # if self.type_activation == "relu":
        #     if activity < 0:
        #         activation = 0
        #     else :
        #         activation = 1
        
        return self.activation_f(activity) * (1 - self.activation_f(activity))

    def train(self , x_set , y_set):
        for epoch in range(self.epochs):
            miss = 0
            for j in range(len(x_set)):
                activity, y, activity_h, activation_h  = self.test(x_set[j])
                #print(y_set[j],y )
                err = y_set[j] - y
                # DESCIDA DE GRADIENTE 

                # APRENDIZADO Supervised Learning Backpropagation
                for neuron in range(self.hidden_layer_dimensions):
                    S_error = err * self.activation_d(activity)
                    gradient_HtoO = S_error * activation_h[neuron]
                            
                    for i in range(self.inputs_dimension):
                        input_value = x_set[j][i]
                        gradient_ItoH = S_error * self.weights_h[neuron] * self.activation_d(activity_h[neuron]) * input_value
                        
                        self.weights_i[i][neuron] -= self.learning_rate * gradient_ItoH
                        
                    self.weights_h[neuron] -= self.learning_rate * gradient_HtoO
                
                
                if y > 0.5:
                    y =1
                else: 
                    y=0
                if y != y_set[j]:
                    miss += 1 

            print("Epoch :",epoch,"Errors:", miss)
           
    def test(self, x): 
        activity_h = [0,0,0]
        activation_h = [0,0,0]
        
        for neuron in range(self.hidden_layer_dimensions):
            for i in range(self.inputs_dimension):
                activity_h[neuron] += self.weights_i[i][neuron]*x[i]
            activity_h[neuron] += self.bias
            activation_h[neuron] = self.activation_f(activity_h[neuron]) 
        # print(1,activity_h) 
        # print(2,activation_h)
        activity =0
        for neuron in range(self.hidden_layer_dimensions):
            activity += self.weights_h[neuron]*activation_h[neuron]
        activity += self.bias
        activation = self.activation_f(activity)
        # print(3,activity) 
        # print(4,activation)
        #print(activity, activation, activity_h, activation_h)
        # time.sleep(0.2)
        return  activity, activation, activity_h, activation_h

p = Perceptron()
x_set, y_set = dataset_analisys.dataset()
x_train = x_set[0:30]+x_set[50:80]
y_train = y_set[0:30]+y_set[50:80]

# for i in range(100):
x_test = x_set[34]
y_test = y_set[34]

p.train(x_train, y_train)
result = p.test(x_test)