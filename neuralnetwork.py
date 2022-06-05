"""
Perceptron - Matemática Computacional

Bruno Porto de Ascenção - 120053812
Lucca Gandra Ventura - 120101720
Pedro Henrique de Jesus Teixeira - 
"""
import dataset_analisys

euler = 2.71828

class Perceptron():
    def __init__(self, weights=[1,1], bias=1, learning_rate=0.04, n_iterations=500):
        """
        Função de atividade 
            Pesos
            Bias
            
            A = w1x1+w2x2+b
        
        Função de Ativação
            Transforma a saida da função de atividade em 0 ou 1
        """
        
        self.weights = weights
        self.bias = bias
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
    
    def train(self , x_set , y_set):
        # Definir pesos Aleatorios
        for epoch in range(self.n_iterations):
            miss = 0
            for j in range(len(x_set)):
                y = self.test(x_set[j])
                err = y_set[j] - y
                # DELTA = n derivada parcial de Err em relacao a w[i]
                # DESCIDA DE GRADIENTE 
                # APRENDIZADO Supervised Learning
                for i in range(len(self.weights)):
                    self.weights[i] = self.weights[i] + self.learning_rate* err *  x_set[j][i]
                    self.bias += self.learning_rate* err
                if abs(err) != 0:
                    miss+=1
            print("Epoch :",epoch,"Errors:", miss)
           
    def test(self, x, type_activation="relu"):

        activity = self.weights[0]*x[0]+self.weights[0]*x[1] + self.bias
        
        if type_activation == "relu":
            activation = max(-1, activity)
        elif type_activation == "special":
            if activity >= 0:
                activation =1
            else:
                activation = -1
        else:
            activation = 1/(1+euler**(-activity))
        
        return 1.0 if activation >= 0.0 else -1

p = Perceptron()
x_set, y_set = dataset_analisys.dataset()
x_train = x_set[0:30]+x_set[50:80]
y_train = y_set[0:30]+y_set[50:80]

# for i in range(100):
x_test = x_set[40]
y_test = y_set[40]

p.train(x_train, y_train)
result = p.test(x_test, "relu")
print(result, y_test)