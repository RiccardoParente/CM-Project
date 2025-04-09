import numpy as np

class NeuralNetworkMSE:
    def __init__(self, input_size, hidden_sizes, output_size, learning_rate, momentum, epochs):
        np.random.seed(200)
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes  # lista con i neuroni dei layer nascosti
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.epochs = epochs

        # Inizializzazione dei pesi e bias
        self.w1 = np.random.randn(hidden_sizes[0], input_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros(hidden_sizes[0])

        #self.w2 = np.random.randn(hidden_sizes[1], hidden_sizes[0]) * np.sqrt(2.0 / hidden_sizes[0])
        #self.b2 = np.zeros(hidden_sizes[1])

        self.w3 = np.random.randn(output_size, hidden_sizes[0]) * np.sqrt(2.0 / hidden_sizes[0])
        self.b3 = np.zeros(output_size)

        # Per NAG: inizializza le velocità dei pesi a zero
        self.v_w1 = np.zeros_like(self.w1)
        self.v_b1 = np.zeros_like(self.b1)
        #self.v_w2 = np.zeros_like(self.w2)
        #self.v_b2 = np.zeros_like(self.b2)
        self.v_w3 = np.zeros_like(self.w3)
        self.v_b3 = np.zeros_like(self.b3)


    def train(self, X, y):
        loss_mse = []
        count = 0
        for i in range(X.shape[0]):
            w1_pre, b1_pre, w3_pre, b3_pre = self.anticipate_weights()

            # Forward propagation
            net_hidden1 = np.dot(w1_pre, X[i]) + b1_pre
            act1 = self.leacky_relu(net_hidden1)
            #net_hidden2 = np.dot(w2_pre, act1) + b2_pre
            #act2 = self.leacky_relu(net_hidden2)
            net_output = np.dot(w3_pre, act1) + b3_pre
            output = net_output

            # Compute Loss
            loss = self.compute_mse(output, y[i])
            print(loss)
            loss_mse.append(loss)

            # Backward propagation
            sigma_output = y[i] - output 
            delta_w3 = sigma_output[:, np.newaxis] * act1
            delta_b3 = sigma_output
            #print(delta_w3)
            #print(delta_b3)

            #sigma_hidden2 = np.sum(sigma_output[:, np.newaxis] * w3_pre, axis=0) * self.leacky_relu_derivative(net_hidden2)
            #delta_w2 = sigma_hidden2[:, np.newaxis] * act1
            #delta_b2 = sigma_hidden2
            #print(delta_w2)
            #print(delta_b2)
        
            sigma_hidden1 = np.sum(sigma_output[:, np.newaxis] * w3_pre, axis=0) * self.leacky_relu_derivative(net_hidden1)
            delta_w1 = sigma_hidden1[:, np.newaxis] * X[i]
            delta_b1 = sigma_hidden1
            #print(delta_w1)
            #print(delta_b1)

            # Update weights and biases
            self.w1 = self.w1 + ((self.learning_rate * delta_w1) + (self.momentum * self.v_w1))
            self.b1 = self.b1 + ((self.learning_rate * delta_b1) + (self.momentum * self.v_b1))
            #self.w2 = self.w2 + ((self.learning_rate * delta_w2) + (self.momentum * self.v_w2))
            #self.b2 = self.b2 + ((self.learning_rate * delta_b2) + (self.momentum * self.v_b2))
            self.w3 = self.w3 + ((self.learning_rate * delta_w3) + (self.momentum * self.v_w3))
            self.b3 = self.b3 + ((self.learning_rate * delta_b3) + (self.momentum * self.v_b3))

            # Update velocity
            self.v_w1 = ((self.learning_rate * delta_w1) + (self.momentum * self.v_w1))
            self.v_b1 = ((self.learning_rate * delta_b1) + (self.momentum * self.v_b1))
            #self.v_w2 = ((self.learning_rate * delta_w2) + (self.momentum * self.v_w2))
            #self.v_b2 = ((self.learning_rate * delta_b2) + (self.momentum * self.v_b2))
            self.v_w3 = ((self.learning_rate * delta_w3) + (self.momentum * self.v_w3))
            self.v_b3 = ((self.learning_rate * delta_b3) + (self.momentum * self.v_b3))

            #if count == 58:
            #    return
            #count+=1
        return loss_mse

            
    def compute_mse(self, outputs, targets):
        return np.mean(np.power(targets - outputs, 2))
    
    def leacky_relu(self, z):
        alpha = 0.01
        return np.maximum(alpha * z, z)
    
    def leacky_relu_derivative(self, a):
        alpha = 0.01
        return np.where(a > 0, 1, alpha)
    
    # Funzione per anticipare i pesi (pre-update dei pesi)
    def anticipate_weights(self):
        # Calcolare i pesi anticipati con la velocità
        w1_pre = self.w1 + (self.momentum * self.v_w1)
        b1_pre = self.b1 + (self.momentum * self.v_b1)

        #w2_pre = self.w2 + (self.momentum * self.v_w2)
        #b2_pre = self.b2 + (self.momentum * self.v_b2)

        w3_pre = self.w3 + (self.momentum * self.v_w3)
        b3_pre = self.b3 + (self.momentum * self.v_b3)

        return w1_pre, b1_pre, w3_pre, b3_pre
          
    # Metodo per stampare la struttura della rete
    def print_structure(self):
        print(f"Struttura della rete neurale:")
        print(f"Input size: {self.input_size}")
        print(f"Hidden layers sizes: {self.hidden_sizes}")
        print(f"Output size: {self.output_size}")
        
        # Pesi e bias di ogni layer
        print("\nPesi e bias:")
        print(f"Layer 1 (input -> hidden1):")
        print(f"Pesos (w1): \n{self.w1}")
        print(f"Bias (b1): \n{self.b1}")
        
        #print(f"\nLayer 2 (hidden1 -> hidden2):")
        #print(f"Pesos (w2): \n{self.w2}")
        #print(f"Bias (b2): \n{self.b2}")
        
        print(f"\nLayer 3 (hidden2 -> output):")
        print(f"Pesos (w3): \n{self.w3}")
        print(f"Bias (b3): \n{self.b3}")
        