import numpy as np

class NeuralNetworkMSE:
    def __init__(self, input_size, hidden_sizes, output_size, learning_rate, momentum, epochs):
        np.random.seed(45)
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes  # lista con i neuroni dei layer nascosti
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.epochs = epochs

        # Inizializzazione dei pesi e bias
        self.w1 = np.random.randn(hidden_sizes[0], input_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros(hidden_sizes[0])

        self.w2 = np.random.randn(output_size, hidden_sizes[0]) * np.sqrt(2.0 / hidden_sizes[0])
        self.b2 = np.zeros(output_size)

        # Per NAG: inizializza le velocità dei pesi a zero
        self.v_w1 = np.zeros_like(self.w1)
        self.v_b1 = np.zeros_like(self.b1)
        self.v_w2 = np.zeros_like(self.w2)
        self.v_b2 = np.zeros_like(self.b2)


    def train(self, X, y):
        loss_mse = []
        T = len(X)
        t = 1
        prev_loss = None
        patience = 50
        patience_counter = 0
        tolerance = 1e-6
        end = False
        for j in range(self.epochs):
            indices = np.random.permutation(len(X))
            X = X[indices]
            y = y[indices]

            w1_pre, b1_pre, w2_pre, b2_pre = self.anticipate_weights()

            # Forward propagation
            net_hidden = np.dot(w1_pre, X.T).T + b1_pre
            act = self.leacky_relu(net_hidden)
            net_output = np.dot(w2_pre, act.T).T + b2_pre
            output = net_output

            # Compute Loss
            w_total = np.concatenate([
                self.w1.flatten(),
                self.b1.flatten(),
                self.w2.flatten(),
                self.b2.flatten()
            ])
            loss = self.compute_mse(output, y) + 0.1*np.sum(w_total**2)
            loss_mse.append(loss)

            # Controllo divergenza
            if np.isnan(loss) or loss > 1e5:
                print("❌ Loss diverging. Stopping.")
                end = True
                break

            # Controllo convergenza
            if prev_loss is not None:
                if abs(loss - prev_loss) < tolerance:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print("✅ Loss converged. Stopping.")
                        end = True
                        break
                else:
                    patience_counter = 0

            prev_loss = loss

            # Backward propagation
            sigma_output = y - output
            delta_w2 = np.mean(sigma_output[:, :, np.newaxis] * act[:, np.newaxis, :])
            delta_b2 = np.mean(sigma_output)
            #print(delta_w2)
            #print(delta_b2)
        
            sigma_hidden = np.sum(sigma_output[:, :, np.newaxis] * w2_pre[np.newaxis, :, :], axis=1) * self.leacky_relu_derivative(net_hidden)
            delta_w1 = np.mean(X[:, :, np.newaxis] * sigma_hidden[:, np.newaxis, :]).T
            delta_b1 = np.mean(sigma_hidden).reshape(-1)
            #print(delta_w1)
            #print(delta_b1)

            # Update weights and biases
            self.w1 = self.w1 + ((self.learning_rate * delta_w1) + (self.momentum * self.v_w1) - (2*0.1*self.w1))
            self.b1 = self.b1 + ((self.learning_rate * delta_b1) + (self.momentum * self.v_b1) - (2*0.1*self.b1))
            self.w2 = self.w2 + ((self.learning_rate * delta_w2) + (self.momentum * self.v_w2) - (2*0.1*self.w2))
            self.b2 = self.b2 + ((self.learning_rate * delta_b2) + (self.momentum * self.v_b2) - (2*0.1*self.b2))

            # Update velocity
            self.v_w1 = ((self.learning_rate * delta_w1) + (self.momentum * self.v_w1))
            self.v_b1 = ((self.learning_rate * delta_b1) + (self.momentum * self.v_b1))
            self.v_w2 = ((self.learning_rate * delta_w2) + (self.momentum * self.v_w2))
            self.v_b2 = ((self.learning_rate * delta_b2) + (self.momentum * self.v_b2))

            #self.momentum = self.momentum *(1 - (t/T))
            t+=1
            
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

        w2_pre = self.w2 + (self.momentum * self.v_w2)
        b2_pre = self.b2 + (self.momentum * self.v_b2)


        return w1_pre, b1_pre, w2_pre, b2_pre
          

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
        
        