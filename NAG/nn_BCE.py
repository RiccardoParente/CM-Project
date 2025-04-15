import numpy as np

class NeuralNetworkBCE:
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

        self.w2 = np.random.randn(output_size, hidden_sizes[0]) * np.sqrt(2.0 / hidden_sizes[0])
        self.b2 = np.zeros(output_size)

        # Per NAG: inizializza le velocità dei pesi a zero
        self.v_w1 = np.zeros_like(self.w1)
        self.v_b1 = np.zeros_like(self.b1)
        self.v_w2 = np.zeros_like(self.w2)
        self.v_b2 = np.zeros_like(self.b2)
    
    def train(self, X, y):
        loss_bce = []
        T = len(X )* self.epochs
        t = 1
        prev_loss = None
        patience = 10
        patience_counter = 0
        tolerance = 1e-2
        end = False
        for j in range(self.epochs):
            indices = np.random.permutation(len(X))
            X = X[indices]
            y = y[indices]
            for i in range(X.shape[0]):
                w1_pre, b1_pre, w2_pre, b2_pre = self.anticipate_weights()

                # Forward propagation
                net_hidden = np.dot(w1_pre, X[i]) + b1_pre
                act = self.leacky_relu(net_hidden)
                net_output = np.dot(w2_pre, act) + b2_pre
                output = self.sigmoid(net_output)

                # Compute Loss
                loss = self.compute_bce(output, y[i])
                loss_bce.append(loss)

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
                sigma_output = y[i] - output
                delta_w2 = (sigma_output * act).reshape(1,self.hidden_sizes[0])
                delta_b2 = sigma_output

                sigma_hidden = (sigma_output * w2_pre) * self.leacky_relu_derivative(net_hidden)
                delta_w1 = sigma_hidden.T * X[i]
                delta_b1 = sigma_hidden.reshape(-1)

                # Update weights and biases
                self.w1 = self.w1 + ((self.learning_rate * delta_w1) + (self.momentum * self.v_w1))
                self.b1 = self.b1 + ((self.learning_rate * delta_b1) + (self.momentum * self.v_b1))
                self.w2 = self.w2 + ((self.learning_rate * delta_w2) + (self.momentum * self.v_w2))
                self.b2 = self.b2 + ((self.learning_rate * delta_b2) + (self.momentum * self.v_b2))

                # Update velocity
                self.v_w1 = ((self.learning_rate * delta_w1) + (self.momentum * self.v_w1))
                self.v_b1 = ((self.learning_rate * delta_b1) + (self.momentum * self.v_b1))
                self.v_w2 = ((self.learning_rate * delta_w2) + (self.momentum * self.v_w2))
                self.v_b2 = ((self.learning_rate * delta_b2) + (self.momentum * self.v_b2))

                self.momentum = self.momentum *(1 - (t/T))
                t+=1

            if end:
                break

        return loss_bce


    def compute_bce(self,output,y):
        outputs_clipped = np.clip(output, 1e-15, 1-1e-15)
        return np.mean(-(1 - y) * np.log(1 - outputs_clipped) - y * np.log(outputs_clipped))
    
    def compute_bce_derivate(self,output,y):
        outputs_clipped = np.clip(output, 1e-15, 1-1e-15)
        return -np.mean(y/outputs_clipped - (1-y)/(1-outputs_clipped))

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def sigmoid_derivate(selfself, z):
        return z * (1 - z) 
    
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
        
        print(f"\nLayer 2 (hidden1 -> output):")
        print(f"Pesos (w2): \n{self.w2}")
        print(f"Bias (b2): \n{self.b2}")