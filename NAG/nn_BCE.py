import numpy as np

class NeuralNetworkBCE:
    def __init__(self, input_size, hidden_sizes, output_size, learning_rate, momentum, epochs):
        np.random.seed(60)
        
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
        for i in range(X.shape[0]):
            w1_pre, b1_pre, w2_pre, b2_pre = self.anticipate_weights()

            # Forward propagation
            act = self.leacky_relu(np.dot(w1_pre, X[i]) + b1_pre)
            output = self.sigmoid(np.dot(w2_pre, act) + b2_pre)
            
            loss = self.compute_bce(output, y[i])
            
        



    def compute_bce(self,output,y):
        outputs_clipped = np.clip(output, 1e-15, 1-1e-15)
        return np.mean(-(1 - y) * np.log(1 - outputs_clipped) - y * np.log(outputs_clipped))

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def leacky_relu(self, z):
        alpha = 0.01
        return np.maximum(alpha * z, z)
    

    # Funzione per anticipare i pesi (pre-update dei pesi)
    def anticipate_weights(self):
        # Calcolare i pesi anticipati con la velocità
        w1_pre = self.w1 + self.momentum * self.v_w1
        b1_pre = self.b1 + self.momentum * self.v_b1

        w2_pre = self.w2 + self.momentum * self.v_w2
        b2_pre = self.b2 + self.momentum * self.v_b2

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