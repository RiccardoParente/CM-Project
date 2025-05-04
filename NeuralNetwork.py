import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, loss, regularization=0.01, momentum=0.9, learning_rate=0.01):
        np.random.seed(0)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.loss = loss
        self.regularization = regularization
        self.momentum = momentum
        self.learning_rate = learning_rate

        # Initialize weights and biases randomly
        self.wh = np.random.randn(self.input_size, self.hidden_size) * np.sqrt(2.0 / input_size)
        self.bh = np.zeros(self.hidden_size)
        self.wo = np.random.randn(self.hidden_size, self.output_size) * np.sqrt(2.0 / hidden_size)
        self.bo = np.zeros(self.output_size)

    def flatten_params(self):
        """Flatten the weights and biases into a single vector."""
        output = np.array([])
        for i in range(self.hidden_size):
            output = np.concatenate((output, self.wh[:,i], [self.bh[i]]))
        for i in range(self.output_size):
            output = np.concatenate((output, self.wo[:,i], [self.bo[i]]))
        return output

    def unflatten_params(self, params):
        """Restore the weights and biases from a flattened vector."""
        wh_temp = []
        bh_temp = []
        wo_temp = []
        bo_temp = []
        for i in range(self.hidden_size):
            ptr = self.input_size*i
            wh_temp = np.concatenate((wh_temp, params[ptr:ptr+self.input_size]))
            bh_temp = np.concatenate((bh_temp, [params[ptr+self.input_size]]))
        self.wh = np.array(wh_temp).reshape(self.input_size, self.hidden_size)
        self.bh = np.array(bh_temp)
        offset = (self.input_size*self.hidden_size)+self.hidden_size
        for i in range(self.output_size):
            ptr = offset+(self.hidden_size*i)
            wo_temp = np.concatenate((wo_temp, params[ptr:ptr+self.hidden_size]))
            bo_temp = np.concatenate((bo_temp, [params[ptr+self.hidden_size]]))
        self.wo = np.array(wo_temp).reshape(self.hidden_size, self.output_size)
        self.bo = np.array(bo_temp)

    def leacky_relu(self, z):
        alpha = 0.01
        return np.maximum(alpha * z, z)
    
    def leacky_relu_derivative(self, a):
        alpha = 0.01
        return np.where(a > 0, 1, alpha)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)

    def print_structure(self):
        '''Print the structure of the neural network.'''
        
        print(f"Struttura della rete neurale:")
        print(f"Input size: {self.input_size}")
        print(f"Hidden layers sizes: {self.hidden_size}")
        print(f"Output size: {self.output_size}")
        
        print("\nPesi e bias:")
        print(f"Layer 1 (input -> hidden):")
        print(f"Weights (w1): \n{self.wh}")
        print(f"Bias (b1): \n{self.bh}")
        
        print(f"\nLayer 2 (hidden -> output):")
        print(f"Weights (w2): \n{self.wo}")
        print(f"Bias (b2): \n{self.bo}")