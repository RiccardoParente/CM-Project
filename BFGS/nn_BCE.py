import numpy as np
from NeuralNetworkBFGS import NeuralNetworkBFGS

class BCE:
    """ Binary Cross Entropy loss """

    def __init__(self):
        self.name = "Binary Cross Entropy"

    def compute(self, outputs, targets):
        outputs_clipped = np.clip(outputs, 1e-15, 1-1e-15)
        return np.mean(-(1 - targets) * np.log(1 - outputs_clipped) - targets * np.log(outputs_clipped))

    def derivative(self, outputs, targets):
        outputs_clipped = np.clip(outputs, 1e-15, 1-1e-15)
        return -np.mean(targets/outputs_clipped - (1-targets)/(1-outputs_clipped))


if __name__ == '__main__':
    input_size = 6
    hidden_size = 8
    output_size = 1
    learning_rate = 1
    max_iterations = 2000
    tolerance = 1e-5

    # Create a simple dataset (XOR problem)
    data_bce = np.loadtxt("../datasets/MONK/monks-3.train", delimiter=" ", dtype=str)
    data_bce = data_bce[:, 1:-1]
    data_bce = data_bce.astype(int)

    X_bce = data_bce[:, 1:]
    y_bce = data_bce[:, 0].reshape(-1, 1) 

    # Initialize the neural network with BFGS-N
    nn = NeuralNetworkBFGS(input_size, hidden_size, output_size, BCE())

    history = nn.train(X_bce, y_bce, max_iter=max_iterations, tol=tolerance)

    # Test the trained network
    print("\nPredictions after training:")
    predictions = nn.forward(X_bce)
    #print(predictions)

    import matplotlib.pyplot as plt
    plt.plot(history)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training Loss over Iterations (BFGS-N from Scratch)")
    plt.show()