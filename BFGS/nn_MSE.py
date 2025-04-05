import numpy as np
from losses import MSE
import NeuralNetworkBFGS

if __name__ == '__main__':
    # Example usage
    input_size = 2
    hidden_size = 3
    output_size = 1
    learning_rate = 1
    max_iterations = 200
    tolerance = 1e-5

    # Create a simple dataset (XOR problem)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    # Initialize the neural network with BFGS-N
    nn = NeuralNetworkBFGS(input_size, hidden_size, output_size, MSE())

    history = nn.train(X, y, learning_rate=learning_rate, max_iter=max_iterations, tol=tolerance)

    # Test the trained network
    print("\nPredictions after training:")
    predictions = nn.forward(X)
    print(predictions)

    import matplotlib.pyplot as plt
    plt.plot(history)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training Loss over Iterations (BFGS-N from Scratch)")
    plt.show()