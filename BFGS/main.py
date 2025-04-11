import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
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
    
class MSE:
    """ Mean Squared Error loss """

    def __init__(self):
        self.name = "Mean Squared Error"

    def compute(self, outputs, targets):
        return np.mean(np.power(targets - outputs, 2))

    def derivative(self, outputs, targets):
        return -2 * (targets - outputs)


if __name__ == '__main__':
    input_size_bce = 6
    hidden_size_bce = 8
    output_size_bce = 1
    input_size_mse = 12
    hidden_size_mse = 8
    output_size_mse = 3
    max_iterations = 1
    tolerance = 1e-5

    data_bce = np.loadtxt("../datasets/MONK/monks-3.train", delimiter=" ", dtype=str)
    data_bce = data_bce[:, 1:-1]
    data_bce = data_bce.astype(int)

    X_bce = data_bce[:, 1:]
    y_bce = data_bce[:, 0].reshape(-1, 1) 

    data_mse = np.loadtxt("../datasets/CUP/ML-CUP24-TR.csv", delimiter=",")

    X_mse = data_mse[:, 1:-3]
    scaler = StandardScaler()
    X_mse_normalized = scaler.fit_transform(X_mse)
    y_mse = data_mse[:, -3:]
    y_mse_normalized = scaler.fit_transform(y_mse)

    nn_bce = NeuralNetworkBFGS(input_size_bce, hidden_size_bce, output_size_bce, BCE())

    loss_bce = nn_bce.train(X_bce, y_bce, max_iter=max_iterations, tol=tolerance)

    nn_mse = NeuralNetworkBFGS(input_size_mse, hidden_size_mse, output_size_mse, MSE())

    loss_mse = nn_mse.train(X_mse_normalized, y_mse_normalized, max_iter=max_iterations, tol=tolerance)

    data_bce = np.loadtxt("../datasets/MONK/monks-3.test", delimiter=" ", dtype=str)
    data_bce = data_bce[:, 1:-1]
    data_bce = data_bce.astype(int)

    X_bce = data_bce[:, 1:]
    y_bce = data_bce[:, 0].reshape(-1, 1) 
    print("\nMSE:")
    predictions = nn_bce.forward(X_bce)
    print(np.mean(np.power(y_bce - predictions, 2)))

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(loss_bce, label='Loss BCE')
    plt.title('Loss BCE durante il training')
    plt.xlabel('Campioni')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(loss_mse, label='Loss MSE', color='orange')
    plt.title('Loss MSE durante il training')
    plt.xlabel('Campioni')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()