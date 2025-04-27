from NeuralNetworkBFGS_BCE import NeuralNetworkBFGS_BCE
from NeuralNetworkBFGS_MSE import NeuralNetworkBFGS_MSE
from utils import load_dataBCE, load_dataMSE, plot_losses
from losses import BCE, MSE
    
if __name__ == '__main__':
    input_size_bce = 6
    hidden_size_bce = 8
    output_size_bce = 1
    input_size_mse = 12
    hidden_size_mse = 8
    output_size_mse = 3
    epochs = 1000
    tolerance = 1e-6
    regularization = 0.001

    X_bce, y_bce = load_dataBCE()
    
    X_mse_normalized, y_mse_normalized = load_dataMSE()

    losses_bce = []
    losses_mse = []

    trials = 1

    for i in range(trials):

        nn_bce = NeuralNetworkBFGS_BCE(input_size_bce, hidden_size_bce, output_size_bce, BCE(), 0)

        loss_bce, mean_time_bce = nn_bce.train(X_bce, y_bce, epochs=epochs, tol=tolerance, batch=True)
        losses_bce.append(loss_bce)

        nn_mse = NeuralNetworkBFGS_MSE(input_size_mse, hidden_size_mse, output_size_mse, MSE(), regularization)

        loss_mse, mean_time_mse = nn_mse.train(X_mse_normalized, y_mse_normalized, epochs=epochs, tol=tolerance, batch=True)
        losses_mse.append(loss_mse)

    plot_losses(losses_bce, losses_mse)