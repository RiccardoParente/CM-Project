from NeuralNetworkBFGS_BCE import NeuralNetworkBFGS_BCE
from NeuralNetworkBFGS_MSE import NeuralNetworkBFGS_MSE
from utils import load_dataBCE, load_dataMSE, plot_losses, plot_gradients
from losses import BCE, MSE
    
if __name__ == '__main__':
    input_size_bce = 6
    hidden_size_bce = 8
    output_size_bce = 1
    input_size_mse = 12
    hidden_size_mse = 8
    output_size_mse = 3
    epochs = 10000
    tolerance = 1e-4
    regularization = 0.05

    X_bce, y_bce = load_dataBCE()
    
    X_mse_normalized, y_mse_normalized, x_mse, y_mse = load_dataMSE()

    losses_bce = []
    losses_mse = []
    mean_time_bce = 0
    mean_time_mse = 0

    trials = 1

    for i in range(trials):

        nn_bce = NeuralNetworkBFGS_BCE(input_size_bce, hidden_size_bce, output_size_bce, BCE(), regularization)

        loss_bce, mt, gradients_bce = nn_bce.train(X_bce, y_bce, epochs=epochs, tol=tolerance, batch=True)
        losses_bce.append(loss_bce)
        mean_time_bce += mt

        nn_mse = NeuralNetworkBFGS_MSE(input_size_mse, hidden_size_mse, output_size_mse, MSE(), regularization)

        loss_mse, mt, gradients_mse = nn_mse.train(X_mse_normalized, y_mse_normalized, epochs=epochs, tol=tolerance, batch=True)
        losses_mse.append(loss_mse)
        mean_time_mse += mt

    print("mean times: ", mean_time_bce/trials, mean_time_mse/trials)
    plot_losses(losses_bce, losses_mse)
    plot_gradients(gradients_bce, gradients_mse)
    convergence_bce = []
    convergence_mse = []
    for i in range(len(losses_bce[0])-1):
        convergence_bce.append(losses_bce[0][i+1]/losses_bce[0][i])
    for i in range(len(losses_mse[0])-1):
        convergence_mse.append(losses_mse[0][i+1]/losses_mse[0][i])
    plot_losses([convergence_bce], [convergence_mse], label="Convergence BFGS", plot_labels=["Convergence BCE", "Convergence MSE"])
