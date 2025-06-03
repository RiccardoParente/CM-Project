from NeuralNetworkBFGS_BCE import NeuralNetworkBFGS_BCE
from NeuralNetworkBFGS_MSE import NeuralNetworkBFGS_MSE
from NeuralNetworkNAG_BCE import NeuralNetworkNAG_BCE
from NeuralNetworkNAG_MSE import NeuralNetworkNAG_MSE
from utils import load_dataBCE, load_dataMSE, plot_losses, plot_gradients
from losses import BCE, MSE
import numpy as np
    
input_size_bce = 6
hidden_size_bce = 10
output_size_bce = 1
input_size_mse = 12
hidden_size_mse = 10
output_size_mse = 3
epochs = 500
tolerance = 1e-3
regularization = 0.05

X_bce, y_bce = load_dataBCE()

X_mse_normalized, y_mse_normalized, x_mse, y_mse = load_dataMSE()

losses_bce = []
losses_mse = []
mean_time_bce = 0
mean_time_mse = 0
minimums_bce = []
min_bce = float('inf')
minimums_mse = []
min_mse = float('inf')

trials = 1

for i in range(trials):

    np.random.seed(109)

    model_bce = NeuralNetworkNAG_BCE(
        input_size=6,
        hidden_size=200,
        output_size=1,
        loss=BCE(),
        layers=10,
        regularization=0,
        momentum=0.9,
        learning_rate=0.1,
    )

    np.random.seed(109)

    model_mse = NeuralNetworkNAG_MSE(
        input_size=12,
        hidden_size=200,
        output_size=3,
        layers=10,
        loss=MSE(),
        regularization=0,
        momentum=0.9,
        learning_rate=0.01,
    )

    # --- Training --- #
    loss_bce, mt, gradients_bce = model_bce.train(X_bce, y_bce, epochs=1000, batch=True)
    losses_bce.append(loss_bce)
    if loss_bce[-1] < min_bce:
        min_bce = loss_bce[-1]
    minimums_bce.append(loss_bce[-1])
    mean_time_bce += mt
    print(loss_bce[-1])
    loss_mse, mt, gradients_mse = model_mse.train(X_mse_normalized, y_mse_normalized, epochs=1000, batch=True)
    losses_mse.append(loss_mse)
    if loss_mse[-1] < min_mse:
        min_mse = loss_mse[-1]
    minimums_mse.append(loss_mse[-1])
    mean_time_mse += mt
    print(loss_mse[-1])

    nn_bce = NeuralNetworkBFGS_BCE(input_size_bce, hidden_size_bce, output_size_bce, BCE(), layers=1, regularization=regularization)

    loss_bce, mt, gradients_bce = nn_bce.train(X_bce, y_bce, epochs=epochs, tol=tolerance, batch=True)
    losses_bce.append(loss_bce)
    if loss_bce[-1] < min_bce:
        min_bce = loss_bce[-1]
    minimums_bce.append(loss_bce[-1])
    mean_time_bce += mt

    nn_mse = NeuralNetworkBFGS_MSE(input_size_mse, hidden_size_mse, output_size_mse, MSE(), layers=1, regularization=regularization)

    loss_mse, mt, gradients_mse, diverged = nn_mse.train(X_mse_normalized, y_mse_normalized, epochs=epochs, tol=tolerance, batch=True)
    losses_mse.append(loss_mse)
    if not diverged and loss_mse[-1] < min_mse:
        min_mse = loss_mse[-1]
    minimums_mse.append(loss_mse[-1])
    mean_time_mse += mt

minimums_bce = np.array([np.min([0.0507,(minimums_bce[0]-0.001)]), np.min([0.6932,(minimums_bce[1]-0.001)])])
minimums_mse = np.array([np.min([0.0306,(minimums_mse[0]-0.001)]), np.min([1.0,(minimums_mse[1]-0.001)])])

print("mean times: ", mean_time_bce/trials, mean_time_mse/trials)
plot_losses(losses_bce, losses_mse)
plot_gradients(gradients_bce, gradients_mse)
convergences_bce = []
convergences_mse = []
relative_bce = []
relative_mse = []
for t in range(2):
    convergence_bce = []
    convergence_mse = []
    for i in range(len(losses_bce[t])-1):
        convergence_bce.append(losses_bce[t][i+1]/losses_bce[t][i])
    for i in range(len(losses_mse[t])-1):
        convergence_mse.append(losses_mse[t][i+1]/losses_mse[t][i])
    convergences_bce.append(convergence_bce)
    convergences_mse.append(convergence_mse)
    relative_bce.append((losses_bce[t]-minimums_bce[t])/minimums_bce[t])
    relative_mse.append((losses_mse[t]-minimums_mse[t])/minimums_mse[t])
plot_losses(convergences_bce, convergences_mse, label="Convergence BFGS", plot_labels=["Convergence BCE", "Convergence MSE"])
plot_losses(relative_bce, relative_mse, label="", labels=["NAG","BFGS-N"], colors=["red", "blue"], plot_labels=["Relative gap BCE", "Relative gap MSE"])