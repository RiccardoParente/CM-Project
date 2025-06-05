from NeuralNetworkBFGS_BCE import NeuralNetworkBFGS_BCE
from NeuralNetworkBFGS_MSE import NeuralNetworkBFGS_MSE
from utils import load_dataBCE, load_dataMSE, plot_losses, plot_gradients
from losses import BCE, MSE

# hyperparameters  
input_size_bce = 6
hidden_size_bce = 10
output_size_bce = 1
input_size_mse = 12
hidden_size_mse = 10
output_size_mse = 3
epochs = 1000
tolerance = 1e-4
regularization = 0.05

# --- Loading dataset from CSV --- #
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

# --- Instantiating and training models --- #
for i in range(trials):

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

print("mean times: ", mean_time_bce/trials, mean_time_mse/trials)
# plot loss and gradient norm
plot_losses(losses_bce, losses_mse)
plot_gradients(gradients_bce, gradients_mse)
relative_bce = []
relative_mse = []
for t in range(trials):
    # relative gap calculation
    relative_bce.append([(x - 0.6932) / 0.6932 for x in losses_bce[t]])
    relative_mse.append([(x - 1.0000) / 1.0000 for x in losses_mse[t]])

# plot relative gap
plot_losses(relative_bce, relative_mse, label="Relative gap BFGS", plot_labels=["Relative gap BCE", "Relative gap MSE"])