from NeuralNetworkNAG_BCE import NeuralNetworkNAG_BCE
from NeuralNetworkNAG_MSE import NeuralNetworkNAG_MSE
from utils import load_dataBCE, load_dataMSE, plot_losses, plot_gradients
from losses import BCE, MSE

# --- Caricamento dataset da CSV --- #
X_bce, y_bce = load_dataBCE()
    
X_mse_normalized, y_mse_normalized, x_mse, y_mse = load_dataMSE()

losses_bce = []
losses_mse = []
mean_time_bce = 0
mean_time_mse = 0

trials = 1

for i in range(trials):
    # --- Istanziamento modelli --- #
    model_bce = NeuralNetworkNAG_BCE(
        input_size=6,
        hidden_size=8,
        output_size=1,
        loss=BCE(),
        regularization=0,
        momentum=0.9,
        learning_rate=0.1,
    )

    model_mse = NeuralNetworkNAG_MSE(
        input_size=12,
        hidden_size=8,
        output_size=3,
        loss=MSE(),
        regularization=0,
        momentum=0.9,
        learning_rate=0.01,
    )

    # --- Training --- #
    loss_bce, mt, gradients_bce = model_bce.train(X_bce, y_bce, epochs=10000, batch=True)
    losses_bce.append(loss_bce)
    mean_time_bce += mt
    print(loss_bce[-1])
    loss_mse, mt, gradients_mse = model_mse.train(X_mse_normalized, y_mse_normalized, epochs=10000, batch=True)
    losses_mse.append(loss_mse)
    mean_time_mse += mt
    print(loss_mse[-1])

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