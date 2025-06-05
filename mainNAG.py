from NeuralNetworkNAG_BCE import NeuralNetworkNAG_BCE
from NeuralNetworkNAG_MSE import NeuralNetworkNAG_MSE
from utils import load_dataBCE, load_dataMSE, plot_losses, plot_gradients
from losses import BCE, MSE

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

for i in range(trials):
    # --- Instantiating models --- #
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


print("mean times: ", mean_time_bce/trials, mean_time_mse/trials)
# plot loss and gradient norm
plot_losses(losses_bce, losses_mse)
plot_gradients(gradients_bce, gradients_mse)
relative_bce = []
relative_mse = []
for t in range(trials):
    # relative gap calculation
    relative_bce.append([(x - 0.06) / 0.06 for x in losses_bce[t]])
    relative_mse.append([(x - 0.03) / 0.03 for x in losses_mse[t]])

# plot relative gap
plot_losses(relative_bce, relative_mse, label="Relative gap NAG", plot_labels=["Relative gap BCE", "Relative gap MSE"])