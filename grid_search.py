from NeuralNetworkNAG_BCE import NeuralNetworkNAG_BCE
from NeuralNetworkNAG_MSE import NeuralNetworkNAG_MSE
from NeuralNetworkBFGS_BCE import NeuralNetworkBFGS_BCE
from NeuralNetworkBFGS_MSE import NeuralNetworkBFGS_MSE
from utils import load_dataBCE, load_dataMSE, plot_losses
from losses import BCE, MSE

X_bce, y_bce = load_dataBCE()
    
X_mse_normalized, y_mse_normalized = load_dataMSE()


for lr in [0.01, 0.1, 0.2]:
    for reg in [0.001, 0.01, 0.1]:
        for n in [2, 4, 8, 10]:
            losses_nag_bce = []
            losses_nag_mse = []
            losses_bfgs_bce = []
            losses_bfgs_mse = []
            model_nag_bce = NeuralNetworkNAG_BCE(
                input_size=6,
                hidden_size=n,
                output_size=1,
                loss=BCE(),
                regularization=reg,
                momentum=0.9,
                learning_rate=lr,
            )

            model_nag_mse = NeuralNetworkNAG_MSE(
                input_size=12,
                hidden_size=n,
                output_size=3,
                loss=MSE(),
                regularization=reg,
                momentum=0.9,
                learning_rate=lr,
            )

            loss_bce, mean_time_bce = model_nag_bce.train(X_bce, y_bce, epochs=100000, batch=True)
            loss_mse, mean_time_mse = model_nag_mse.train(X_mse_normalized, y_mse_normalized, epochs=100000, batch=True)
            plot_losses([loss_bce], [loss_mse], save=True, filename=f"GRID_SEARCH/NAG/grid_search_nag__N{n}_LR{lr}_REG{reg}.png", label=f"Grid Search NAG with N={n}, LR={lr}, REG={reg}", plot_labels=[f"Loss BCE durante il training (final loss={loss_bce[-1]})", f"Loss MSE durante il training (final loss={loss_mse[-1]})"])

            model_bfgs_bce = NeuralNetworkBFGS_BCE(6, n, 1, BCE(), reg)
            model_bfgs_mse = NeuralNetworkBFGS_MSE(12, n, 3, MSE(), reg)

            loss_bce, mean_time_bce = model_bfgs_bce.train(X_bce, y_bce, epochs=200, tol=1e-6, batch=True)
            loss_mse, mean_time_mse = model_bfgs_mse.train(X_mse_normalized, y_mse_normalized, epochs=200, tol=1e-6, batch=True)
            plot_losses([loss_bce], [loss_mse], save=True, filename=f"GRID_SEARCH/BFGS/grid_search_bfgs__N{n}_LR{lr}_REG{reg}.png", label=f"Grid Search BFGS with N={n}, LR={lr}, REG={reg}", plot_labels=[f"Loss BCE durante il training (final loss={loss_bce[-1]})", f"Loss MSE durante il training (final loss={loss_mse[-1]})"])
