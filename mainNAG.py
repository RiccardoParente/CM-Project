import numpy as np
import matplotlib.pyplot as plt
from NeuralNetworkNAG_BCE import NeuralNetworkNAG_BCE
from NeuralNetworkNAG_MSE import NeuralNetworkNAG_MSE
from utils import load_dataBCE, load_dataMSE, plot_losses
from losses import BCE, MSE

# --- Caricamento dataset da CSV --- #
X_bce, y_bce = load_dataBCE()
    
X_mse_normalized, y_mse_normalized = load_dataMSE()

losses_bce = []
losses_mse = []

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
        regularization=0.1,
        momentum=0.9,
        learning_rate=0.01,
    )

    # --- Training --- #
    loss_bce = model_bce.train(X_bce, y_bce, epochs=10000)
    losses_bce.append(loss_bce)
    print(loss_bce[-1])
    loss_mse = model_mse.train(X_mse_normalized, y_mse_normalized, epochs=10000)
    losses_mse.append(loss_mse)
    print(loss_mse[-1])

plot_losses(losses_bce, losses_mse)