import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

def plot_losses(losses_bce, losses_mse):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    for l in losses_bce:
        plt.plot(l, alpha = 0.5 if len(losses_bce) != 1 else 1)

    if len(losses_bce) != 1:
        plt.plot(np.mean(np.array(losses_bce), axis=0), alpha=1, linewidth=2, color='black', linestyle='--', label='Media')
    plt.title('Loss BCE durante il training')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    for l in losses_mse:
        plt.plot(l, alpha = 0.5 if len(losses_mse) != 1 else 1)

    if len(losses_mse) != 1:
        plt.plot(np.mean(np.array(losses_mse), axis=0), alpha=1, linewidth=2, color='black', linestyle='--', label='Media')
    plt.title('Loss MSE durante il training')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def load_dataBCE():

    data_bce = np.loadtxt("datasets/MONK/monks-3.train", delimiter=" ", dtype=str)
    data_bce = data_bce[:, 1:-1]
    data_bce = data_bce.astype(int)

    X_bce = data_bce[:, 1:]
    y_bce = data_bce[:, 0].reshape(-1, 1)

    return X_bce, y_bce

def load_dataMSE():
    data_mse = np.loadtxt("datasets/CUP/ML-CUP24-TR.csv", delimiter=",")

    X_mse = data_mse[:, 1:-3]
    scaler = StandardScaler()
    X_mse_normalized = scaler.fit_transform(X_mse)
    y_mse = data_mse[:, -3:]
    y_mse_normalized = scaler.fit_transform(y_mse)
    return X_mse_normalized, y_mse_normalized
