import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

def plot_losses(losses_bce, losses_mse, labels=None, colors=None, save=False, filename=None, label='', plot_labels=['Loss BCE durante il training','Loss MSE durante il training']):
    '''function to plot the losses or save the graphs to file'''
    plt.figure(figsize=(12, 5))
    plt.suptitle(label)
    plt.subplot(1, 2, 1)
    for i in range(len(losses_bce)):
        plt.plot(losses_bce[i], alpha = 1, label=labels[i] if labels is not None else "", color=colors[i] if colors is not None else "blue")

    plt.title(plot_labels[0])
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    for i in range(len(losses_mse)):
        plt.plot(losses_mse[i], alpha = 1, label=labels[i] if labels is not None else "", color=colors[i] if colors is not None else "blue")

    plt.title(plot_labels[1])
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    if save and filename is not None:
        plt.savefig(filename)
        plt.clf()
    else:
        plt.show()

def plot_gradients(grad_bce, grad_mse):
    '''function to plot the gradients norm'''
    plt.figure(figsize=(12, 5))
    plt.suptitle('Gradients Norm')

    plt.subplot(1, 2, 1)
    plt.plot(grad_bce, color='green')

    plt.title('Gradients Norm BCE')
    plt.xlabel('Epochs')
    plt.ylabel('Gradient Norm')
    plt.yscale('log')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(grad_mse, color='green')

    plt.title('Gradients Norm MSE')
    plt.xlabel('Epochs')
    plt.ylabel('Gradient Norm')
    plt.yscale('log')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def load_dataBCE():
    '''function to load the MONK dataset'''
    data_bce = np.loadtxt("datasets/MONK/monks-3.train", delimiter=" ", dtype=str)
    data_bce = data_bce[:, 1:-1]
    data_bce = data_bce.astype(int)

    X_bce = data_bce[:, 1:]
    y_bce = data_bce[:, 0].reshape(-1, 1)

    return X_bce, y_bce

def load_dataMSE():
    '''function to load the CUP dataset'''
    data_mse = np.loadtxt("datasets/CUP/ML-CUP24-TR.csv", delimiter=",")

    X_mse = data_mse[:, 1:-3]
    scaler = StandardScaler()
    X_mse_normalized = scaler.fit_transform(X_mse)
    y_mse = data_mse[:, -3:]
    y_mse_normalized = scaler.fit_transform(y_mse)
    return X_mse_normalized, y_mse_normalized, X_mse, y_mse
