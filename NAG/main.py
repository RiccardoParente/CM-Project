import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from nn_BCE import NeuralNetworkBCE
from nn_MSE import NeuralNetworkMSE
from Draft_nn_BCE_BFGS import NeuralNetworkBCEBFGS

# --- Caricamento dataset da CSV --- #
# Dataset BCE: 6 input, 1 target
data_bce = np.loadtxt("datasets/MONK/monks-3.train", delimiter=" ", dtype=str)
data_bce = data_bce[:, 1:-1]
data_bce = data_bce.astype(int)

scaler = StandardScaler()

X_bce = data_bce[:, 1:]
X_bce_normalized = scaler.fit_transform(X_bce)
y_bce = data_bce[:, 0].reshape(-1, 1) 


# Dataset MSE: 12 input, 3 target
data_mse = np.loadtxt("datasets/CUP/ML-CUP24-TR.csv", delimiter=",")

X_mse = data_mse[:, 1:-3]
X_mse_normalized = scaler.fit_transform(X_mse)
y_mse = data_mse[:, -3:]
y_mse_normalized = scaler.fit_transform(y_mse)
'''
# --- Istanziamento modelli --- #
model_bce = NeuralNetworkBCEBFGS(
    input_size=6,
    hidden_sizes=[8],
    output_size=1,
    epochs=1
)

loss_bce = model_bce.train(X_bce_normalized, y_bce)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(loss_bce, label='Loss BCE')
plt.title('Loss BCE durante il training')
plt.xlabel('Campioni')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
'''
# --- Istanziamento modelli --- #
model_bce = NeuralNetworkBCE(
    input_size=6,
    hidden_sizes=[8],
    output_size=1,
    learning_rate=0.01,
    momentum=0.9,
    epochs=1
)
model_bce.print_structure()

model_mse = NeuralNetworkMSE(
    input_size=12,
    hidden_sizes=[8],
    output_size=3,
    learning_rate=0.01,
    momentum=0.9,
    epochs=1
)
model_mse.print_structure()

# --- Training --- #
loss_bce = model_bce.train(X_bce, y_bce)
print(loss_bce[-1])

loss_mse = model_mse.train(X_mse_normalized, y_mse_normalized)
print(loss_mse[-1])

# --- Plot --- #
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(loss_bce, label='Loss BCE')
plt.title('Loss BCE durante il training')
plt.xlabel('Campioni')
#plt.xscale('log')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss_mse, label='Loss MSE', color='orange')
plt.title('Loss MSE durante il training')
plt.xlabel('Campioni')
#plt.xscale('log')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
