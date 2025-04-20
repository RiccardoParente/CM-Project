import numpy as np
from NeuralNetwork import NeuralNetwork

class NeuralNetworkNAG_MSE(NeuralNetwork):
    def __init__(self, input_size, hidden_size, output_size, loss, regularization, learning_rate, momentum):
        super().__init__(input_size, hidden_size, output_size, loss, regularization, learning_rate, momentum)
        # inizializza le velocità dei pesi a zero
        self.v_wh = np.zeros_like(self.wh)
        self.v_bh = np.zeros_like(self.bh)
        self.v_wo = np.zeros_like(self.wo)
        self.v_bo = np.zeros_like(self.bo)

    def train(self, X, y, epochs):
        loss_mse = []
        prev_loss = None
        patience = 50
        patience_counter = 0
        tolerance = 1e-6

        for j in range(epochs):
            indices = np.random.permutation(len(X))
            X = X[indices]
            y = y[indices]

            wh_pre, bh_pre, wo_pre, bo_pre = self.anticipate_weights()

            # Forward propagation
            net_hidden = np.dot(X, wh_pre) + bh_pre
            hidden_output = self.leacky_relu(net_hidden)
            net_output = np.dot(hidden_output, wo_pre) + bo_pre
            output = net_output

            # Compute Loss
            loss = self.loss.compute(output, y) + self.regularization*np.linalg.norm(self.flatten_params())
            loss_mse.append(loss)

            # Controllo divergenza
            if np.isnan(loss) or loss > 1e5:
                print("❌ Loss diverging. Stopping.")
                break

            # Controllo convergenza
            if prev_loss is not None:
                if abs(loss - prev_loss) < tolerance:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print("✅ Loss converged. Stopping.")
                        break
                else:
                    patience_counter = 0

            prev_loss = loss

            # Backward propagation
            sigma_output = -self.loss.derivative(output, y)
            delta_wo = np.dot(hidden_output.T, sigma_output) / X.shape[0]
            delta_bo = sum(sigma_output) / X.shape[0]
    
            sigma_hidden = np.dot(sigma_output, wo_pre.T) * self.leacky_relu_derivative(net_hidden)
            delta_wh = np.dot(X.T, sigma_hidden) / X.shape[0]
            delta_bh = sum(sigma_hidden) / X.shape[0]

            # Update weights and biases
            self.wh = self.wh + ((self.learning_rate * delta_wh) + (self.momentum * self.v_wh) - (2*self.regularization*self.wh))
            self.bh = self.bh + ((self.learning_rate * delta_bh) + (self.momentum * self.v_bh) - (2*self.regularization*self.bh))
            self.wo = self.wo + ((self.learning_rate * delta_wo) + (self.momentum * self.v_wo) - (2*self.regularization*self.wo))
            self.bo = self.bo + ((self.learning_rate * delta_bo) + (self.momentum * self.v_bo) - (2*self.regularization*self.bo))

            # Update velocity
            self.v_wh = ((self.learning_rate * delta_wh) + (self.momentum * self.v_wh))
            self.v_bh = ((self.learning_rate * delta_bh) + (self.momentum * self.v_bh))
            self.v_wo = ((self.learning_rate * delta_wo) + (self.momentum * self.v_wo))
            self.v_bo = ((self.learning_rate * delta_bo) + (self.momentum * self.v_bo))
            print(self.wh, self.bh, self.wo, self.bo)

        return loss_mse

    def leacky_relu(self, z):
        alpha = 0.01
        return np.maximum(alpha * z, z)
    
    def leacky_relu_derivative(self, a):
        alpha = 0.01
        return np.where(a > 0, 1, alpha)
    
    # Funzione per anticipare i pesi (pre-update dei pesi)
    def anticipate_weights(self):
        # Calcolare i pesi anticipati con la velocità
        wh_pre = self.wh + (self.momentum * self.v_wh)
        bh_pre = self.bh + (self.momentum * self.v_bh)

        wo_pre = self.wo + (self.momentum * self.v_wo)
        bo_pre = self.bo + (self.momentum * self.v_bo)

        return wh_pre, bh_pre, wo_pre, bo_pre