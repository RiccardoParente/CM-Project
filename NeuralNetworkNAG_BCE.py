import numpy as np
from NeuralNetwork import NeuralNetwork
import time

class NeuralNetworkNAG_BCE(NeuralNetwork):
    def __init__(self, input_size, hidden_size, output_size, loss, regularization, learning_rate, momentum):
        super().__init__(input_size, hidden_size, output_size, loss, regularization, momentum, learning_rate)
        # inizializza le velocità dei pesi a zero
        self.v_wh = np.zeros_like(self.wh)
        self.v_bh = np.zeros_like(self.bh)
        self.v_wo = np.zeros_like(self.wo)
        self.v_bo = np.zeros_like(self.bo)
    
    def train(self, X_train, y_train, epochs, batch=False):
        loss_bce = []
        t = 1
        prev_loss = None
        patience = 50
        patience_counter = 0
        tolerance = 1e-3
        exit = False
        x_size = 1 if batch else X_train.shape[0]
        T = epochs*x_size
        mean_time = 0

        for i in range(epochs):
            indices = np.random.permutation(len(X_train))
            X_train = X_train[indices]
            y_train = y_train[indices]
            for i in range(x_size):
                start_time = time.time()
                x = X_train if batch else np.array([X_train[i]])
                y = y_train if batch else y_train[i]

                wh_pre, bh_pre, wo_pre, bo_pre = self.anticipate_weights()

                # Forward propagation
                net_hidden = np.dot(x, wh_pre) + bh_pre
                hidden_output = self.leacky_relu(net_hidden)
                net_output = np.dot(hidden_output, wo_pre) + bo_pre
                output = self.sigmoid(net_output)
                
                loss = self.loss.compute(output, y) + self.regularization*np.linalg.norm(self.flatten_params())
                loss_bce.append(loss)

                # Controllo divergenza
                if np.isnan(loss) or loss > 1e5:
                    print("❌ Loss diverging. Stopping.")
                    exit = True
                    break

                # Controllo convergenza
                if prev_loss is not None:
                    if abs(loss - prev_loss) < tolerance:
                        patience_counter += 1
                        if patience_counter >= patience:
                            print("✅ Loss converged. Stopping.")
                            exit = True
                            break
                    else:
                        patience_counter = 0

                prev_loss = loss

                # Backward propagation
                sigma_output = y - output
                delta_wo = np.dot(hidden_output.T, sigma_output) / x.shape[0]
                delta_bo = sum(sigma_output) / x.shape[0]
        
                sigma_hidden = np.dot(sigma_output, wo_pre.T) * self.leacky_relu_derivative(net_hidden)
                delta_wh = np.dot(x.T, sigma_hidden) / x.shape[0]
                delta_bh = sum(sigma_hidden) / x.shape[0]
                
                # Update velocity
                self.v_wh = ((self.learning_rate * delta_wh) + (self.momentum * self.v_wh))
                self.v_bh = ((self.learning_rate * delta_bh) + (self.momentum * self.v_bh))
                self.v_wo = ((self.learning_rate * delta_wo) + (self.momentum * self.v_wo))
                self.v_bo = ((self.learning_rate * delta_bo) + (self.momentum * self.v_bo))

                # Update weights and biases
                self.wh = self.wh + self.v_wh - (2*self.regularization*self.wh)
                self.bh = self.bh + self.v_bh - (2*self.regularization*self.bh)
                self.wo = self.wo + self.v_wo - (2*self.regularization*self.wo)
                self.bo = self.bo + self.v_bo - (2*self.regularization*self.bo)

                # Update momentum
                self.momentum = self.momentum *(1 - (t/T))
                t+=1

                mean_time += (time.time() - start_time)
                
            if exit:
                break

        return loss_bce, mean_time / T

    # Funzione per anticipare i pesi (pre-update dei pesi)
    def anticipate_weights(self):
        # Calcolare i pesi anticipati con la velocità
        wh_pre = self.wh + (self.momentum * self.v_wh)
        bh_pre = self.bh + (self.momentum * self.v_bh)

        wo_pre = self.wo + (self.momentum * self.v_wo)
        bo_pre = self.bo + (self.momentum * self.v_bo)

        return wh_pre, bh_pre, wo_pre, bo_pre