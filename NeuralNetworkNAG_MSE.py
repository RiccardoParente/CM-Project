import numpy as np
from NeuralNetwork import NeuralNetwork
import time

class NeuralNetworkNAG_MSE(NeuralNetwork):
    def __init__(self, input_size, hidden_size, output_size, loss, regularization, learning_rate, momentum):
        super().__init__(input_size, hidden_size, output_size, loss, regularization, momentum, learning_rate )
        # velocity initialization
        self.v_wh = np.zeros_like(self.wh)
        self.v_bh = np.zeros_like(self.bh)
        self.v_wo = np.zeros_like(self.wo)
        self.v_bo = np.zeros_like(self.bo)

    def train(self, X_train, y_train, epochs, batch=False):
        loss_mse = []
        gradients = []
        prev_loss = None
        patience = 50
        patience_counter = 0
        tolerance = 1e-3
        exit = False
        x_size = 1 if batch else X_train.shape[0]
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
                output = net_output

                # Compute Loss
                loss = self.loss.compute(output, y) + self.regularization*np.linalg.norm(self.flatten_params())
                loss_mse.append(loss)
                

                # Divergence check
                if np.isnan(loss) or loss > 1e5:
                    print("❌ Loss diverging. Stopping.")
                    exit = True
                    break

                # Convergence check
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
                sigma_output = -self.loss.derivative(output, y)
                delta_wo = np.dot(hidden_output.T, sigma_output) / x.shape[0]
                delta_bo = np.sum(sigma_output, axis=0) / x.shape[0]
                
                sigma_hidden = np.dot(sigma_output, wo_pre.T) * self.leacky_relu_derivative(net_hidden)
                delta_wh = np.dot(x.T, sigma_hidden) / x.shape[0]
                delta_bh = np.sum(sigma_hidden, axis=0) / x.shape[0]

                grad = np.array([])
                grad = np.concatenate(([], delta_wh.flatten(), delta_bh.flatten(), delta_wo.flatten(), delta_bo.flatten()))
                gradients.append(np.linalg.norm(grad))

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

                mean_time += (time.time() - start_time) 

            if exit:
                break

        return loss_mse, mean_time / (epochs * x_size), gradients
        

    def anticipate_weights(self):
        '''function to anticipate the weights'''
        wh_pre = self.wh + (self.momentum * self.v_wh)
        bh_pre = self.bh + (self.momentum * self.v_bh)

        wo_pre = self.wo + (self.momentum * self.v_wo)
        bo_pre = self.bo + (self.momentum * self.v_bo)

        return wh_pre, bh_pre, wo_pre, bo_pre