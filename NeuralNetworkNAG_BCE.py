import numpy as np
from NeuralNetwork import NeuralNetwork
import time

class NeuralNetworkNAG_BCE(NeuralNetwork):
    def __init__(self, input_size, hidden_size, output_size, loss, layers, regularization, learning_rate, momentum):
        super().__init__(input_size, hidden_size, output_size, loss, layers, regularization, momentum, learning_rate)
        # velocity initialization
        self.v_wh = np.zeros_like(self.wh)
        self.v_bh = np.zeros_like(self.bh)
        self.v_w_inner = []
        self.v_b_inner = []
        for h in range(len(self.w_inner)):
            self.v_w_inner.append(np.zeros_like(self.w_inner[h]))
            self.v_b_inner.append(np.zeros_like(self.b_inner[h]))
        self.v_wo = np.zeros_like(self.wo)
        self.v_bo = np.zeros_like(self.bo)
    
    def train(self, X_train, y_train, epochs, batch=False):
        loss_bce = []
        mean_loss_epoch = []
        gradients = []
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
            losses = []
            for i in range(x_size):
                start_time = time.time()
                x = X_train if batch else np.array([X_train[i]])
                y = y_train if batch else y_train[i]

                wh_pre, bh_pre, w_inner_pre, b_inner_pre, wo_pre, bo_pre = self.anticipate_weights()

                # Forward propagation
                net_hidden = np.dot(x, wh_pre) + bh_pre
                hidden_output = self.tanh(net_hidden)
                inn_out = hidden_output
                net_inner = net_hidden
                net_inners = [net_inner]
                inner_outputs = [inn_out]
                for h in range(len(self.w_inner)):
                    net_inner = np.dot(inn_out, w_inner_pre[h]) + b_inner_pre[h]
                    inn_out = self.tanh(net_inner)
                    net_inners.append(net_inner)
                    inner_outputs.append(inn_out)
                net_output = np.dot(inn_out, wo_pre) + bo_pre
                output = self.sigmoid(net_output)
                
                loss = self.loss.compute(output, y) + self.regularization*np.linalg.norm(self.flatten_params())
                if batch:
                    loss_bce.append(loss)
                else:
                    losses.append(loss)

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
                sigma_output = y - output
                delta_wo = np.dot(inn_out.T, sigma_output) / x.shape[0]
                delta_bo = sum(sigma_output) / x.shape[0]
                delta_w_inner = []
                delta_b_inner = []
                sigma_inner = np.dot(sigma_output, wo_pre.T) * self.tanh_derivative(net_inners[-1])
                for h in reversed(range(len(self.w_inner))):
                    delta_w_inner.append(np.dot(inner_outputs[h].T, sigma_inner) / x.shape[0])
                    delta_b_inner.append(sum(sigma_inner) / x.shape[0])
                    sigma_inner = np.dot(sigma_inner, w_inner_pre[h].T) * self.tanh_derivative(net_inners[h])

                sigma_hidden = sigma_inner
                delta_wh = np.dot(x.T, sigma_hidden) / x.shape[0]
                delta_bh = sum(sigma_hidden) / x.shape[0]

                grad = np.array([])
                grad = np.concatenate(([], delta_wh.flatten(), delta_bh.flatten()))
                for h in range(len(self.w_inner)):
                    grad = np.concatenate((grad, delta_w_inner[h].flatten(), delta_b_inner[h].flatten()))
                grad = np.concatenate((grad, delta_wo.flatten(), delta_bo.flatten()))
                gradients.append(np.linalg.norm(grad))

                # Update velocity
                self.v_wh = ((self.learning_rate * delta_wh) + (self.momentum * self.v_wh))
                self.v_bh = ((self.learning_rate * delta_bh) + (self.momentum * self.v_bh))
                for h in range(len(self.w_inner)):
                    self.v_w_inner[h] = ((self.learning_rate * delta_w_inner[h]) + (self.momentum * self.v_w_inner[h]))
                    self.v_b_inner[h] = ((self.learning_rate * delta_b_inner[h]) + (self.momentum * self.v_b_inner[h]))
                self.v_wo = ((self.learning_rate * delta_wo) + (self.momentum * self.v_wo))
                self.v_bo = ((self.learning_rate * delta_bo) + (self.momentum * self.v_bo))

                # Update weights and biases
                self.wh = self.wh + self.v_wh - (2*self.regularization*self.wh)
                self.bh = self.bh + self.v_bh - (2*self.regularization*self.bh)
                for h in range(len(self.w_inner)):
                    self.w_inner[h] = self.w_inner[h] + self.v_w_inner[h] - (2*self.regularization*self.w_inner[h])
                    self.b_inner[h] = self.b_inner[h] + self.v_b_inner[h] - (2*self.regularization*self.b_inner[h])
                self.wo = self.wo + self.v_wo - (2*self.regularization*self.wo)
                self.bo = self.bo + self.v_bo - (2*self.regularization*self.bo)

                # Update momentum
                self.momentum = self.momentum *(1 - (t/T))
                

                mean_time += (time.time() - start_time)
            t+=1
            mean_loss_epoch.append(np.mean(losses))    
            
            if exit:
                break
        if batch:
            return loss_bce, mean_time / T, gradients
        else:
            return mean_loss_epoch, mean_time / T, gradients

    def anticipate_weights(self):
        '''function to anticipate the weights'''
        wh_pre = self.wh + (self.momentum * self.v_wh)
        bh_pre = self.bh + (self.momentum * self.v_bh)
        w_inner_pre = []
        b_inner_pre = []
        for h in range(len(self.w_inner)):
            w_inner_pre.append(self.w_inner[h] + (self.momentum * self.v_w_inner[h]))
            b_inner_pre.append(self.b_inner[h] + (self.momentum * self.v_b_inner[h]))
        wo_pre = self.wo + (self.momentum * self.v_wo)
        bo_pre = self.bo + (self.momentum * self.v_bo)

        return wh_pre, bh_pre, w_inner_pre, b_inner_pre, wo_pre, bo_pre