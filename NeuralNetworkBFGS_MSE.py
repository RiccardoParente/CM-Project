import numpy as np
from NeuralNetwork import NeuralNetwork
import time

class NeuralNetworkBFGS_MSE(NeuralNetwork):

    def forward(self, X):
        # Hidden layer
        self.net_h = np.dot(X, self.wh) + self.bh
        self.hidden_output = self.leacky_relu(self.net_h)

        # Output layer
        self.net_o = np.dot(self.hidden_output, self.wo) + self.bo
        self.predicted_output = self.net_o
        return self.predicted_output

    def compute_gradients(self, X, y):
        output_delta = self.loss.derivative(self.predicted_output, y)
        hidden_delta = np.dot(output_delta, self.wo.T) * self.leacky_relu_derivative(self.net_h)
        grad_wo = np.dot(self.hidden_output.T, output_delta)
        grad_bo = np.sum(output_delta, axis=0)
        grad_wh = np.dot(X.T, hidden_delta)
        grad_bh = np.sum(hidden_delta, axis=0)

        output = np.array([])
        for i in range(self.hidden_size):
            output = np.concatenate((output, grad_wh[:, i], [grad_bh[i]]))
        for i in range(self.output_size):
            output = np.concatenate((output, grad_wo[:, i], [grad_bo[i]]))
        return output
    
    def initialize_hessian(self):
        '''Hessian initialization'''
        H_k_blocks = {'hidden': [], 'output': []}
        for _ in range(self.hidden_size):
            H_k_blocks['hidden'].append(np.eye(self.input_size + 1))
        for _ in range(self.output_size):
            H_k_blocks['output'].append(np.eye(self.hidden_size + 1))
        return H_k_blocks

    def update_hessian(self, H_k_blocks, s_k, y_k):
        '''Approximate inverse Hessian update'''

        for i in range(self.hidden_size):
            ptr = (self.input_size+1)*i
            s_k_block = s_k[ptr : ptr + self.input_size+1][:, np.newaxis]
            y_k_block = y_k[ptr : ptr + self.input_size+1][:, np.newaxis]
            H_k_blocks['hidden'][i] = self.update_block(H_k_blocks['hidden'][i], s_k_block, y_k_block)

        offset = (self.input_size*self.hidden_size)+self.hidden_size
        for i in range(self.output_size):
            ptr = offset+((self.hidden_size+1)*i)
            s_k_block = s_k[ptr : ptr + self.hidden_size+1][:, np.newaxis]
            y_k_block = y_k[ptr : ptr + self.hidden_size+1][:, np.newaxis]
            H_k_blocks['output'][i] = self.update_block(H_k_blocks['output'][i], s_k_block, y_k_block)
        return H_k_blocks

    def update_block(self, H_k, s_k, y_k, epsilon=1e-4):
        '''Approximate inverse Hessian block update'''
        s_k_t = s_k.T
        y_k_t = y_k.T
        s_k_dot_y_k = np.dot(y_k_t, s_k)[0, 0]

        if s_k_dot_y_k > epsilon:
            rho_k = 1.0 / s_k_dot_y_k
            I = np.eye(H_k.shape[0])
            term1 = (I - rho_k * np.dot(s_k, y_k_t))
            term2 = (I - rho_k * np.dot(y_k, s_k_t))
            term3 = rho_k * np.dot(s_k, s_k_t)
            return np.dot(term1, np.dot(H_k, term2)) + term3
        return H_k
    
    def line_search_wolfe(self, p_k, grad_f_k, X_train, y_train, t, T, c1=1e-4, c2=0.9, max_alpha=1.0):
        '''Wolfe line search'''

        alpha_low = 0.0
        alpha_high = max_alpha
        phi_prev = self.current_loss
        dphi_prev = np.dot(grad_f_k, p_k)
        params = self.flatten_params()
        alpha_i = 0

        for i in range(20):
            alpha_i = (alpha_low + alpha_high) / 2.0
            grads = 0
            params_temp = params + alpha_i * p_k
            self.unflatten_params(params_temp)
            self.forward(X_train)
            phi_i = self.loss.compute(self.predicted_output, y_train) + self.regularization*np.linalg.norm(params)
            grads = self.compute_gradients(X_train, y_train) / X_train.shape[0]
            dphi_i = np.dot(grads, p_k)

            # Check sufficient decrease
            if phi_i <= phi_prev + c1 * alpha_i * dphi_prev:
                # Check curvature condition
                if np.abs(dphi_i) <= c2 * np.abs(dphi_prev):
                    return alpha_i

                if dphi_i > 0:
                    alpha_high = alpha_i
                else:
                    alpha_low = alpha_i
            else:
                alpha_high = alpha_i

        return alpha_i*(1-(t/T))

    def train(self, X_train, y_train, epochs=100, tol=1e-4, batch=False):
        params = self.flatten_params()
        H_k_blocks = self.initialize_hessian()
        history = []
        t = 1
        gradients = 0
        self.current_loss = 0
        best_gradient = [float('inf')]
        best_iter = 0
        prev_loss = None
        patience_counter = 0
        exit = False
        x_size = 1 if batch else X_train.shape[0]
        T = epochs*x_size
        mean_time = 0

        for k in range(epochs):
            indices = np.random.permutation(len(X_train))
            X_train = X_train[indices]
            y_train = y_train[indices]
            for i in range(x_size):
                start_time = time.time()
                x = X_train if batch else np.array([X_train[i]])
                y = y_train if batch else y_train[i]
                self.unflatten_params(params)
                self.forward(x)
                self.current_loss = self.loss.compute(self.predicted_output, y) + self.regularization*np.linalg.norm(params)
                gradients = self.compute_gradients(x, y) / x.shape[0]
                history.append(self.current_loss)

                # Controllo divergenza
                if np.isnan(self.current_loss) or self.current_loss > 1e5:
                    print("❌ Loss diverging. Stopping.")
                    exit = True
                    break

                # Controllo convergenza
                if prev_loss is not None:
                    if abs(self.current_loss - prev_loss) < tol:
                        patience_counter += 1
                        if patience_counter >= 5:
                            print(f"✅ Loss converged at iteration {k+1}, loss: {self.current_loss:.6f}, gradient norm: {np.linalg.norm(best_gradient)}. Stopping.")
                            exit = True
                            break
                    else:
                        patience_counter = 0

                prev_loss = self.current_loss

                if np.linalg.norm(gradients) < np.linalg.norm(best_gradient):
                    best_gradient = gradients
                    best_iter = k

                #if np.linalg.norm(gradients) < tol:
                    #print(f"Converged at iteration {k+1}, loss: {self.current_loss:.6f}, gradient norm: {np.linalg.norm(best_gradient)}")
                    #break

                p_k = np.zeros_like(gradients)

                for i in range(self.hidden_size):
                    ptr = (self.input_size+1)*i
                    grad_block = gradients[ptr : ptr + self.input_size + 1]
                    H_block = H_k_blocks['hidden'][i]
                    p_k[ptr : ptr + self.input_size + 1] = -np.dot(H_block, grad_block)

                offset = (self.input_size*self.hidden_size)+self.hidden_size
                for i in range(self.output_size):
                    ptr = offset+((self.hidden_size+1)*i)
                    grad_block = gradients[ptr : ptr + self.hidden_size + 1]
                    H_block = H_k_blocks['output'][i]
                    p_k[ptr : ptr + self.hidden_size + 1] = -np.dot(H_block, grad_block)

                alpha_k = self.line_search_wolfe(p_k, gradients, x, y, t, T)

                params_new = params + (alpha_k * p_k) - (self.regularization * params)
                self.unflatten_params(params_new)

                s_k = params_new - params
                self.forward(x)
                gradients_new = self.compute_gradients(x, y) / x.shape[0]
                y_k = gradients_new - gradients

                self.update_hessian(H_k_blocks, s_k, y_k)

                params = params_new

                mean_time += (time.time() - start_time)

            if exit:
                break

        else:
            print(f"Maximum iterations reached, final loss: {self.current_loss:.6f}, best gradient: {best_iter+1}, gradient norm: {np.linalg.norm(best_gradient)}")

        self.unflatten_params(params)
        return history, mean_time / T