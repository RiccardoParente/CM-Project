import numpy as np

class NeuralNetworkBFGS_MSE:
    def __init__(self, input_size, hidden_size, output_size, loss):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.loss = loss

        # Initialize weights and biases randomly
        self.wh = np.random.randn(self.input_size, self.hidden_size) * np.sqrt(2.0 / input_size)
        self.bh = np.zeros(self.hidden_size)
        self.wo = np.random.randn(self.hidden_size, self.output_size) * np.sqrt(2.0 / hidden_size)
        self.bo = np.zeros(self.output_size)

        self.hidden_param_size = input_size + 1  # Weights + bias for each hidden neuron
        self.output_param_size = hidden_size + 1 # Weights + bias for each output neuron

    def leacky_relu(self, z):
        alpha = 0.01
        return np.maximum(alpha * z, z)
    
    def leacky_relu_derivative(self, a):
        alpha = 0.01
        return np.where(a > 0, 1, alpha)

    def forward(self, X):
        # Hidden layer
        self.net_h = np.dot(X, self.wh) + self.bh
        self.hidden_output = self.leacky_relu(self.net_h)

        # Output layer
        self.net_o = np.dot(self.hidden_output, self.wo) + self.bo
        self.predicted_output = self.net_o
        return self.predicted_output

    def flatten_params(self):
        output = np.array([])
        for i in range(self.hidden_size):
            output = np.concatenate((output, self.wh[:,i], [self.bh[i]]))
        for i in range(self.output_size):
            output = np.concatenate((output, self.wo[:,i], [self.bo[i]]))
        return output

    def unflatten_params(self, params):
        """Restore the weights and biases from a flattened vector."""
        wh_temp = []
        bh_temp = []
        wo_temp = []
        bo_temp = []
        for i in range(self.hidden_size):
            ptr = self.input_size*i
            wh_temp = np.concatenate((wh_temp, params[ptr:ptr+self.input_size]))
            bh_temp = np.concatenate((bh_temp, [params[ptr+self.input_size]]))
        self.wh = np.array(wh_temp).reshape(self.input_size, self.hidden_size)
        self.bh = np.array(bh_temp)
        offset = (self.input_size*self.hidden_size)+self.hidden_size
        for i in range(self.output_size):
            ptr = offset+(self.hidden_size*i)
            wo_temp = np.concatenate((wo_temp, params[ptr:ptr+self.hidden_size]))
            bo_temp = np.concatenate((bo_temp, [params[ptr+self.hidden_size]]))
        self.wo = np.array(wo_temp).reshape(self.hidden_size, self.output_size)
        self.bo = np.array(bo_temp)

    def compute_gradients(self, X, y):
        output_delta = self.loss.derivative(self.predicted_output, y)
        hidden_delta = np.dot(output_delta, self.wo.T) * self.leacky_relu_derivative(self.net_h)
        grad_wo = self.hidden_output.T * output_delta[:, np.newaxis]
        grad_bo = output_delta
        grad_wh = X.T * hidden_delta[:, np.newaxis]
        grad_bh = hidden_delta

        output = np.array([])
        for i in range(self.hidden_size):
            output = np.concatenate((output, grad_wh[i], [grad_bh[i]]))
        for i in range(self.output_size):
            output = np.concatenate((output, grad_wo[i], [grad_bo[i]]))
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

        def phi(params, alpha):
            params_temp = params + alpha * p_k
            self.unflatten_params(params_temp)
            self.forward(X_train)
            return self.loss.compute(self.predicted_output, y_train)+0.005*np.linalg.norm(params_temp)

        def dphi(params, alpha):
            params_temp = params + alpha * p_k
            self.unflatten_params(params_temp)
            self.forward(X_train)
            return np.dot(self.compute_gradients(X_train, y_train), p_k)

        alpha_low = 0.0
        alpha_high = max_alpha
        phi_prev = self.current_loss
        dphi_prev = np.dot(grad_f_k, p_k)
        params = self.flatten_params()
        alpha_i = 0

        for i in range(20):
            alpha_i = (alpha_low + alpha_high) / 2.0
            loss = 0
            grads = 0
            params_temp = params + alpha_i * p_k
            self.unflatten_params(params_temp)
            for j in range(X_train.shape[0]):
                x = X_train[j]
                y = y_train[j]
                predicted_out = self.forward(x)
                loss += self.loss.compute(predicted_out, y)
                grads += self.compute_gradients(x, y)
            phi_i = loss/X_train.shape[0]
            grads /= X_train.shape[0]
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

    def train(self, X_train, y_train, max_iter=100, tol=1e-4):
        params = self.flatten_params()
        H_k_blocks = self.initialize_hessian()
        history = []
        T = max_iter
        t = 1
        gradients = 0
        self.current_loss = 0
        best_gradient = [float('inf')]
        best_iter = 0

        for k in range(max_iter):
            indices = np.random.permutation(len(X_train))
            X_train = X_train[indices]
            y_train = y_train[indices]
            self.unflatten_params(params)
            gradients = 0
            self.current_loss = 0
            for j in range(X_train.shape[0]):
                x = X_train[j]
                y = y_train[j]
                y_predicted = self.forward(x)
                self.current_loss += self.loss.compute(y_predicted, y)+0.005*np.linalg.norm(params)
                gradients += self.compute_gradients(x, y)

            gradients /= X_train.shape[0]
            self.current_loss /= X_train.shape[0]
            history.append(self.current_loss)

            if np.linalg.norm(gradients) < np.linalg.norm(best_gradient):
                best_gradient = gradients
                best_iter = k

            if np.linalg.norm(gradients) < tol:
                print(f"Converged at iteration {k+1}, loss: {self.current_loss:.6f}, gradient norm: {np.linalg.norm(best_gradient)}")
                break

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

            alpha_k = self.line_search_wolfe(p_k, gradients, X_train, y_train, t, T)

            params_new = params + (alpha_k * p_k) - (0.005*params)
            self.unflatten_params(params_new)

            s_k = params_new - params
            gradients_new = 0
            for j in range(X_train.shape[0]):
                x = X_train[j]
                y = y_train[j]
                y_predicted = self.forward(x)
                gradients_new += self.compute_gradients(x, y)
            gradients_new /= X_train.shape[0]
            y_k = gradients_new - gradients

            self.update_hessian(H_k_blocks, s_k, y_k)

            params = params_new

        else:
            print(f"Maximum iterations reached, final loss: {self.current_loss:.6f}, best gradient: {best_iter+1}, gradient norm: {np.linalg.norm(best_gradient)}")

        self.unflatten_params(params)
        return history