import numpy as np
from losses import BCE

class NeuralNetworkBFGS:
    def __init__(self, input_size, hidden_size, output_size, loss):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.loss = loss

        # Initialize weights and biases randomly
        self.wh = np.random.randn(self.input_size, self.hidden_size)
        self.bh = np.zeros(self.hidden_size)
        self.wo = np.random.randn(self.hidden_size, self.output_size)
        self.bo = np.zeros(self.output_size)

        # Store the dimensions for easier indexing
        self.hidden_param_size = input_size + 1  # Weights + bias for each hidden neuron
        self.output_param_size = hidden_size + 1 # Weights + bias for each output neuron

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)

    def forward(self, X):
        # Hidden layer
        self.net_h = np.dot(X, self.wh) + self.bh
        self.hidden_output = self.sigmoid(self.net_h)

        # Output layer
        self.net_o = np.dot(self.hidden_output, self.wo) + self.bo
        self.predicted_output = self.sigmoid(self.net_o)

        return self.predicted_output

    def bce(self, y_true, y_pred, epsilon=1e-15):
        y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
    
    def bce_derivative(self, y_true, y_pred, epsilon=1e-15):
        y_pred_clipped = np.clip(y_pred, 1e-15,1 - epsilon)  # avoids div by 0
        return -np.mean(y_true/y_pred_clipped - (1-y_true)/(1-y_pred_clipped))

    def _flatten_params(self):
        return np.concatenate([
            self.wh.ravel(),
            self.bh.ravel(),
            self.wo.ravel(),
            self.bo.ravel()
        ])

    def _unflatten_params(self, params):
        """Restore the weights and biases from a flattened vector."""
        start = 0
        end = self.input_size * self.hidden_size
        self.wh = params[start:end].reshape(self.input_size, self.hidden_size)
        start = end
        end = start + self.hidden_size
        self.bh = params[start:end]
        start = end
        end = start + self.hidden_size * self.output_size
        self.wo = params[start:end].reshape(self.hidden_size, self.output_size)
        start = end
        end = start + self.output_size
        self.bo = params[start:end]

    def compute_gradients(self, X, y):
        output_delta = np.multiply(self.loss.derivative(y, self.predicted_output), self.sigmoid_derivative(self.net_o))   

        hidden_delta = np.multiply(np.dot(output_delta, self.wo.T), self.sigmoid_derivative(self.net_h))

        # Gradients
        self.grad_wo = np.dot(self.hidden_output.T, output_delta)
        self.grad_bo = np.sum(output_delta, axis=0)
        self.grad_wh = np.dot(X.T, hidden_delta)
        self.grad_bh = np.sum(hidden_delta, axis=0)

        return np.concatenate([
            self.grad_wh.ravel(),
            self.grad_bh.ravel(),
            self.grad_wo.ravel(),
            self.grad_bo.ravel()
        ])
    
    def initialize_hessian(self):
        H_k_blocks = {'hidden': [], 'output': []}
        for _ in range(self.hidden_size):
            H_k_blocks['hidden'].append(np.eye(self.input_size + 1))
        for _ in range(self.output_size):
            H_k_blocks['output'].append(np.eye(self.hidden_size + 1))
        return H_k_blocks

    def _update_block_diagonal_hessian(self, H_k_blocks, s_k, y_k):
        """Updates the blocks of the inverse Hessian approximation."""
        s_k_ptr = 0
        y_k_ptr = 0

        # Update hidden layer blocks
        for i in range(self.hidden_size):
            s_k_block = s_k[s_k_ptr : s_k_ptr + self.input_size + 1][:, np.newaxis]
            y_k_block = y_k[y_k_ptr : y_k_ptr + self.input_size + 1][:, np.newaxis]
            H_k = H_k_blocks['hidden'][i]
            self._update_bfgs_block(H_k, s_k_block, y_k_block)
            H_k_blocks['hidden'][i] = H_k
            s_k_ptr += self.input_size + 1
            y_k_ptr += self.input_size + 1

        # Update output layer blocks
        for i in range(self.output_size):
            s_k_block = s_k[s_k_ptr : s_k_ptr + self.hidden_size + 1][:, np.newaxis]
            y_k_block = y_k[y_k_ptr : y_k_ptr + self.hidden_size + 1][:, np.newaxis]
            H_k = H_k_blocks['output'][i]
            self._update_bfgs_block(H_k, s_k_block, y_k_block)
            H_k_blocks['output'][i] = H_k
            s_k_ptr += self.hidden_size + 1
            y_k_ptr += self.hidden_size + 1

    def _update_bfgs_block(self, H_k, s_k, y_k, epsilon=1e-8):
        """Applies the BFGS update to a single block of the inverse Hessian."""
        s_k_t = s_k.T
        y_k_t = y_k.T
        s_k_dot_y_k = np.dot(s_k_t, y_k)[0, 0]

        if s_k_dot_y_k > epsilon:
            rho_k = 1.0 / s_k_dot_y_k
            I = np.eye(H_k.shape[0])
            term1 = (I - rho_k * np.dot(s_k, y_k_t))
            term2 = (I - rho_k * np.dot(y_k, s_k_t))
            term3 = rho_k * np.dot(s_k, s_k_t)
            return np.dot(term1, np.dot(H_k, term2)) + term3
        return H_k # Return the old H_k if s_k_dot_y_k is too small

    def train(self, X_train, y_train, learning_rate=0.01, max_iter=100, tol=1e-5):
        params = self._flatten_params()
        H_k_blocks = self.initialize_hessian()
        history = []

        for k in range(max_iter):
            self._unflatten_params(params)
            y_predicted = self.forward(X_train)
            current_loss = self.loss.compute(y_train, y_predicted)
            gradients = self.compute_gradients(X_train, y_train)
            history.append(current_loss)

            if np.linalg.norm(gradients) < tol:
                print(f"Converged at iteration {k+1}, loss: {current_loss:.6f}")
                break

            # Calculate search direction using the block-diagonal inverse Hessian
            p_k = np.zeros_like(gradients)
            ptr = 0

            # Hidden layer blocks
            for i in range(self.hidden_size):
                grad_block = gradients[ptr : ptr + self.input_size + 1]
                H_block = H_k_blocks['hidden'][i]
                p_k[ptr : ptr + self.input_size + 1] = -np.dot(H_block, grad_block)
                ptr += self.input_size + 1

            # Output layer blocks
            for i in range(self.output_size):
                grad_block = gradients[ptr : ptr + self.hidden_size + 1]
                H_block = H_k_blocks['output'][i]
                p_k[ptr : ptr + self.hidden_size + 1] = -np.dot(H_block, grad_block)
                ptr += self.hidden_size + 1

            # TODO Wolfe line search
            alpha_k = learning_rate
            params_new = params + alpha_k * p_k
            self._unflatten_params(params_new)
            loss_new = self.loss.compute(y_train, self.forward(X_train))

            # Basic Wolfe condition check (sufficient decrease)
            c1 = 1e-4
            self._unflatten_params(params) # Restore old params for gradient calculation
            if loss_new > current_loss + c1 * alpha_k * np.dot(gradients, p_k):
                print(f"Line search failed at iteration {k+1}")
                break

            s_k = params_new - params
            gradients_new = self.compute_gradients(X_train, y_train)
            y_k = gradients_new - gradients

            # Update the block-diagonal inverse Hessian approximation
            self._update_block_diagonal_hessian(H_k_blocks, s_k, y_k)

            params = params_new

        else:
            print(f"Maximum iterations reached, final loss: {current_loss:.6f}")

        self._unflatten_params(params)
        return history