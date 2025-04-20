'''import numpy as np

class NeuralNetworkBCEBFGS:
    def __init__(self, input_size, hidden_sizes, output_size, epochs):
        np.random.seed(100)
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes  # lista con i neuroni dei layer nascosti
        self.output_size = output_size
        self.epochs = epochs

        # Inizializzazione dei pesi e bias
        self.w1 = np.random.randn(hidden_sizes[0], input_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros(hidden_sizes[0])

        self.w2 = np.random.randn(output_size, hidden_sizes[0]) * np.sqrt(2.0 / hidden_sizes[0])
        self.b2 = np.zeros(output_size)

        # Per BFGS: inizializza la hessiana per ogni neurone
        self.hessian_hidden = np.array([np.eye(input_size+1) for _ in range(hidden_sizes[0])])
        self.hessian_output = np.array([np.eye(hidden_sizes[0]+1) for _ in range(output_size)])
    
    def train(self, X, y):
        loss_bce = []
        T = len(X)*self.epochs
        t = 1
        prev_loss = None
        patience = 5
        patience_counter = 0
        tolerance = 1e-2
        end = False
        update = True
        for j in range(self.epochs):
            indices = np.random.permutation(len(X))
            X = X[indices]
            y = y[indices]
            for i in range(X.shape[0]):
                
                # Forward propagation
                net_hidden = np.dot(self.w1, X[i]) + self.b1
                act = self.leacky_relu(net_hidden)
                print(f"\n act: \n{act}")
                net_output = np.dot(self.w2, act) + self.b2
                print(f"\n net_output: \n{net_output}")
                output = self.sigmoid(net_output)
                print(f"\n output: \n{output}")

                # Compute Loss
                w_total = np.concatenate([
                    self.w1.flatten(),
                    self.b1.flatten(),
                    self.w2.flatten(),
                    self.b2.flatten()
                ])
                print(w_total)
                loss = self.compute_bce(output, y[i]) + 0.1*np.sum(w_total**2)
                print(f"\n loss: \n{loss}")
                loss_bce.append(loss)

                # Controllo divergenza
                if np.isnan(loss) or loss > 1e5:
                    print("❌ Loss diverging. Stopping.")
                    end = True
                    break

                # Controllo convergenza
                if prev_loss is not None:
                    if abs(loss - prev_loss) < tolerance:
                        patience_counter += 1
                        if patience_counter >= patience:
                            print("✅ Loss converged. Stopping.")
                            end = True
                            break
                    else:
                        patience_counter = 0

                prev_loss = loss

                # Backward propagation
                sigma_output = output - y[i]
                print(f"\n sigma_output: \n{sigma_output}") 
                delta_w2 = (sigma_output * act).reshape(1,self.hidden_sizes[0])
                delta_b2 = sigma_output
                w_output = np.hstack((self.w2, self.b2.reshape(1,1)))
                g_output1 = np.hstack((delta_w2, delta_b2.reshape(1,1)))
                g_output1 = g_output1 + 0.1*w_output
                
                sigma_hidden = (sigma_output * self.w2) * self.leacky_relu_derivative(net_hidden)
                delta_w1 = sigma_hidden.T * X[i]
                delta_b1 = sigma_hidden.reshape(-1)
                w_hidden = np.hstack((self.w1, self.b1.reshape(self.hidden_sizes[0],1)))
                g_hidden1 = np.hstack((delta_w1, delta_b1.reshape(self.hidden_sizes[0],1)))
                g_hidden1 = g_hidden1 + 0.1*w_hidden
                
                print(f"\nG Hidden Pre: \n{g_hidden1}")
                print(f"\nG Output Pre: \n{g_output1}")

                # Compute direction 
                if update:
                    p_output = [np.dot(h,g) for h, g in zip(self.hessian_output, g_output1)]
                    p_output = (-np.array(p_output))
                    p_hidden = [np.dot(h,g) for h, g in zip(self.hessian_hidden, g_hidden1)]
                    p_hidden = (-np.array(p_hidden))
                else:
                    p_output = (-np.array(g_output1))
                    p_hidden = (-np.array(g_hidden1))

                print(f"\nP Hidden : \n{p_hidden}")
                print(f"\nP Output : \n{p_output}")

                # Update weights and save precedent weights
                alpha = self.wolfe_line_search(t,T,X[i],y[i],loss,self.w1,self.b1,self.w2,self.b2,p_hidden,p_output,g_hidden1,g_output1)
                w1_pre = self.w1
                b1_pre = self.b1
                w2_pre = self.w2
                b2_pre = self.b2
                self.w1 = self.w1 + alpha * p_hidden[:, :-1]
                self.b1 = self.b1 + alpha * p_hidden[:, -1]
                self.w2 = self.w2 + alpha * p_output[:, :-1]
                self.b2 = self.b2 + alpha * p_output[:, -1]
                # Pesi e bias di ogni layer
                #print("\nPesi e bias:")
                #print(f"Layer 1 (input -> hidden1):")
                #print(f"Pesos (w1): \n{self.w1}")
                #print(f"Bias (b1): \n{self.b1}")
                
                #print(f"\nLayer 2 (hidden1 -> output):")
                #print(f"Pesos (w2): \n{self.w2}")
                #print(f"Bias (b2): \n{self.b2}")
                

                # Seconda forward propagation
                net_hidden = np.dot(self.w1, X[i]) + self.b1
                act = self.leacky_relu(net_hidden)
                net_output = np.dot(self.w2, act) + self.b2
                output = self.sigmoid(net_output)

                # Seconda backward propagation
                sigma_output = output - y[i]
                delta_w2 = (sigma_output * act).reshape(1,self.hidden_sizes[0])
                delta_b2 = sigma_output
                g_output2 = np.hstack((delta_w2, delta_b2.reshape(1,1)))

                sigma_hidden = (sigma_output * self.w2) * self.leacky_relu_derivative(net_hidden)
                delta_w1 = sigma_hidden.T * X[i]
                delta_b1 = sigma_hidden.reshape(-1)
                g_hidden2 = np.hstack((delta_w1, delta_b1.reshape(self.hidden_sizes[0],1)))

                #Update Hessian
                w_hidden = np.hstack((self.w1, self.b1.reshape(self.hidden_sizes[0],1)))
                w_hidden_pre = np.hstack((w1_pre, b1_pre.reshape(self.hidden_sizes[0],1)))
                w_output = np.hstack((self.w2, self.b2.reshape(1,1)))
                w_output_pre = np.hstack((w2_pre, b2_pre.reshape(1,1)))
                print(f"\nW Hidden: \n{w_hidden}")
                print(f"\nW Hidden Pre: \n{w_hidden_pre}")
                print(f"\nW Output: \n{w_output}")
                print(f"\nW Output Pre: \n{w_output_pre}") 

                print(f"\nG Hidden: \n{g_hidden2}")
                print(f"\nG Hidden Pre: \n{g_hidden1}")
                print(f"\nG Output: \n{g_output2}")
                print(f"\nG Output Pre: \n{g_output1}") 

                s_hidden = w_hidden - w_hidden_pre
                s_output = w_output - w_output_pre
                print("--------------S e Y------------")
                print(s_hidden)
                print(s_output)
                y_hidden = g_hidden2 - g_hidden1
                y_output = g_output2 - g_output1
                print(y_hidden)
                print(y_output)
                update = False
                for i, (h_o, s_o, y_o) in enumerate(zip(self.hessian_output, s_output, y_output)):
                    print("sᵗy output =", np.dot(y_o, s_o))
                    if np.dot(y_o, s_o) > 0:
                        update = True
                        theta = 1/np.dot(y_o,s_o)
                        self.hessian_output[i] = np.array((np.eye(9) - (theta * np.outer(s_o,y_o))) @ h_o @ (np.eye(9) - (theta * np.outer(y_o,s_o))) + (theta * np.outer(s_o,s_o))) 
                    
                for i, (h_h, s_h, y_h) in enumerate(zip(self.hessian_hidden, s_hidden, y_hidden)):
                    print("sᵗy hidden =", np.dot(y_h, s_h))
                    if np.dot(y_h, s_h) > 0:
                        update = True
                        theta = 1/np.dot(y_h,s_h)
                        self.hessian_hidden[i] = np.array((np.eye(7) - (theta * np.outer(s_h,y_h))) @ h_h @ (np.eye(7) - (theta * np.outer(y_h,s_h))) + (theta * np.outer(s_h,s_h)))
                    
                print(f"\nHessian Hidden: \n{self.hessian_hidden}")
                print(f"\nHessian Output: \n{self.hessian_output}")

                t+=1

            if end:
                break

        return loss_bce


    def compute_bce(self,output,y):
        outputs_clipped = np.clip(output, 1e-15, 1-1e-15)
        return np.mean(-(1 - y) * np.log(1 - outputs_clipped) - y * np.log(outputs_clipped))
    
    def compute_bce_derivate(self,output,y):
        outputs_clipped = np.clip(output, 1e-15, 1-1e-15)
        return -np.mean(y/outputs_clipped - (1-y)/(1-outputs_clipped))

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def sigmoid_derivate(selfself, z):
        return z * (1 - z) 
    
    def leacky_relu(self, z):
        alpha = 0.01
        return np.maximum(alpha * z, z)
    
    def leacky_relu_derivative(self, a):
        alpha = 0.01
        return np.where(a > 0, 1, alpha)
    

    def wolfe_line_search(self, t, T, X, y, loss, w1, b1, w2, b2, p1, p2, g1, g2, c1=1e-4, c2=0.9, alpha_init = 1):
        alpha_max = alpha_init 
        alpha_low = 0.0
        for i in range(20):
            alpha = (alpha_low + alpha_max) / 2.0
            w1_new = w1 + alpha * p1[:, :-1]
            b1_new = b1 + alpha * p1[:, -1]
            w2_new = w2 + alpha * p2[:, :-1]
            b2_new = b2 + alpha * p2[:, -1]
            p_total = np.concatenate([p1.flatten(), p2.flatten()])
            g_total = np.concatenate([g1.flatten(), g2.flatten()])
            print("gᵗp =", np.dot(g_total, p_total))
            # Forward propagation
            net_hidden = np.dot(w1_new, X) + b1_new
            act = self.leacky_relu(net_hidden)
            net_output = np.dot(w2_new, act) + b2_new
            output = self.sigmoid(net_output)

            # Compute Loss
            loss_new = self.compute_bce(output, y)
            
            # Backward propagation
            sigma_output = output - y
            delta_w2 = (sigma_output * act).reshape(1,self.hidden_sizes[0])
            delta_b2 = sigma_output
            g_output = np.hstack((delta_w2, delta_b2.reshape(1,1)))
            sigma_hidden = (sigma_output * w2_new) * self.leacky_relu_derivative(net_hidden)
            delta_w1 = sigma_hidden.T * X
            delta_b1 = sigma_hidden.reshape(-1)
            g_hidden = np.hstack((delta_w1, delta_b1.reshape(self.hidden_sizes[0],1)))
            g_total_new = np.concatenate([g_hidden.flatten(), g_output.flatten()])
            
            print(f"loss_new: {loss_new:.4f} vs RHS: {loss + c1 * alpha * np.dot(g_total, p_total):.4f} (delta: {c1 * alpha * np.dot(g_total, p_total):.4f}) (alpha: {alpha})")
            # Condizione Armijo
            if round(loss_new,4) > round(loss + c1 * alpha * np.dot(g_total, p_total),4):
                alpha_max = alpha
                print("entrato")
                continue
            
            print("gᵗp new =", np.dot(g_total_new, p_total))
            # Condizione curvatura
            if abs(np.dot(g_total_new, p_total)) < abs(c2 * np.dot(g_total, p_total)):
                if np.dot(g_total_new, p_total) > 0:
                    alpha_max = alpha
                else:
                    alpha_low = alpha
                continue
            
            print(f"🟢 Alpha: \n{alpha}")
            return alpha  # 🟢 Trovato uno buono, ritorna

        # 🔴 Nessuno trovato in T iterazioni: fallback "sicuro"
        print(f"🔴 Alpha: \n{alpha_init * (1 - (t/T))}")
        return  0.001
        


    # Metodo per stampare la struttura della rete
    def print_structure(self):
        print(f"Struttura della rete neurale:")
        print(f"Input size: {self.input_size}")
        print(f"Hidden layers sizes: {self.hidden_sizes}")
        print(f"Output size: {self.output_size}")
        
        # Pesi e bias di ogni layer
        print("\nPesi e bias:")
        print(f"Layer 1 (input -> hidden1):")
        print(f"Pesos (w1): \n{self.w1}")
        print(f"Bias (b1): \n{self.b1}")
        
        print(f"\nLayer 2 (hidden1 -> output):")
        print(f"Pesos (w2): \n{self.w2}")
        print(f"Bias (b2): \n{self.b2}")

        print(f"\nHessian Hidden: \n{self.hessian_hidden}")
        print(f"\nHessian Output: \n{self.hessian_output}")
'''  

import numpy as np

class NeuralNetworkBCEBFGS:
    def __init__(self, input_size, hidden_sizes, output_size, epochs):
        np.random.seed(45)
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes  # lista con i neuroni dei layer nascosti
        self.output_size = output_size
        self.epochs = epochs

        # Inizializzazione dei pesi e bias
        self.w1 = np.random.randn(hidden_sizes[0], input_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros(hidden_sizes[0])

        self.w2 = np.random.randn(output_size, hidden_sizes[0]) * np.sqrt(2.0 / hidden_sizes[0])
        self.b2 = np.zeros(output_size)

        # Per BFGS: inizializza la hessiana per ogni neurone
        self.hessian_hidden = np.array([np.eye(input_size+1) for _ in range(hidden_sizes[0])])
        self.hessian_output = np.array([np.eye(hidden_sizes[0]+1) for _ in range(output_size)])

        print(f"Pesi w1 iniziali (layer nascosto):\n{self.w1}")
        print(f"Bias b1 iniziali (layer nascosto):\n{self.b1}")
        print(f"Pesi w2 iniziali (output):\n{self.w2}")
        print(f"Bias b2 iniziali (output):\n{self.b2}")
    
    def train(self, X, y):
        loss_bce = []
        count = 1
        for epoch in range(self.epochs):
            for i in range(X.shape[0]):
                xi = X[i].reshape(1, -1)   # singolo esempio
                yi = y[i].reshape(1, -1)   # singola etichetta

                # === FORWARD ===
                _, z1, a1, y_pred = self.forward(xi)
                w_total = np.concatenate([
                    self.w1.flatten(),
                    self.b1.flatten(),
                    self.w2.flatten(),
                    self.b2.flatten()
                ])
                loss = self.compute_loss(y_pred, yi)
                loss_bce.append(loss)
                print(f"Epoch {epoch}, Sample {i+1}: Loss = {loss}")

                # === BACKWARD ===
                dw1, db1, dw2, db2 = self.backward(xi, yi, self.w2, z1, a1, y_pred)
                #dw1 += 2*0.1*self.w1
                #db1 += 2*0.1*self.b1
                #dw2 += 2*0.1*self.w2
                #db2 += 2*0.1*self.b2
                print(f"\nSample {i+1}:")
                print("dw1:", dw1)
                print("db1:", db1)
                print("dw2:", dw2)
                print("db2:", db2)
                
                # === DIREZIONE (p = -H @ g) ===
                p_hidden, p_output = self.compute_direction(dw1, db1, dw2, db2)
                print("p_w1 (direzioni layer nascosto):", p_hidden)
                print("p_w2 (direzione output):", p_output)

                # === LINE SEARCH (per ogni neurone) ===
                alphas_hidden = [self.line_search(xi, yi, self.w1[j], self.b1[j], p_hidden[j], layer='hidden', j=j) for j in range(self.hidden_sizes[0])]
                alpha_output = self.line_search(xi, yi, self.w2[0], self.b2[0], p_output[0], layer='output', j=0)
                print(f"[Epoch {epoch} Sample {i}] Alpha hidden: {alphas_hidden}, Alpha output: {alpha_output}")

                # === SALVATAGGIO PESI CORRENTI ===
                w1_pre = self.w1.copy()
                b1_pre = self.b1.copy()
                w2_pre = self.w2.copy()
                b2_pre = self.b2.copy()
                print(f"Pesi w1 precedenti (layer nascosto):\n{w1_pre}")
                print(f"Bias b1 precedenti (layer nascosto):\n{b1_pre}")
                print(f"Pesi w2 precedenti (output):\n{w2_pre}")
                print(f"Bias b2 precedenti (output):\n{b2_pre}")

                # === AGGIORNAMENTO PESI (per ogni neurone) ===
                self.update_weights(p_hidden, p_output, alphas_hidden, alpha_output)
                print(f"Pesi w1 aggiornati (layer nascosto):\n{self.w1}")
                print(f"Bias b1 aggiornati (layer nascosto):\n{self.b1}")
                print(f"Pesi w2 aggiornati (output):\n{self.w2}")
                print(f"Bias b2 aggiornati (output):\n{self.b2}")


                # === CALCOLO GRADIENTI AGGIORNATI ===
                _, z1, a1, y_pred = self.forward(xi)
                loss = self.compute_loss(y_pred, yi)
                print(f"Epoch {epoch}, Sample {i+1}: Loss new = {loss}")
                dw1_new, db1_new, dw2_new, db2_new = self.backward(xi, yi, self.w2, z1, a1, y_pred)
                print(f"\nSample {i+1}:")
                print("dw1 new:", dw1_new)
                print("db1 new:", db1_new)
                print("dw2 new:", dw2_new)
                print("db2 new:", db2_new)

                # === AGGIORNAMENTO H (per ogni neurone) ===
                self.update_hessian(w1_pre, b1_pre, w2_pre, b2_pre, dw1, db1, dw2, db2, dw1_new, db1_new, dw2_new, db2_new)
                print(f"\nHessian Hidden: \n{self.hessian_hidden}")
                print(f"\nHessian Output: \n{self.hessian_output}")

                if count == 10:
                    break

                count+=1

        return loss_bce
                


    def forward(self, X):
        # Layer 1: input -> hidden
        z1 = np.dot(X, self.w1.T) + self.b1  
        a1 = leaky_relu(z1)                 

        # Layer 2: hidden -> output
        z2 = np.dot(a1, self.w2.T) + self.b2  
        a2 = sigmoid(z2)                    

        return X, z1, a1, a2
    
    def compute_loss(self, y_pred, y_true, epsilon=1e-12):
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # evita log(0)
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss
    
    def backward(self, xi, yi, w2, z1, a1, y_pred):
        # Output layer
        dz2 = y_pred - yi  # derivative of BCE + sigmoid  (1, 1)

        dw2 = np.dot(dz2.T, a1)              # (1, 8)
        db2 = dz2.flatten()                  # (1,)

        # Hidden layer
        da1 = np.dot(dz2, w2)           # (1, 8)
        dz1 = da1 * leaky_relu_derivative(z1)  # (1, 8)

        dw1 = np.dot(dz1.T, xi)              # (8, 6)
        db1 = dz1.flatten()                  # (8,)

        return dw1, db1, dw2, db2
    
    def compute_direction(self, dw1, db1, dw2, db2):
        # Calcolo direzione BFGS per ogni neurone del layer 1
            p_w1 = []
            for j in range(self.hidden_sizes[0]):
                g = np.concatenate([dw1[j], [db1[j]]])  # (6+1,)
                H = self.hessian_hidden[j]
                p = -H @ g
                p_w1.append(p)
            p_w1 = np.array(p_w1)  # shape (8, 7)

            # Calcolo direzione BFGS per l’unico neurone dell’output
            p_w2 = []
            for j in range(self.output_size):
                g = np.concatenate([dw2[j], [db2[j]]])  # (8+1,)
                H = self.hessian_output[j]
                p = -H @ g
                p_w2.append(p)
            p_w2 = np.array(p_w2)  # shape (1, 9)

            return p_w1, p_w2
    
    def line_search(self, xi, yi, w, b, p, layer='hidden', j=0, c1=1e-4, c2=0.9, alpha_init=1.0):
        alpha = alpha_init
        max_iter = 20

        # Funzione per calcolo loss e gradiente locale
        def f_and_grad(wb):
            if layer == 'hidden':
                w_j = wb[:-1]
                b_j = wb[-1]
                z1 = np.dot(xi, self.w1.T) + self.b1
                z1[:, j] = np.dot(xi, w_j.T) + b_j
                a1 = leaky_relu(z1)
                z2 = np.dot(a1, self.w2.T) + self.b2
            else:  # output
                w_j = wb[:-1]
                b_j = wb[-1]
                z1 = np.dot(xi, self.w1.T) + self.b1
                a1 = leaky_relu(z1)
                z2 = np.dot(a1, w_j.T) + b_j

            y_pred = sigmoid(z2)
            loss = self.compute_loss(y_pred, yi)

            # backward pass per ottenere il gradiente locale
            if layer == 'hidden':
                dw1, db1, dw2, db2 = self.backward(xi, yi, self.w2, z1, a1, y_pred)
                grad = np.concatenate([dw1[j], [db1[j]]])
            else:
                dw1, db1, dw2, db2 = self.backward(xi, yi, w_j.reshape(1,-1), z1, a1, y_pred)
                grad = np.concatenate([dw2[j], [db2[j]]])
            
            return loss, grad

        # Stato attuale
        wb = np.concatenate([w, [b]])
        f0, g0 = f_and_grad(wb)
        phi0 = f0
        dphi0 = g0 @ p  # directional derivative

        for _ in range(max_iter):
            wb_new = wb + alpha * p
            f_new, g_new = f_and_grad(wb_new)
            phi = f_new
            dphi = g_new @ p
            #print(f"[Line search neuron {j}]: alpha = {alpha:.6f}, loss = {f_new:.6f}, grad·p = {np.dot(g0, p):.6f}, grad·p new = {np.dot(g_new, p):.6f}")

            if phi > phi0 + c1 * alpha * dphi0:
                alpha *= 0.5
            elif dphi < c2 * dphi0:
                alpha *= 1.1
            else:

                return alpha

        return 0.001
    
    def update_weights(self, p_h, p_o, a_h, a_o):
        for j in range(self.hidden_sizes[0]):
            self.w1[j] += a_h[j] * p_h[j][:-1]  # Aggiorna i pesi
            self.b1[j] += a_h[j] * p_h[j][-1]   # Aggiorna il bias
            self.w2[0] += a_o * p_o[0][:-1]  # Aggiorna i pesi
            self.b2[0] += a_o * p_o[0][-1]   # Aggiorna il bias
    
    def update_hessian(self, w1_pre, b1_pre, w2_pre, b2_pre, dw1, db1, dw2, db2, dw1_new, db1_new, dw2_new, db2_new):
        for j in range(self.hidden_sizes[0]):
            # s = theta_new - theta_old (w + b concatenati)
            theta_old = np.append(w1_pre[j], b1_pre[j])
            theta_new = np.append(self.w1[j], self.b1[j])
            s_vec = theta_new - theta_old
            
            # y = grad_new - grad_old (dw + db concatenati)
            grad_old = np.append(dw1[j], db1[j])
            grad_new = np.append(dw1_new[j], db1_new[j])
            y_vec = grad_new - grad_old
            
            # Hessian update
            H = self.hessian_hidden[j]
            rho = 1.0 / (y_vec @ s_vec + 1e-8)  # stabilizzazione con eps
            I = np.eye(H.shape[0])
            outer_sy = np.outer(s_vec, y_vec)
            outer_ys = np.outer(y_vec, s_vec)
            outer_ss = np.outer(s_vec, s_vec)

            self.hessian_hidden[j] = (I - rho * outer_sy) @ H @ (I - rho * outer_ys) + rho * outer_ss

            print(f"\n[Epoch {self.epochs}, Neurone {j}]")
            print(f" pesi precedenti: {theta_old}")
            print(f" pesi aggiornati: {theta_new}")
            print(f"s (delta theta): {s_vec}")
            print(f"y (delta grad): {y_vec}")
            print(f"rho: {rho:.6e}")

        # Output layer (solo un neurone)
        theta_old = np.append(w2_pre[0], b2_pre[0])
        theta_new = np.append(self.w2[0], self.b2[0])
        s_vec = theta_new - theta_old

        grad_old = np.append(dw2[0], db2[0])
        grad_new = np.append(dw2_new[0], db2_new[0])
        y_vec = grad_new - grad_old

        H = self.hessian_output[0]
        rho = 1.0 / (y_vec @ s_vec + 1e-8)
        I = np.eye(H.shape[0])
        outer_sy = np.outer(s_vec, y_vec)
        outer_ys = np.outer(y_vec, s_vec)
        outer_ss = np.outer(s_vec, s_vec)

        self.hessian_output[0] = (I - rho * outer_sy) @ H @ (I - rho * outer_ys) + rho * outer_ss

        print(f"\n[Epoch {self.epochs}, Neurone output]")
        print(f" pesi precedenti: {theta_old}")
        print(f" pesi aggiornati: {theta_new}")
        print(f"s (delta theta): {s_vec}")
        print(f"y (delta grad): {y_vec}")
        print(f"rho: {rho:.6e}")


def leaky_relu(z, alpha=0.01):
    return np.where(z > 0, z, alpha * z)

def leaky_relu_derivative(z, alpha=0.01):
    return np.where(z > 0, 1, alpha)

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

    
    