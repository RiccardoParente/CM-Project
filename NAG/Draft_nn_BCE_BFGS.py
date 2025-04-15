import numpy as np

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
        T = len(X)
        t = 1
        prev_loss = None
        patience = 8
        patience_counter = 0
        tolerance = 1e-3
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
                loss = self.compute_bce(output, y[i])
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
                sigma_output = y[i] - output
                print(f"\n sigma_output: \n{sigma_output}") 
                delta_w2 = (sigma_output * act).reshape(1,self.hidden_sizes[0])
                delta_b2 = sigma_output
                g_output1 = np.hstack((delta_w2, delta_b2.reshape(1,1)))
                
                sigma_hidden = (sigma_output * self.w2) * self.leacky_relu_derivative(net_hidden)
                delta_w1 = sigma_hidden.T * X[i]
                delta_b1 = sigma_hidden.reshape(-1)
                g_hidden1 = np.hstack((delta_w1, delta_b1.reshape(self.hidden_sizes[0],1)))
                
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
                sigma_output = y[i] - output
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
    

    def wolfe_line_search(self, t, T, X, y, loss, w1, b1, w2, b2, p1, p2, g1, g2, c1=1e-4, c2=0.9, alpha_init =0.1):
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
            sigma_output = y - output
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
        return  alpha_init * (1 - (t/T)) 
        


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
        