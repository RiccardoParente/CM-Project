import time
import numpy as np
import csv
import random
import math
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


def load_dataset_from_assets(file_path):
    x_list = []
    y_list = []
    try:
        with open(file_path, mode='r') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip header
            for row in reader:
                x_row = [float(value) for value in row[:-1]]
                y_value = int(row[-1])
                x_list.append(x_row)
                y_list.append(y_value)
    except Exception as e:
        print(f"Error loading dataset: {e}")

    x = np.array(x_list)
    y = np.array(y_list)
    return x, y


# Funzione per normalizzare i dati
def normalize(data):
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(data)
    return normalized_data


# Funzione per calcolare l'accuratezza
def accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)


# Classe NeuralNetwork
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        np.random.seed(60)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.w1 = np.random.randn(hidden_size, input_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros(hidden_size)
        self.w2 = np.random.randn(hidden_size, hidden_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros(hidden_size)
        self.w3 = np.random.randn(hidden_size, hidden_size) * np.sqrt(2.0 / hidden_size)
        self.b3 = np.zeros(hidden_size)
        self.w4 = np.random.randn(output_size, hidden_size) * np.sqrt(2.0 / hidden_size)
        self.b4 = np.zeros(output_size)

    def train(self, X, y, learning_rate):
        loss = np.zeros(5)
        count = 0
        atot1 = np.zeros((5, self.hidden_size))
        atot2 = np.zeros((5, self.hidden_size))
        atot3 = np.zeros((5, self.hidden_size))
        output = np.zeros(5)
        input_data = np.zeros((5, X.shape[1]))

        for i in range(X.shape[0]):
            # Forward propagation
            input_data[count] = X[i]

            a1 = self.relu(np.dot(self.w1, X[i][:, np.newaxis]).T + self.b1)
            atot1[count] = a1

            a2 = self.relu(np.dot(a1, self.w2) + self.b2)
            atot2[count] = a2

            a3 = self.relu(np.dot(a2, self.w3) + self.b3)
            atot3[count] = a3

            z1 = self.sigmoid(np.dot(self.w4, a3.T) + self.b4)[0]
            output[count] = z1

            loss[count] = abs(y[i] - z1)

            count += 1

            if count == 5:

                # Backpropagation
                dZ4 = loss * self.sigmoid_derivate(output)
                dW4 = self.GradientProductOutput(dZ4, atot3)
                db4 = np.sum(dZ4)

                dA3 = self.error_hidden(self.w4, dZ4)
                dZ3 = dA3 * self.relu_derivative(atot3)
                dW3 = self.GradientProductHidden(dZ3, atot2)
                db3 = dZ3

                dA2 = self.error_hidden_matrix(self.w3, dZ3)
                dZ2 = dA2 * self.relu_derivative(atot2)
                dW2 = self.GradientProductHidden(dZ2, atot1)
                db2 = dZ2

                dA1 = self.error_hidden_matrix(self.w2, dZ2)
                dZ1 = dA1 * self.relu_derivative(atot1)
                dW1 = self.GradientProductInput(dZ1, input_data)
                db1 = dZ1

                # Aggiornamento pesi
                self.w4 -= learning_rate * dW4
                self.w3 -= learning_rate * dW3
                self.w2 -= learning_rate * dW2
                self.w1 -= learning_rate * dW1

                self.b4[0] -= learning_rate * db4
                sumdb3 = np.sum(db3, axis=0)
                self.b3 -= learning_rate * sumdb3
                sumdb2 = np.sum(db2, axis=0)
                self.b2 -= learning_rate * sumdb2
                sumdb1 = np.sum(db1, axis=0)
                self.b1 -= learning_rate * sumdb1

                count = 0

    def predict(self, X):
        y = np.zeros(X.shape[0], dtype=int)
        for i in range(X.shape[0]):
            a1 = self.relu(np.dot(self.w1, X[i][:, np.newaxis]).T + self.b1)
            a2 = self.relu(np.dot(a1, self.w2) + self.b2)
            a3 = self.relu(np.dot(a2, self.w3) + self.b3)
            z1 = self.sigmoid(np.dot(self.w4, a3.T) + self.b4)[0]

            if z1 >= 0.5:
                y[i] = 1
        return y

    def relu(self, z):
        alpha = 0.01
        return np.maximum(alpha * z, z)

    def relu_derivative(self, a):
        alpha = 0.01
        return np.where(a > 0, 1, alpha)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivate(selfself, z):
        return z * (1 - z)

    def GradientProductInput(self, a, b):
        return np.dot(a.T, b)

    def GradientProductHidden(self, a, b):
        return np.dot(a.T, b)

    def GradientProductOutput(self, a, b):
        return np.dot(a, b).reshape(1, -1)

    def error_hidden(self, a, b):
        return b[:, np.newaxis] * a

    def error_hidden_matrix(self, a, b):
        return np.dot(b, a.T)


def main(file_path, downloads_path):

    # Definisce i parametri della rete neurale
    input_size = 5
    hidden_size = 333
    output_size = 1

    data = []

    for i in range(1,101):

        print(f"---------------------- {i} -------------------------")

        # Caricamento del dataset
        start_time = time.time()
        X, y = load_dataset_from_assets(file_path)
        end_time = time.time()
        durationLoad = round((end_time - start_time) * 1000)  # Converti in millisecondi
        print(f"Durata loading Dataset: {durationLoad} ms")

        # Normalizza il dataset
        start_time = time.time()
        normalized_X = normalize(X)
        end_time = time.time()
        durationNormalized = round((end_time - start_time) * 1000)  # Converti in millisecondi
        print(f"Durata normalized data: {durationNormalized} ms")

        # Creazione della rete neurale
        start_time = time.time()
        nn = NeuralNetwork(input_size, hidden_size, output_size)
        end_time = time.time()
        durationCreate = round((end_time - start_time) * 1000)  # Converti in millisecondi
        print(f"Durata creation NeuralNetwork: {durationCreate} ms")

        # Addestramento della rete neurale
        start_time = time.time()
        nn.train(normalized_X, y, learning_rate=0.00001)
        end_time = time.time()
        durationTrain = round((end_time - start_time) * 1000) # Converti in millisecondi
        print(f"Durata training NeuralNetwork: {durationTrain} ms")

        # Esempio di predizione
        start_time = time.time()
        prediction_train = nn.predict(normalized_X)
        # Calcolo dell'accuratezza
        training_accuracy = accuracy(y, prediction_train)
        end_time = time.time()
        durationPredict = round((end_time - start_time) * 1000)  # Converti in millisecondi
        print(f"Durata predict NeuralNetwork: {durationPredict} ms")
        print(f"Training Accuracy: {training_accuracy}")

        data.append([str(durationLoad), str(durationNormalized), str(durationCreate), str(durationTrain), str(durationPredict)])

    start_time_final = time.time()
    csv_file_path = os.path.join(downloads_path, 'outputPython.csv')
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        for row in data:
            writer.writerow(row)
    end_time_final = time.time()
    duration_write_csv = round((end_time_final - start_time_final) * 1000)  # Converti in millisecondi
    print(f"Durata Write CSV: {duration_write_csv:.2f} ms")


if __name__ == "__main__":
 main()
