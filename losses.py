import numpy as np

class MSE:
    """ Mean Squared Error loss """

    def __init__(self):
        self.name = "Mean Squared Error"

    def compute(self, outputs, targets):
        return np.mean(np.power(targets - outputs, 2))

    def derivative(self, outputs, targets):
        return -2 * (targets - outputs)


class BCE:
    """ Binary Cross Entropy loss """

    def __init__(self):
        self.name = "Binary Cross Entropy"

    def compute(self, outputs, targets):
        outputs_clipped = np.clip(outputs, 1e-15, 1-1e-15)
        return np.mean(-(1 - targets) * np.log(1 - outputs_clipped) - targets * np.log(outputs_clipped))

    def derivative(self, outputs, targets):
        outputs_clipped = np.clip(outputs, 1e-15, 1-1e-15)
        return -np.mean(targets/outputs_clipped - (1-targets)/(1-outputs_clipped))
