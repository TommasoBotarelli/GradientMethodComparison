from abc import ABC, abstractmethod
import numpy as np
import copy

class Stack:
    def __init__(self, size):
        self.values = [None for _ in range(size)]

    def add(self, value):
        for i in range(len(self.values)-1, 0, -1):
            self.values[i] = copy.deepcopy(self.values[i-1])
        self.values[0] = copy.deepcopy(value)

    def get(self, index):
        return self.values[index]
    

class StepSizeMethod(ABC):
    name = None

    @abstractmethod
    def save(self, xk, gk):
        pass
    
    @abstractmethod
    def get_initial_alpha(self, step):
        pass

    @abstractmethod
    def reset(self):
        pass

    def get_name(self):
        return self.name
    
class NoStepSize(StepSizeMethod):
    def save(self, xk, gk):
        pass
    
    def get_initial_alpha(self, step):
        pass

    def reset(self):
        pass

    def get_name(self):
        return "No Particular Choice"

class BarzilaiBorwein(StepSizeMethod):
    # alpha = 1 -> bb1
    # alpha = 2 -> bb2
    def __init__(self, alpha):
        self.x_saved = Stack(2)
        self.g_saved = Stack(2)

        self.name = f"Barzilai Borwein (bb{alpha})"
        self.chosen_alpha = alpha

    def reset(self):
        self.x_saved = Stack(2)
        self.g_saved = Stack(2)

    def save(self, xk, gk):
        self.x_saved.add(xk)
        self.g_saved.add(gk)

    def get_initial_alpha(self, step):
        alpha = 1

        if step != 0:
            delta_x = np.subtract(self.x_saved.get(0), self.x_saved.get(1))
            delta_g = np.subtract(self.g_saved.get(0), self.g_saved.get(1))

            delta_x_delta_g = abs(np.dot(delta_x, delta_g))
            delta_g_delta_g = np.dot(delta_g, delta_g)
            delta_x_delta_x = np.dot(delta_x, delta_x)

            if delta_x_delta_g != 0 and delta_x_delta_x != 0 and self.chosen_alpha == 1:
                alpha = delta_x_delta_x / delta_x_delta_g
            if delta_x_delta_g != 0 and delta_g_delta_g != 0 and self.chosen_alpha == 2:
                alpha = delta_x_delta_g / delta_g_delta_g

        # Check if alpha is infinite or NaN
        if not np.isfinite(alpha) or np.isnan(alpha):
            alpha = 1

        return alpha
    

class ConstantStep(StepSizeMethod):
    def __init__(self, value):
        self.alpha = value
        
        self.name = "ConstantStep"

    def reset(self):
        pass

    def save(self, xk, gk):
        pass
    
    def get_initial_alpha(self, step):
        return self.alpha