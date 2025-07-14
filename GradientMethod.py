from abc import ABC, abstractmethod
import numpy as np
from StepSizeMethods import *
from scipy.optimize import minimize


LINE_SEARCH_MAX_ITERATIONS = 10**6

class GradientMethod(ABC):
    def __init__(self, step_size_method, tolerance, max_steps):
        self.step_size_method = step_size_method
        self.tolerance = tolerance
        self.max_steps = max_steps

        self.completion_steps = None
        self.optimized_x_value = None
        self.optimized_f_value = None
        self.completion_time = None

        self.name = None

    '''
    metodo del gradiente

        Dato x0 ∈ Rn e k = 0
        while ∇f(xk) ̸= 0 do
            scelgo dk = −∇f(xk)
            calcolo αk con Armijo
            xk+1 = xk + αkdk
            k = k + 1
        end while
    '''
    @abstractmethod
    def optimize(self, problem):
        pass

    def save_info(self, step, x, f):
        self.completion_steps = step
        self.optimized_x_value = x
        self.optimized_f_value = f

    def reset(self):
        self.step_size_method.reset()

    def infinity_norm(self, vector):
        return np.max(np.abs(vector))
    
    def get_name(self):
        return self.name
    
    def get_step_method(self):
        return self.step_size_method
    
    def get_iterations(self):
        return self.completion_steps
    
    def get_final_value(self):
        return self.optimized_f_value
    
    def print_found_solution(self, x, f, step):
        print(f"Found minimum x = {x} value = {f} after {step} iterations")

    

class Lbfgs(GradientMethod):
    def __init__(self, step_size_method, tolerance, max_steps):
        super().__init__(step_size_method, tolerance, max_steps)
        self.name = "BFGS"
        self.problem = None

    def optimize(self, problem):
        self.problem = problem
        x0 = problem.x0
        result = minimize(self.get_f_g, x0, method='L-BFGS-B', jac=True, options={'maxiter': self.max_steps, "gtol": self.tolerance})
        self.save_info(result.nit, result.x, result.fun)
        self.print_found_solution(result.x, result.fun, result.nit)

    def get_f_g(self, x):
        return self.problem.obj(x, gradient=True)


class GradientMethodArmijo(GradientMethod):
    def __init__(self, step_size_method, tolerance, max_steps, gamma=1e-4, sigma=0.5):
        super().__init__(step_size_method, tolerance, max_steps)

        self.gamma = gamma
        self.sigma = sigma
        self.name = "Standard Armijo"
    
    def optimize(self, problem):
        x_k = np.array(problem.x0)
        f_xk, gradient_xk = problem.obj(x_k, gradient=True)

        self.step_size_method.save(x_k, gradient_xk)

        step = 0

        while np.linalg.norm(gradient_xk) > self.tolerance and step < self.max_steps:
            d_k = -gradient_xk

            alpha_k = self.armijo_method(problem, x_k, d_k, step)

            x_k = x_k + alpha_k * d_k
            f_xk, gradient_xk = problem.obj(x_k, gradient=True)

            self.step_size_method.save(x_k, gradient_xk)

            step += 1

        self.save_info(step, x_k, f_xk)

        self.print_found_solution(x_k, f_xk, step)

        return x_k

    def armijo_method(self, problem, x_k, d_k, step):
        alpha = self.step_size_method.get_initial_alpha(step)

        f_xk_value, gradient_xk = problem.obj(x_k, gradient=True)

        while problem.obj(x_k + alpha * d_k) > f_xk_value + self.gamma * alpha * np.dot(gradient_xk, d_k):
            alpha = alpha * self.sigma

        return alpha
    

class GradientMethodGrippo(GradientMethod):
    def __init__(self, step_size_method, tolerance, max_steps, M, gamma=1e-4, sigma=0.5):
        super().__init__(step_size_method, tolerance, max_steps)

        self.M = M
        self.gamma = gamma
        self.sigma = sigma
        self.m = []
        self.f = []
        self.name = f"Grippo (M={M})"


    def optimize(self, problem):
        x_k = problem.x0
        f_xk, gradient_xk = problem.obj(x_k, gradient=True)

        self.step_size_method.save(x_k, gradient_xk)

        self.m = [0]
        self.f = [f_xk]

        step = 0

        while np.linalg.norm(gradient_xk) > self.tolerance and step < self.max_steps:
            d_k = -gradient_xk

            alpha_k = self.grippo_method(problem, step, x_k, d_k, gradient_xk)

            x_k += d_k * alpha_k
            f_xk, gradient_xk = problem.obj(x_k, gradient=True)

            self.step_size_method.save(x_k, gradient_xk)

            self.m.append(min([self.m[step]+1, self.M]))
            self.f.append(f_xk)

            step += 1

        self.save_info(step, x_k, f_xk)
        
        self.print_found_solution(x_k, f_xk, step)

        return x_k
    

    def grippo_method(self, problem, step, x_k, d_k, g_k):
        alpha = self.step_size_method.get_initial_alpha(step)
            
        while problem.obj(x_k+alpha*d_k) > self.get_condition_value(step, alpha, g_k, d_k):
            alpha *= self.sigma

        return alpha
    

    def get_condition_value(self, step, alpha, g_k, d_k):
        gk_dk = np.matmul(np.transpose(g_k[:, np.newaxis]), d_k)[0]

        max_value = self.f[step] + self.gamma * alpha * gk_dk
        
        for j in range(0, self.m[step]):
            actual_value = self.f[step-j] + self.gamma * alpha * gk_dk
            if actual_value > max_value:
                max_value = actual_value
        
        return max_value


class GradientMethodNLSA(GradientMethod):
    def __init__(self, step_size_method, tolerance, max_steps, eta_min=0.85, eta_max=0.85, delta=1e-4, rho=0.5):
        super().__init__(step_size_method, tolerance, max_steps)
        
        self.eta_min = eta_min
        self.eta_max = eta_max
        self.delta = delta
        self.rho = rho
        self.name = "NLSA"

    def optimize(self, problem):
        x_k = problem.x0
        f_xk, gradient_xk = problem.obj(x_k, gradient=True)

        self.step_size_method.save(x_k, gradient_xk)

        C_k = f_xk
        Q_k = 0

        step = 0

        while np.linalg.norm(gradient_xk) > self.tolerance and step < self.max_steps:
            d_k = -gradient_xk

            alpha_k = self.NLSA_method(problem, x_k, d_k, C_k, gradient_xk, step)

            x_k += d_k * alpha_k
            f_xk, gradient_xk = problem.obj(x_k, gradient=True)

            self.step_size_method.save(x_k, gradient_xk)

            eta_k = np.random.uniform(self.eta_min, self.eta_max)

            Q_k1 = eta_k * Q_k + 1
            C_k1 = (eta_k * Q_k * C_k + f_xk) / Q_k1
            
            Q_k = Q_k1
            C_k = C_k1

            step += 1

        self.save_info(step, x_k, f_xk)
        
        self.print_found_solution(x_k, f_xk, step)

        return x_k
    

    def NLSA_method(self, problem, x_k, d_k, C_k, g_k, step):
        alpha = self.step_size_method.get_initial_alpha(step)

        while not self.armijo_condition(problem, x_k, alpha, d_k, C_k, g_k):
            alpha *= self.rho

        return alpha

    def armijo_condition(self, problem, x_k, alpha_k, d_k, C_k, g_k):
        first_term = problem.obj(x_k + alpha_k*d_k)
        second_term = C_k + self.delta * alpha_k * np.matmul(np.transpose(g_k[:, np.newaxis]), d_k)[0]

        return first_term <= second_term