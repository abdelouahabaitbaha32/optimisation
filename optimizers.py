

import numpy as np

class GradientDescent:
    def __init__(self, lr=0.01):
        self.lr = lr
    def update(self, theta, grad):
        return theta - self.lr * grad

class GDLineSearch:
    def __init__(self, lr_init=0.01):
        self.lr_init = lr_init
    def update(self, theta, grad, loss_func=None, current_loss=None):
        # بحث بسيط عن الخطوة أو استخدام ثابت كمبدأ أولئي
        return theta - self.lr_init * grad

class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr
    def update(self, theta, grad):
        return theta - self.lr * grad

class Momentum:
    def __init__(self, lr=0.01, mu=0.9):
        self.lr = lr
        self.mu = mu
        self.v = None
    def update(self, theta, grad):
        if self.v is None:
            self.v = np.zeros_like(theta)
        self.v = self.mu * self.v + grad
        return theta - self.lr * self.v

class AdaGrad:
    def __init__(self, lr=0.01, eps=1e-8):
        self.lr = lr
        self.eps = eps
        self.s = None
    def update(self, theta, grad):
        if self.s is None:
            self.s = np.zeros_like(theta)
        self.s += grad ** 2
        return theta - self.lr * grad / (np.sqrt(self.s) + self.eps)

class RMSprop:
    def __init__(self, lr=0.01, rho=0.9, eps=1e-8):
        self.lr = lr
        self.rho = rho
        self.eps = eps
        self.s = None
    def update(self, theta, grad):
        if self.s is None:
            self.s = np.zeros_like(theta)
        self.s = self.rho * self.s + (1 - self.rho) * (grad ** 2)
        return theta - self.lr * grad / (np.sqrt(self.s) + self.eps)

class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None
        self.s = None
        self.t = 0
    def update(self, theta, grad):
        if self.m is None:
            self.m = np.zeros_like(theta)
            self.s = np.zeros_like(theta)
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.s = self.beta2 * self.s + (1 - self.beta2) * (grad ** 2)
        
        m_hat = self.m / (1 - self.beta1 ** self.t)
        s_hat = self.s / (1 - self.beta2 ** self.t)
        
        return theta - self.lr * m_hat / (np.sqrt(s_hat) + self.eps)