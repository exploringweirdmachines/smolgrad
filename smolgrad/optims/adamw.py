from typing import *

from ._optimizer import Optimizer
from ..core import Tensor

class AdamW(Optimizer):
    """
    Implements AdamW optimizer, which decouples weight decay from the gradient updates.
    
    AdamW performs weight decay at the parameter level instead of in the gradient, 
    which helps avoid issues with adaptive gradient methods and L2 regularization.

    Args:
        parameters (List[Tensor]): Parameters to optimize
        lr (float): Learning rate
        weight_decay (float): Weight decay coefficient (default: 0.01)
        beta1 (float): Exponential decay rate for first moment estimates (default: 0.9)
        beta2 (float): Exponential decay rate for second moment estimates (default: 0.999)
        eps (float): Term added to denominator to improve numerical stability (default: 1e-8)
    """
    def __init__(
            self, parameters: List[Tensor], 
            lr: float, weight_decay: float = 0.01,
            beta1: float = 0.9, beta2: float = 0.999,
            eps: float = 1e-8
        ) -> None:
        super().__init__(parameters, lr)

        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        # Initialize momentum and velocity terms for each parameter
        self.mo1 = [self._d.zeros_like(p.data) for p in self.parameters]
        self.mo2 = [self._d.zeros_like(p.data) for p in self.parameters]
        
        # Initialize timestep for bias correction
        self.timestep = 0

    def step(self):
        """
        Performs a single optimization step.
        
        1. Apply weight decay directly to parameters
        2. Update momentum terms using gradients
        3. Apply bias correction
        4. Update parameters using Adam style updates
        """
        self.timestep += 1
        
        for i in range(len(self.parameters)):
            p, m, v = self.parameters[i], self.mo1[i], self.mo2[i]

            # Weight decay term applied directly to parameters
            if self.weight_decay != 0:
                p.data *= (1 - self.lr * self.weight_decay)

            # Update momentum terms
            m[:] = (self.beta1 * m) + ((1 - self.beta1) * p.grad)
            v[:] = (self.beta2 * v) + ((1 - self.beta2) * p.grad**2)

            # Bias correction
            m_hat = m / (1 - self.beta1 ** self.timestep)
            v_hat = v / (1 - self.beta2 ** self.timestep)

            # Update parameters using Adam style updates
            p.data -= self.lr * (m_hat / (self._d.sqrt(v_hat) + self.eps))
