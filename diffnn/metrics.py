import torch
from torch.autograd import functional
import numpy as np

class RiemmanianMetric:
    
    def __init__(self, dim=2):
        self.dim = dim
        
    def metric(self, x):
        """
        the covariant metric tensor of shape [dim, dim], must me implemented in concrete classes
        """
        raise NotImplementedError()
        
    def cometric(self, x):
        """
        the contravariant metric tensor (or the inverse)
        """
        return torch.linalg.pinv(self.metric(x))
        
    def metric_derivative(self, x):
        """
        the derivative of the metric tensor with respect to each coordinate
        """
        return functional.jacobian(self.metric, x, create_graph=True, vectorize=True) 
    
    def christoffels(self, x):
        """
        the christoffel symbols at x
        """
        cometric = self.cometric(x)
        metric_derivative = self.metric_derivative(x)

        term_1 = torch.einsum(
            "...ba,...amv->...bmv", cometric, metric_derivative
        )
        term_2 = torch.einsum(
            "...ba,...avm->...bmv", cometric, metric_derivative
        )
        term_3 = torch.einsum(
            "...ba,...mva->...bmv", cometric, metric_derivative
        )

        r = 0.5 * (term_1 + term_2 - term_3)
        return r
    
