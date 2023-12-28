import torch
from torch.autograd import functional
from torch import vmap
from torch import nn
import numpy as np
from time import time

def create_symetric_matrices(z):
    """
    z is a tensor (m,k,..) with m elements and k components per element
      where k is the number of distinct elements in a symmetric matrix
      such as 3 (for a 2x2 matrix), 6 (for a 3x3 matrix), 10 (for a 4x4 matrix)
      and so on
      
    returns: a tensor (m,n,n,...) where n is the shape of the corresponding square
             symmetric matrix to k
    
    """
    
    # number of distinct elements to be placed in the symmetric matrix
    k = z.shape[1]
    
    # dimension of the symmetric matrix
    n = int((np.sqrt(1+8*k)-1)/2)
    
    # check
    _k = n*(n+1)/2
    if _k!=k:
        raise ValueError(f"invalid number of distinct elements for a symmetric matrix, {k}")
        
    i,j = torch.triu_indices(n,n)

    edims = [n,n]+list(z.shape[2:])
    r = torch.zeros(len(z),*edims)
    r[:,i,j,...] = z
    r.transpose(1,2)[:,i,j,...] = z

    return r

class RiemmanianMetric(nn.Module):
    """
    a torch module which, given an input [batch_size, dim] produces a metric tensor [batch_size, dim, dim]
    """

    def __init__(self, dim=2):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return self.metric(x)

    def metric(self, x):
        """
        the covariant metric tensor, must me implemented in concrete classes.
        it must accept either a single element [dim], or a batch of elements [batch_size, dim].
        it must return a structure [dim, dim] for a single element, or [batch_size, dim, dim] otherwise.
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
        f = lambda x: functional.jacobian(self.metric, x, create_graph=True)
        if len(x.shape)==1:
            # single element
            return f(x) 
        elif len(x.shape)==2:
            # batch. not using vmap since if metric is a nn, already is vectorized and
            # this confuses vmap
            return torch.stack([f(xi) for xi in x])
        else:
            raise ValueError(f"x must be 1D (a single element) or 2D (a batch of elements), but found shape {x.shape}")
            
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
    

class PolynomialRiemmanianMetric(RiemmanianMetric):
    
    """
    a metric tensor parametrized by a polynomial function on the input elemnts.
    
    $g_{ij}$
    
    $g_{ij}=\sum_{d,p}\\underset{ijdp}{C}(x^d)^p$

    
    where $g_{ij}$ is the produced metric tensor which, since it is symetric it contains k = dim*(dim+1)/2 
    distinct elements. $d$ runs over the number of dimensions and $p$ runs over the polynomial degree.
    
    thus, there are k * dim * (degree + 1) parameters.
    """

    def __init__(self, degree, dim):
        super().__init__()
        self.degree = degree
        self.dim    = dim
        
        # number of distinct elements of a symmetric matrix of size n x n
        self.dim_ = dim*(dim+1)//2 
        
        self.C = nn.Parameter(
            torch.rand(size=(self.dim_, self.dim, self.degree+1))
        )
        
    def add_powers(self, x):
        return torch.stack([x**pi for pi in range(self.degree+1)]).transpose(0,1).transpose(1,2)

    def metric(self, x):
        """
        returns a metric tensor for each input element
        """
        is_single = False
        if len(x.shape)==1:
            is_single = True
            x = x.reshape(1,-1)
            
        # add power items of x
        xp = self.add_powers(x)
        
        # build the polynomials for the metric tensor
        r = torch.einsum("anp,mnp->ma", self.C, xp)
        
        # recreate symmetric metric tensors, since we only deal
        # with matrix elements that are distinct
        r = create_symetric_matrices(r)
        
        if is_single:
            return r[0]
        return r

    def metric_derivative(self, x):
        """
        returns the derivatives of the metric tensor with respect to each coordinate.
        we exploit the fact that the metric tensor is expressed as polynomials to
        compute the matrix of metric derivatives much faster.
        """        
        # add power items of x
        xp = self.add_powers(x)
        
        # for derivatives we lower the powers and shift the coeficients
        dC  = self.C[:,:,1:]
        dxp = xp[:,:,:-1]
        dp  = torch.arange(1,self.degree+1)*1.

        # and add one dimension to the output
        r = torch.einsum("anp,mnp,p->man", dC, dxp, dp)
        
        # recreate symmetry from metric tensors
        r = create_symetric_matrices(r)
        return r
    
    @classmethod
    def test(cls):
        print ("comparing metric derivatives analytical vs. torch.jacobian")
        for _ in range(10):
            degree = np.random.randint(5)+1   # polynomial degree
            n = np.random.randint(10)+2        # manifold dimension
            m = np.random.randint(1000)+100

            print (f". degree {degree}, dim {n}, data size {m} :: ", end="", flush=True)
            
            x = torch.rand(size=(m,n))
            
            pm = cls(degree=degree, dim=x.shape[-1])
            
            t1 = time()
            dm1 = pm.metric_derivative(x)
            t2 = time()
            dm2 = super(cls, pm).metric_derivative(x)
            t3 = time()
            if not torch.allclose(dm1, dm2):
                raise ValueError("test failed")

            print(f"result correct :: time analytical {t2-t1:.5f} secs, time autograd {t3-t2:.5f}, speedup x{(t3-t2)/(t2-t1):.2f}") 


class ParametrizedRiemmanianMetric(RiemmanianMetric):
    """
    a metric tensor parametrized by a torch nn.Module
    """
    def __init__(self, nn, dim):
        """
        nn: a torch nn module
        the nn input are points in a certain manifold [batch_size, dim]
        the nn output are the metric tensors evaluated at those points [batch_size, dim, dim]
        """
        super().__init__(dim)
        self.nn = nn
        
    def metric(self, x):
        """
        the metric tensor computed by the network
        """
        return self.nn(x)
            
    def metric_derivative(self, x):
        """
        the derivative of the metric tensor with respect to each coordinate.
        assumes x is [batch_size, dim]
        """
        # batch. not using vmap since if metric is a nn, already is vectorized and
        # this confuses vmap
        f = lambda x: functional.jacobian(self.metric, x, create_graph=True)
        return torch.stack([f(xi.reshape(1,-1))[0] for xi in x])[:,:,:,0,:]
