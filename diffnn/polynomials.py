import torch
from torch import nn
import numpy as np

def sum_cross_diagonals(x):
    from itertools import product
    r = torch.tensor([i+j for i,j in product(range(x.shape[0]), range(x.shape[1]))]).reshape(x.shape)
    r = torch.stack([(x*(r==i)).sum() for i in range(r.max()+1)])
    return r

class PolynomialLayer(nn.Module):
    
    def __init__(self, degree, input_dim, init_C=None):
        super().__init__()
        self.degree = degree
        self.input_dim = input_dim
        
        if init_C is not None and (init_C.shape[0]!=input_dim or init_C.shape[1]!=degree+1):
            raise ValueError(f"incorrect shape for 'init_C' {init_C.shape}, expected {input_dim, degree+1}")
        
        self.C = nn.Parameter(
            torch.rand(size=(self.input_dim, self.degree+1)) if init_C is None else init_C
        )

    def forward(self, x):
        x = torch.stack([x**i for i in range(self.degree+1)])
        x = torch.einsum("abc,ca->b", x, self.C)
        return x
    
    def derivative(self, wrt=None):
        """
        for a derivative the degree is lowered and the coefficients adjusted
        
        wrt: the input dimension with respect to which the derivative is taken
             None can be used only when input_dim=1
             
        returns the derivative polynomial with respect to the selected input_dim
        """
        if wrt is None and self.input_dim != 1:
            raise ValueError("can only use wrt=None when input_dim is 1")

        if wrt is None:
            wrt = 0

        if wrt > self.input_dim-1:
            raise ValueError(f"this pol has only {self.input_dim} dimensions (numbered from 0 to {self.input_dim-1}) but you are requestive a derivative wrt dimension {wrt}")
        
        newC = torch.zeros(self.C.shape[0], self.C.shape[1]-1)
        if self.degree==0:
            newPol = self.__class__(degree=self.degree, input_dim = self.input_dim, init_C = self.C*0.)

        else:
            powers = torch.arange(1, self.degree+1).type(torch.float)                    
            newC[wrt] = self.C[wrt,1:]*powers
            newPol = self.__class__(degree=self.degree-1, input_dim = self.input_dim, init_C = newC)
            
        return newPol
        
    def jacobian(self):
        """
        builds all the derivatives with respect to all the input parameters
        """
        powers = torch.arange(1, self.degree+1).type(torch.float)
        C = self.C[:,1:]*powers
        r = PolynomialLayerWithIndependantInputs(input_dim=C.shape[0], degree=C.shape[1]-1, init_C=C)
        return r

    def integral(self):
        """
        for an integral the degree is raised and the coefficients adjusted
        the integration constant is set to zero.

        integrals are only allowed for polinomials of input_dim=1, since otherwise
        they would create mixed terms multiplying different input dims
        """

        if self.input_dim!=1:
            raise ValueError(f"only integrals of polynomials with input_dim=1 are allowed. found input_dim={self.input_dim}")

        powers = 1/torch.arange(1, self.degree+2).type(torch.float)
        C = torch.cat([torch.zeros([1,self.input_dim]), (self.C*powers).T]).T         
        
        r = self.__class__(degree=self.degree+1, input_dim = self.input_dim, init_C = C)
        return r
    
    def __pow__(self, p):       
        if p<0:
            raise ValueError(f"invalid power {p}, must be >= 1")
        elif p==0:
            # for zero a single coeficient = 1
            return self.__class__(degree=0, input_dim=self.input_dim, init_C=torch.ones(self.C[:,:1].shape)*1.)
        elif p==1:
            return self
        else:
            return self*pow(self, p-1)
                
    
    def __add__(self, other):
        if self.input_dim != other.input_dim:
            raise ValueError("can only add polinomials with the same 'input_dim'")
        
        selfC = self.C
        otherC = other.C
        
        degree = np.max([self.degree, other.degree])

        if self.degree>other.degree:
            otherC = torch.cat([otherC.T, torch.zeros([self.degree-other.degree,self.input_dim])]).T         

        if self.degree<other.degree:
            selfC = torch.cat([selfC.T, torch.zeros([other.degree-self.degree,self.input_dim])]).T         

        C = selfC + otherC
        
        r = self.__class__(degree=degree, input_dim=self.input_dim, init_C = C)
        return r
    
    def __mul__(self, other):

        # detect if 'other' is a constant
        isconstant = False
        if isinstance(other, torch.Tensor) and len(other.shape)==0:
            # we are ok
            isconstant = True
        elif isinstance(other, torch.Tensor) and len(other.shape)!=0:
            raise ValueError("constant must be a dimensionless tensor")    
        else:
            try:
                other = torch.tensor(float(other))
                isconstant = True
            except:
                pass

        # if just multiplying by a constant
        if isconstant:
            return self.__class__(degree=self.degree, input_dim=self.input_dim, init_C = other*self.C)

        # if we have another polynomial
        if self.input_dim != other.input_dim:
            raise ValueError("can only multiply polinomials with the same 'input_dim'")
        
        if self.input_dim!=1:
            raise ValueError("can only multiply polynomials with input_dim=1")

        newC = []
        for i in range(self.input_dim):
            c1 = self.C[i]
            c2 = other.C[i]

            newc = c1.reshape(-1,1) @ c2.reshape(1,-1)
            newc = sum_cross_diagonals(newc)
            newC.append(newc)

        newC = torch.stack(newC)
        newdegree = newC.shape[-1]-1
        
        r = self.__class__(degree=newdegree, input_dim=self.input_dim, init_C = newC)
        return r
    
    def reparametrize(self, parametric_polynomials):
        """
        reparametrize this polinomial following one parametric eq per input dimension
        parametric_polynomials: a list of polynomials, one per input dimension
                                each one must have input_dim=1
                                
        return: pol a new polinomial with one input dim such that if 
                parametric_polynomials = [k0, k1] the resulting polynomial
                pol is such that 
                        pol(s) = self(torch.stack([k0(s), k1(s)])) 
        """
        
        if len(parametric_polynomials)!=self.input_dim:
            raise ValueError(f"self.input_dim is {self.input_dim} for parametric_polynomials has {len(parametric_polynomials)} elements. they must be the same")

        pol = None
        for dim in range(len(parametric_polynomials)):
            for i in range(self.degree+1):
                constant = self.C[dim][i]
                newK = parametric_polynomials[dim]**i * constant 
                if pol is None:
                    pol = newK
                else:
                    pol = pol + newK
        return pol
    
    @classmethod
    def sample(cls, input_dim=None, degree=None, max_input_dim=10, max_degree=3):
        if input_dim is None:
            input_dim = np.random.randint(low=2, high=max_input_dim)
        if degree is None:
            degree = np.random.randint(low=1, high=max_degree)
        return cls(input_dim=input_dim,degree=degree)
    


class PolynomialLayerWithIndependantInputs(PolynomialLayer):
    
    """
    a layer with multiple outputs, each one for each input dim independant 
    from the other input dims
    """
    
    def forward(self, x):
        x = torch.stack([x**i for i in range(self.degree+1)])
        x = torch.einsum("abc,ca->bc", x, self.C)
        return x
    
    def derivative(self, *args, **kwargs):
        raise NotImplementedError()
        
    def integral(self, *args, **kwargs):
        raise NotImplementedError()
        
    def reparametrize(self, *args, **kwargs):
        raise NotImplementedError()

    def __mul__(self, other):
        raise NotImplementedError()
    

class PolinomialLayerSet(nn.Module):
    
    def __init__(self, size, input_dim, degree):
        super().__init__()
        self.input_dim = input_dim
        self.degree = degree
        self.size = size

    def init_layers(self):
        self.layers = [PolynomialLayer(input_dim = self.input_dim, 
                                       degree = self.degree) for _ in range(self.size)]
        return self
        
    def set_layers(self, layers):
        if not np.allclose([self.input_dim]+[layer.input_dim for layer in layers]):
            raise ValueError(f"all layers must have {self.input_dim} input_dim")
        
        if not np.allclose([self.degree]+[layer.degree for layer in layers]):
            raise ValueError(f"all layers must have {self.degree} degree")

        if self.size != len(layers):
            raise ValueError(f"expected {self.size} layers but got {len(layers)}")

        self.layers = layers
        return self
        
    def forward(self, x):
        return torch.transpose(torch.stack([layer(x) for layer in self.layers]), 0,1)

    def jacobian(self):
        """
        builds a new layer set with the derivatives of all layers wrt all dims
        """
        return PolynomialLayerWithIndependantInputsSet(input_dim = self.input_dim, degree = self.degree-1, size=self.size)\
                                                      .set_layers([layer.jacobian() for layer in self.layers])

    

class PolynomialLayerWithIndependantInputsSet(nn.Module):

    """
    assembles the coefficients so that we avoid looping over the layers in the set
    when forward is called
    """

    def __init__(self, size, input_dim, degree):
        super().__init__()
        self.input_dim = input_dim
        self.degree = degree
        self.size = size
        self.C = nn.Parameter(torch.rand(size=(self.size, self.input_dim, self.degree+1)))
        self.layers = None
   
    def init_layers(self):
        self.layers = [PolynomialLayer(input_dim = self.input_dim, degree = self.degree) for _ in range(self.size)]
        self.C.data = torch.stack([i.C for i in self.layers])

        return self
        
    def set_layers(self, layers):
        degrees    = [layer.degree for layer in layers]
        input_dims = [layer.input_dim for layer in layers]
        
        if not np.allclose(*degrees):
            raise ValueError(f"all sublayers must have the same degree, but found {degrees}")
        
        if not np.allclose(*input_dims):
            raise ValueError(f"all sublayers must have the same input_dim, but found {input_dims}")
        
        self.degree = degrees[0]
        self.input_dim = input_dims[0]
        self.layers = layers
        self.C.data = torch.stack([i.C for i in self.layers])
        return self
        
    def forward(self, x):
        xp = torch.stack([x**i for i in range(self.degree+1)])
        r  = torch.einsum("abc,dca->bdc", xp, self.C)
        return r
    
    def derivative(self, wrt=None):

        if wrt is None and self.input_dim!=1:
            raise ValueError("can only use wrt=None when input_dim is 1")

        if self.layers is not None:
            r = self.__class__(self.size, self.input_dim, self.degree-1)
            return r.set_layers([layer.derivative(wrt=wrt) for layer in self.layers])

        if wrt is None:
            # for 1d inputs we do it really fast
            if self.degree==0:
                r = self.__class__(self.size, self.input_dim, self.degree)
                r.C.data = r.C.data*0.
                return r
            else:
                r = self.__class__(self.size, self.input_dim, self.degree-1)
                powers = torch.arange(1,self.degree+1)
                r.C.data = (self.C[:,:,1:] * powers ).reshape(self.C[:,:,1:].shape)
                return r

        raise ValueError("cannot compute derivative")

    def integral(self):
        if self.input_dim!=1:
            raise ValueError("can only integrate when input_dim is 1")
        
        r = self.__class__(self.size, self.input_dim, self.degree+1)

        if self.layers is not None:
            return r.set_layers([layer.integral() for layer in self.layers])
        else:
            # fast computation without layers
            r = self.__class__(self.size, self.input_dim, self.degree+1)
            powers = torch.arange(1,self.degree+2)
            r.C.data = torch.cat([torch.zeros(self.size).reshape(-1,1,1), 
                                ( self.C / powers ).reshape(self.C.shape)], -1)
            return r


    def jacobian(self):
        """
        builds a new layer set with the derivatives of all layers wrt all dims
        """
        return PolynomialLayerWithIndependantInputsSet().set_layers([layer.jacobian() for layer in self.layers])
