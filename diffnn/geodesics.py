from torch import nn
from torch import tensor
import numpy as np

class GeodesicEquation(nn.Module):
    def __init__(self, manifold_dim=2):
        super().__init__()
        
        self.manifold_dim = manifold_dim
        self.input_dim  = 1
                             
        self.build()
        self.nparams = sum([np.product(i.shape) for i in self.parameters()])
        self.substitute_parameters_with_tensors()
        
    def build(self):

        raise NotImplementedError()
        
    def forward(self, x):
        raise NotImplementedError()

    def params_checksum(self):
        return sum([i.sum().detach().numpy() for i in self.nograd_params()])
        
    def substitute_parameters_with_tensors(self):
        """
        this network is not going to be learned, so we need the parameters to 
        stay away from automatic gradients.
        """
        named_params = list(self.named_parameters())
        self.named_params = []
        for n,p in named_params:
            n = "self."+".".join([f"[{i}]" if i.isdigit() else i for i in n.split(".")]).replace(".[", "[")
            self.named_params.append(n)
            
            cmd = f"del({n})\n{n} = {p.detach()}"
            exec(cmd)        
        
    def nograd_params(self):
        return eval("["+",".join(self.named_params)+"]")        
        
    def set_nograd_params(self, flat_params):
        if len(flat_params.shape)!=1 or len(flat_params)!=self.nparams:
            raise ValueError(f"incorrect number of parameters for this network. flat_params has shape {flat_params.shape} but model has {self.nparams} parameters")

        i=0
        unflatten_params = []
        for param in self.nograd_params():
            param_len = np.product(param.shape)
            unflatten_params.append(flat_params[i:i+param_len].reshape(param.shape))
            i += param_len

        for param_vals, param_name in zip(unflatten_params, self.named_params):
            cmd = f"{param_name} = param_vals"
            exec(cmd)


class GeodesicEquationWithNeuralNetwork(GeodesicEquation):

    def build(self):
        self.linear_stack = nn.Sequential(
            nn.Linear(self.input_dim, 10),
            nn.ELU(),
            nn.Linear(10,20),
            nn.ELU(),
            nn.Linear(20,10),
            nn.ELU(),
            nn.Linear(10, self.manifold_dim),
        )
        
    def forward(self, x):
        x = self.linear_stack(x)
        return x

class EuclideanGeodesicEquation(GeodesicEquation):

    def build(self):
        self.linear = nn.Linear(self.input_dim, self.manifold_dim, bias=True)
        
    def forward(self, x):
        return self.linear(x)