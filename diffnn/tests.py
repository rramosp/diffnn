    
# -------------------------------------------------------------------------------------
# TESTING
# -------------------------------------------------------------------------------------
import numpy as np
import torch
import sympy
from progressbar import progressbar as pbar
from einsteinpy.symbolic import MetricTensor, ChristoffelSymbols, RiemannCurvatureTensor, RicciTensor
from . import metrics

def create_expr(symbol):
    """ creates an expression of a single symbol """
    
    pkgprefix = "##PKG##"
    
    funcs = ['sin', None]
    powers = [0, 1, 2]

    #funcs = [None]
    #powers = [0, 1, 2]
    
    f = np.random.choice(funcs)
    p = np.random.choice(powers)
    c = np.random.randint(10)-5
    if c==0: c=1
    
    expr = f"{pkgprefix}{f}({symbol})" if f is not None else symbol
    
    if p==0:
        expr = "1"
    elif p==1:
        expr = expr
    else:
        expr = f"{expr}**{p}" 
        
    if c==0:
        expr = "0"
    elif c==1:
        expr = expr
    else:
        if expr=='1':
            expr = c
        else:
            expr = f"{c}*{expr}"
    
    return expr

def asseble_expr(symbols):
    r = [create_expr(s) for s in symbols]
    if '0' in r:
        return '0'
    
    ops = np.random.choice(['*','+','-'], size=len(r)-1)
    re = r[0]
    for i in range(1, len(r)):
        re = f"{re} {ops[i-1]} {r[i]}"
    #re = re.replace("- -", "-").replace("--", "-")
    #re = re.replace("+ -", "-").replace("+-", "-")
    return re

def create_symetric_lists(z):
   
    """
    creates a list of lists with the elements of z arranged in a symmetric matrix
    """

    # number of distinct elements to be placed in the symmetric matrix
    k = len(z)
    
    # dimension of the symmetric matrix
    n = int((np.sqrt(1+8*k)-1)/2)
    
    # check
    _k = n*(n+1)/2
    if _k!=k:
        raise ValueError(f"invalid number of distinct elements for a symmetric matrix, {k}")

    i,j = torch.triu_indices(n,n)
    r = [[0 for _ in range(n)] for _ in range(n)]
    for zi, (_i, _j) in enumerate(zip(i,j)):
        r[_i][_j] = z[zi]
        r[_j][_i] = z[zi]
    
    return r


class RiemmanianMetricTest(metrics.RiemmanianMetric):

    def __init__(self, motest):
        """
        motest: a MetricObjectsTest instance
        """
    

        # build the expressions for the metric tensor
        torch_exprs = [[i.replace("##PKG##", "torch.")+"*torch.tensor(1.)" for i in row] for row in motest.metric_tensor_expressions]
    
        torch_init = f"""{','.join(motest.symbols)} = x"""
        row = torch_exprs[0]
        expr = 'torch.stack(['
        for row in torch_exprs:
            expr += f"torch.stack([{', '.join(row)}]), "
        expr += "])"
    
        # create the 'metric' method of this class dynamically
        self.fun_expr= f"""
def _tmp_metric(x):
    {torch_init}
    return {expr}
self._metric = _tmp_metric
        """
        exec(self.fun_expr)
        

    def metric(self, x):
        if len(x.shape)==1:
            return self._metric(x)
        if len(x.shape)==2:
            return torch.stack([self._metric(xi) for xi in x])
        else:
            raise ValueError(f"x must be 1D (a single element) or 2D (a batch of elements), but found shape {x.shape}")
    
class MainTest:
    
    def __init__(self, dims=3, init_sympy_torch=True):
        self.dims = dims

        # creates a metric tensor with random expressions
        all_symbols = 'abcdefghijklmnopqrstuvxyz'
        self.symbols = list(all_symbols[:dims])

        k = dims*(dims+1)//2 # number of unique values in the metric tensor

        exprs = [asseble_expr(self.symbols) for _ in range(k)]
        self.metric_tensor_expressions = create_symetric_lists(exprs)

        if init_sympy_torch:
            self.torch_init()
            self.sympy_init()
        
    def sympy_init(self):
        
        ## creates the metric tensor expression in sympy
        sympy_exprs = [[i.replace("##PKG##", "sympy.") for i in row] for row in self.metric_tensor_expressions]
        
        # init sympy symbols
        sympy_init = f"""
{','.join(self.symbols)} = sympy.symbols({self.symbols})
        """
        exec(sympy_init)

        # define the symbolic metric tensor
        print ("initializing sympy metric tensor", flush=True)
        expr = "self.sympy_mt = MetricTensor((["
        for row in sympy_exprs:
            expr += "[" + ",".join(row) + "],\n"
        expr += "]), ("+",".join(self.symbols)+"))"        
        exec(expr)
        print ("metric tensor is", self.sympy_mt)
        # compute symbolic christoffels
        print ("computing symbolic christoffels in sympy", flush=True)
        expr = "self.sympy_christoffels = ChristoffelSymbols.from_metric(self.sympy_mt)"
        exec(expr)
        
    def torch_init(self):
        print ("initializing torch graphs", flush=True)        
        self.torch_metric = RiemmanianMetricTest(self)


def test_christoffels(self, nsamples=100):
    print ("testing christoffels torch and sympy implementations", flush=True)
    vals_mean = []    
    for _ in pbar(range(nsamples)):
        x = torch.rand(size=(self.dims,), requires_grad=True)
        substitution = {s:v for s,v in zip(self.symbols, x.detach().numpy())}
        cs = np.r_[self.sympy_christoffels.tensor().subs(substitution)].astype(np.float32)
        ct = self.torch_metric.christoffels(x).detach().numpy()
        if not np.allclose(cs, ct, atol=1e-3):
            print (f"failed at x {x.detach().numpy()}!!")
            print ("christofels evaluated by sympy\n", cs)
            print ("christofels evaluated by torch\n", ct)
            print (f"avg values abs difference {np.mean(np.abs(cs-ct)):.6f}")
            break
        vals_mean.append(cs.mean())
    vals_mean = np.r_[vals_mean]
    print (f"mean value {vals_mean.mean():.4f}, std value {vals_mean.std():.4f}")
        
def test_metric_tensor(self, nsamples=100):
    print ("testing metric tensors torch and sympy implementations", flush=True)
    vals_mean = []
    for _ in pbar(range(nsamples)):
        x = torch.rand(size=(self.dims,), requires_grad=True)
        substitution = {s:v for s,v in zip(self.symbols, x.detach().numpy())}
        cs = np.r_[self.sympy_mt.tensor().subs(substitution)].astype(np.float32)
        ct = self.torch_metric.metric(x).detach().numpy()
        if not np.allclose(cs, ct, atol=1e-3):
            print (f"failed at x {x.detach().numpy()}!!")
            print ("metric tensor evaluated by sympy\n", cs)
            print ("metric tensor evaluated by torch\n", ct)
            print (f"avg values abs difference {np.mean(np.abs(cs-ct)):.6f}")
            break
        vals_mean.append(cs.mean())
    vals_mean = np.r_[vals_mean]
    print (f"mean value {vals_mean.mean():.4f}, std value {vals_mean.std():.4f}")

    
def run():
    nsamples = 10
    for dims in [2,3]:
        print ("----------------------------------------------------")
        print (f"-------- generating {nsamples} random {dims}D metric tensors ")
        print ("----------------------------------------------------")
        for _ in range(nsamples):
            motest = MainTest(dims=dims)
            test_metric_tensor(motest) 
            test_christoffels(motest)
            print()