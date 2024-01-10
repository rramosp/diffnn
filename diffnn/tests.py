    
# -------------------------------------------------------------------------------------
# TESTING
# -------------------------------------------------------------------------------------
import numpy as np
import torch
import sympy
from time import time
from progressbar import progressbar as pbar
from einsteinpy.symbolic import MetricTensor, ChristoffelSymbols, RiemannCurvatureTensor, RicciTensor
import sympy as sy
from itertools import product
from . import metrics
from . import polynomials

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

        mean_error = np.mean(np.abs(cs-ct))
        rel_error  = mean_error / np.mean(abs(cs))

        if rel_error>1e-3:
            print (f"failed at x {x.detach().numpy()}!!")
            print ("christofels evaluated by sympy\n", cs)
            print ("christofels evaluated by torch\n", ct)
            print (f"-- NUMERIC ERRORS: mean abs error {mean_error:.6f}  ::  mean rel error {rel_error:.6f}")
            break

        if rel_error > 1e-6:
            print (f"-- SMALL NUMERIC ERRORS: mean abs error {mean_error:.6f}  ::  mean rel error {rel_error:.6f}")

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

        mean_error = np.mean(np.abs(cs-ct))
        rel_error  = mean_error / np.mean(abs(cs))

        if rel_error>1e-3:
            print (f"failed at x {x.detach().numpy()}!!")
            print ("christofels evaluated by sympy\n", cs)
            print ("christofels evaluated by torch\n", ct)
            print (f"-- NUMERIC ERRORS: mean abs error {mean_error:.6f}  ::  mean rel error {rel_error:.6f}")
            break

        if rel_error > 1e-6:
            print (f"-- SMALL NUMERIC ERRORS: mean abs error {mean_error:.6f}  ::  mean rel error {rel_error:.6f}")


        vals_mean.append(cs.mean())
    vals_mean = np.r_[vals_mean]
    print (f"mean value {vals_mean.mean():.4f}, std value {vals_mean.std():.4f}")

    
class PolynomialLayerTest():
    
    def __init__(self, a_polynomial_layer, Cname="C", Xname="x"):
        self.pl = a_polynomial_layer
        self.Cname = Cname
        
        self.x_sympy = sy.symbols([f"{Xname}"+"^{"+f"{d}"+"}" for d in range(self.pl.input_dim)])
        self.C_sympy = sy.symbols([[f"{Cname}_"+"{"+f"{d}.{p}"+"}" for p in range(self.pl.degree+1)] for d in range(self.pl.input_dim)])
        self.expr_sympy = sum([self.C_sympy[d][p]*self.x_sympy[d]**p for d in range(self.pl.input_dim) for p in range(self.pl.degree+1)])
        
        self.subsC = {self.C_sympy[d][p]: self.pl.C.detach().numpy()[d][p] for d in range(self.pl.input_dim) for p in range(self.pl.degree+1)}
        
    def sample_input(self, nsamples):
        return torch.rand(size=(nsamples, self.pl.input_dim))
    

    def sympy_forward(self, expr_sympy, x):
        sympy_forward = []
        for xi in x:
            subsX = {self.x_sympy[i]:xi[i] for i in range(len(xi))}
            sympy_forward.append(float(expr_sympy.subs(self.subsC).subs(subsX)))

        return np.r_[sympy_forward] 
        
    def check_forward(self, x=None):

        if x is None:
            x = self.sample_input(np.random.randint(low=5,high=30))

        sympy_forward = self.sympy_forward(self.expr_sympy, x)
        polynomial_layer_forward = self.pl(x).detach().numpy()
        return np.allclose(sympy_forward, polynomial_layer_forward)


def test_polynomial_layer_addition():
    for _ in range(10):
        p1 = polynomials.PolynomialLayer.sample(max_input_dim=4, max_degree=3)
        p2 = polynomials.PolynomialLayer.sample(input_dim = p1.input_dim, max_degree=3)
        tp1 = PolynomialLayerTest(p1, Cname='C')
        tp2 = PolynomialLayerTest(p2, Cname='K')

        # set same symbolic variables for input in both test layers 
        tp2.x_sympy = tp1.x_sympy

        expression = (tp1.expr_sympy.subs(tp1.subsC) + tp2.expr_sympy.subs(tp2.subsC)).expand()
        x = tp2.sample_input(10)

        sympy_vals = tp2.sympy_forward(expression, x)
        torch_vals = (p1+p2)(x).detach().numpy()

        if not np.allclose(sympy_vals, torch_vals, atol=1e-3):
            print ("failed addition on polynomial layers with coefficients")
            print (p1.C)
            print ("and ---")
            print (p2.C)
            print ("torch returned", torch_vals)
            print ("sympy returned", sympy_vals)
            break
    return tp1, tp2, x

def test_polynomial_layer_multiplication():
    for _ in range(10):
        p1 = polynomials.PolynomialLayer.sample(input_dim = 1, max_degree=10)
        p2 = polynomials.PolynomialLayer.sample(input_dim = p1.input_dim, max_degree=10)
        tp1 = PolynomialLayerTest(p1, Cname='C')
        tp2 = PolynomialLayerTest(p2, Cname='K')

        # set same symbolic variables for input in both test layers 
        tp2.x_sympy = tp1.x_sympy

        expression = (tp1.expr_sympy.subs(tp1.subsC) * tp2.expr_sympy.subs(tp2.subsC)).expand()
        x = tp2.sample_input(10)

        sympy_vals = tp2.sympy_forward(expression, x)
        torch_vals = (p1*p2)(x).detach().numpy()

        if not np.allclose(sympy_vals, torch_vals, atol=1e-3):
            print ("failed addition on polynomial layers with coefficients")
            print (p1.C)
            print ("and ---")
            print (p2.C)
            print ("torch returned", torch_vals)
            print ("sympy returned", sympy_vals)
            raise ValueError("check failed")
    return tp1, tp2, x

def test_polynomial_derivative_and_integrals():
    input_dim = np.random.randint(low=1, high=10, size=1)[0]
    degree = np.random.randint(low=1, high=10, size=1)[0]
    nelems = np.random.randint(low=100, high=1000)

    _x = torch.rand(size=(nelems, input_dim), requires_grad=True)
    pl = polynomials.PolynomialLayer(degree=degree, input_dim=input_dim)
    pld = pl.derivative()
    pli = pl.integral()

    t0 = time()
    derivative_from_autograd = torch.stack([torch.autograd.grad(pl(_x)[i], _x)[0].sum(axis=1)[i] for i in range(len(_x))])
    t1 = time()
    derivative_from_pol  = pld(_x)
    t2 = time()

    derivative_of_integral_from_autograd = torch.stack([torch.autograd.grad(pli(_x)[i], _x)[0].sum(axis=1)[i] for i in range(len(_x))])
    values_from_pol  = pl(_x)

    test_derivative = torch.allclose(derivative_from_autograd, derivative_from_pol)
    test_integral = torch.allclose(derivative_of_integral_from_autograd, values_from_pol)

    if not test_derivative:
        raise ValueError("derivative test failed")

    if not test_integral:
        raise ValueError("integration test failed")

    print ("computation correct")
    print (f"time autograd              {(t1-t0)*1000:.3f} milisecs")
    print (f"time polynomial derivative {(t2-t1)*1000:.3f} milisecs")    

def test_polynomial_integral():
    input_dim = 1
    degree = np.random.randint(low=1, high=10, size=1)[0]
    nelems = np.random.randint(low=5, high=20)
        
    _x = torch.rand(size=(nelems, input_dim), requires_grad=True)
    pl = polynomials.PolynomialLayer(degree=degree, input_dim=input_dim)

    print ("compting integral with sympy", flush=True)
    t1 = time()
    torch_polynomial_integral = pl.integral()(_x)
    t2 = time()

    print ("compting integral with sympy", flush=True)
    t3 = time()
    for dxi in range(pl.input_dim):
        _xd = []
        for _xi in _x:
            tpl = PolynomialLayerTest(pl)
            # differentiate with sympy
            _xd.append(float(tpl.expr_sympy.integrate(tpl.x_sympy[dxi]).subs(tpl.subsC).subs({tpl.x_sympy[i]: _xi[i] for i in range(pl.input_dim)})))
        _xd = torch.tensor(_xd)

        if not torch.allclose(_xd , torch_polynomial_integral):
            raise ValueError("derivative test with sympy failed")
            
    t4 = time()
    
    time_analytic = t2-t1
    time_sympy    = t4-t3
            
    print ("computation of derivatives correct")
    print (f"time sympy      ({nelems:4d} data points)      {time_sympy*1000:9.2f} milisecs")    
    print (f"time polynomial ({nelems:4d} data points)  {time_analytic*1000:9.2f} milisecs")    


def test_polynomial_derivative():

    input_dim = np.random.randint(low=1, high=10, size=1)[0]
    degree = np.random.randint(low=1, high=10, size=1)[0]
    nelems = np.random.randint(low=10, high=200)
        
    _x = torch.rand(size=(nelems, input_dim), requires_grad=True)
    pl = polynomials.PolynomialLayer(degree=degree, input_dim=input_dim)

    t0 = time()
    print ("compting derivatives with autograd", flush=True)
    derivative_from_autograd = torch.stack([torch.autograd.grad(pl(_x)[i], _x)[0].sum(axis=0) for i in range(len(_x))])
    t1 = time()
    print ("compting derivatives with polynomials", flush=True)
    derivative_from_pol      = torch.stack([pl.derivative(wrt=i)(_x) for i in range(pl.input_dim)]).T
    t2 = time()

    test_derivative = torch.allclose(derivative_from_autograd, derivative_from_pol)

    if not test_derivative:
        raise ValueError("derivative test with autograd failed")

    print ("compting derivatives with sympy", flush=True)
    t3 = time()
    for dxi in range(pl.input_dim):
        _xd = []
        for _xi in _x[:10]:
            tpl = PolynomialLayerTest(pl)
            # differentiate with sympy
            _xd.append(float(tpl.expr_sympy.diff(tpl.x_sympy[dxi]).subs(tpl.subsC).subs({tpl.x_sympy[i]: _xi[i] for i in range(pl.input_dim)})))
        _xd = torch.tensor(_xd)

        if not torch.allclose(_xd , pl.derivative(wrt=dxi)(_x[:10])):
            raise ValueError("derivative test with sympy failed")
            
    t4 = time()
    
    time_autograd = t1-t0
    time_analytic = t2-t1
    time_sympy    = t4-t3
            
    print ("computation of derivatives correct")
    print (f"time autograd   ({nelems:4d} data points)  {time_autograd*1000:9.2f} milisecs")
    print (f"time sympy      ({10:4d} data points)      {time_sympy*1000:9.2f} milisecs")    
    print (f"time polynomial ({nelems:4d} data points)  {time_analytic*1000:9.2f} milisecs")    
  

def test_polynomial_metric_derivatives():
    for _ in range(10):
        degree = np.random.randint(5)+1   # polynomial degree
        n = np.random.randint(8)+2        # manifold dimension
        m = np.random.randint(100)+10

        print (f". degree {degree}, dim {n}, data size {m} :: ", flush=True)
        
        x = torch.rand(size=(m,n))
        
        pms = [metrics.RiemmanianMetricWithPolynomialLayers(degree=degree, input_dim=x.shape[-1]),
               metrics.RiemmanianMetricWithBarePolynomials(degree=degree, dim=x.shape[-1])]        
        
        nruns = 1000
        for pm in pms:
            t1 = time()
            # compute several times with polynomials
            for _ in range(nruns):
                dm1 = pm.metric_derivative(x)
            t2 = time()
            # compute them with torch autograd
            dm2 = super(pm.__class__, pm).metric_derivative(x)
            t3 = time()
            if not torch.allclose(dm1, dm2):
                raise ValueError("test failed")

            print(f"{pm.__class__.__name__:40s} result correct :: time analytical {(t2-t1)/nruns:.5f} secs, time autograd {t3-t2:.5f} secs, speedup x{nruns*(t3-t2)/(t2-t1):.2f}") 


def test_arclen():
    dim = np.random.randint(low=2, high=4)
    degree = np.random.randint(low=1, high=5)
    nelems = np.random.randint(low=5, high=20)

    print (f"dim {dim}, degree {degree}, nelems {nelems}", flush=True)

    # -------- create symbolic objects for metric tensor and geodesic -------
    x = sy.symbols(["x^{"+f"{d}"+"}" for d in range(dim)])
    s = sy.symbols("s")
    K = sy.symbols([["K_{"+f"{d}.{p}"+"}" for p in range(degree+1)] for d in range(dim)])
    C = sy.symbols([[[["C_{"+f"{i}.{j}.{d}.{p}"+"}" for p in range(degree+1)] for d in range(dim)] for j in range(dim)] for i in range(dim)])

    # the metric
    g = np.zeros((dim, dim))
    g = [list(gi) for gi in g]
    for i in range(dim):
        for j in range(dim):
            g[i][j] = sum([C[i][j][d][p]*x[d]**p for d in range(dim) for p in range(degree+1)])

    # the geodesic
    xs = list(np.zeros(dim))
    for d in range(dim):
        xs[d] = sum([K[d][p]*(s**p) for p in range(degree+1)])

    # -------- create torch polinomial objects for metric tensor and geodesic -------
    # -------- and sample data                                                -------
    _s = torch.rand(size=(nelems,1)) 
    _p = polynomials.PolynomialLayer(degree=degree, input_dim=dim)
    _k = [polynomials.PolynomialLayer(degree=degree, input_dim=1) for _ in range(dim)]
    _x = torch.stack([_k[_i](_s) for _i in range(len(_k))]).T
    _pp = _p.reparametrize(_k)

    # ------- check reparametrization works
    if not torch.allclose(_pp(_s), _p(_x)):
        print ("XXX reparametrization check failed")

    # ------- loop over all the elements of the metric tensor
    for i,j in pbar(product(range(dim), range(dim)), max_value=dim**2):

        # create substition maps with specifc values for parameters and data
        subsC = {C[i][j][d][p]: _p.C.detach().numpy()[d][p] for d in range(dim) for p in range(degree+1)}
        subsK = {K[di][p]: _k[di].C[0].detach().numpy()[p] for p in range(degree+1) for di in range(dim)} 
        subsXsymbols = {x[ii]: xs[ii] for ii in range(len(x))}

        # sympy expression for arclen
        arclenexpr = g[i][j].subs(subsXsymbols)*xs[i].diff(s) * xs[j].diff(s)
        arclenexpr_integral = arclenexpr.integrate(s)

        # for each element in data
        valsp,valsarclen, valsarclenintegral = [], [], []
        for ni in range(len(_x)):

            # compute in sympy the metric tensor value
            subsX = {x[d]: _x[ni].detach().numpy()[d] for d in range(dim)} 
            subsS = {s: _s.detach().numpy()[ni,0]}
            valsp.append(float(g[i][j].subs(subsC).subs(subsX)))

            # compute in sympy the arclen infinitesimal element value
            valsarclen.append(float(arclenexpr.subs(subsXsymbols).subs(subsC).subs(subsK).subs(subsS)))    
            valsarclenintegral.append(float(arclenexpr_integral.subs(subsXsymbols).subs(subsC).subs(subsK).subs(subsS)))    

        # compare metric tensor values
        torchpol_metricvalues = _p(_x).detach().numpy()
        metricok = np.allclose(np.r_[valsp], torchpol_metricvalues)

        arclen_infinitesimal_element = _pp*_k[i].derivative()*_k[j].derivative()

        # compare with sympy values for arclen infinitesimal element $g_{ij}\frac{dx^i}{ds}\frac{dx^j}{ds}$
        # 1. multiplying the data output of each equation element modeled as a polynomial
        torchpol_arclenvalues       = (_pp(_s)*_k[i].derivative()(_s)*_k[j].derivative()(_s)).detach().numpy()
        # 2. obtaining a single polynomial for the equation and then applying it to the data
        torchsinglepol_arclenvalues = arclen_infinitesimal_element(_s).detach().numpy()

        arclenok1 = np.allclose( torchpol_arclenvalues, np.r_[valsarclen])
        arclenok2 = np.allclose( torchpol_arclenvalues, torchsinglepol_arclenvalues)

        # compare integration of arclen infinitesimal element
        torch_arclen_integral = arclen_infinitesimal_element.integral()(_s).detach().numpy()
        arclenok3 = np.allclose(torch_arclen_integral, np.r_[valsarclenintegral], atol=1e-3)

        if not metricok:
            print ("check failed to metric tensor")

        if not arclenok1 or not arclenok2:
            print ("check failed computing arclen infinitesimal element")

        if not arclenok3:
            print ("check failed integrating arclen infinitesimal element")


def run_polynomial_derivatives_and_integrals():
    print ("---------------------------------------------------------------------------- ")
    print ("-- testing polynomial layer computes correctly derivatives and integrals --- ")
    print ("-- and comparing execution times wrt torch autograd                      --- ")
    print ("---------------------------------------------------------------------------- ")
    print ("---- derivatives ----")
    test_polynomial_derivative()
    print ("---- integrals ----")
    test_polynomial_integral()

def run_polynomial_metric_derivatives():

    print ("---------------------------------------------------------------------------- ")
    print ("-- testing polynomial metric computes correctly metric derivatives --------- ")
    print ("-- and comparing execution times wrt torch autograd jacobians     ---------- ")
    print ("---------------------------------------------------------------------------- ")
    test_polynomial_metric_derivatives()

def run_metric_tensor_and_christoffels_test():
    print ("-----------------------------------------------------------------------------------------")
    print ("-- testing riemmanian metric computes correctly metric tensor and christoffels ---------")
    print ("-----------------------------------------------------------------------------------------")
    nsamples = 10
    for dims in [2,3]:
        print (f"-------- generating {nsamples} random {dims}D metric tensors ")
        for _ in range(nsamples):
            motest = MainTest(dims=dims)
            test_metric_tensor(motest) 
            test_christoffels(motest)
            print()

def run_polynomial_arclen_test():
    print ("-----------------------------------------------------------------------------------------")
    print ("-- testing polynomial arclens are integrals are computed ok                     ---------")
    print ("-----------------------------------------------------------------------------------------")
    test_arclen()

def run():
    run_polynomial_derivatives_and_integrals()
    run_polynomial_metric_derivatives()
    run_metric_tensor_and_christoffels_test()
