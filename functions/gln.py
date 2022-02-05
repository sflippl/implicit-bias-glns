from itertools import chain

import torch
import torch.nn as nn

from .contexts import HyperplaneContext

class GLNElement(nn.Module):
    def __init__(self, inp_dim, outp_dim, context, n_contexts, bias=True):
        super().__init__()
        self.fun = nn.Linear(inp_dim, outp_dim*n_contexts, bias=bias)
        self.context = context
        self.n_contexts = n_contexts
        self.inp_dim = inp_dim
        self.outp_dim = outp_dim
        self.bias = bias

    def forward(self, x, z):
        if self.n_contexts > 1:
            with torch.no_grad():
                con = self.context(z)
            outp = self.fun(x).reshape(*x.shape[:(-1)], self.outp_dim, self.n_contexts)
            outp = torch.sum(outp*con, dim=-1)
        else:
            outp = self.fun(x)
        return outp

class GLN(nn.Module):
    def __init__(self, inp_dim, latent_dims, outp_dim, shatter_dims, bias=True, seed=1):
        super().__init__()
        torch.manual_seed(seed)
        lst = []
        # The first layer already has a bias by the way we've set up the dataset.
        _bias = False
        for _inp_dim, _outp_dim, _shatter_dim in zip(
            [inp_dim] + list(latent_dims), list(latent_dims) + [outp_dim], shatter_dims
        ):
            if _shatter_dim == 0:
                context_fun = None
            else:
                context_fun = HyperplaneContext(inp_dim, _outp_dim, _shatter_dim)
            lst.append(GLNElement(_inp_dim, _outp_dim, context_fun, 2**_shatter_dim, bias=_bias))
            _bias = bias
        self.funs = nn.ModuleList(lst)
        with torch.no_grad():
            self.init()
    
    def forward(self, x):
        rtn = []
        outp = x
        for fun in self.funs:
            outp = fun(outp, x)
        return outp
    
    def init(self):
        for linear in self.funs:
            nn.init.orthogonal_(linear.fun.weight)
    
    def quantile_init_beta(self, x):
        for fun in self.funs:
            if fun.context is not None:
                fun.context.quantile_init_beta(x)

    def parameters(self):
        return iter(sum([list(gln_element.fun.parameters()) for gln_element in self.funs], start=[]))

