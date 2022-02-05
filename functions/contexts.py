import itertools

import torch
import torch.nn as nn
from torch.distributions.uniform import Uniform

class HyperplaneContext(nn.Module):
    def __init__(self, inp_dim, outp_dim, fun_dim):
        super().__init__()
        self.fun = nn.Linear(inp_dim, outp_dim*fun_dim, bias=False)
        self.outp_dim = outp_dim
        self.beta = torch.zeros(outp_dim, fun_dim)
        self.fun_dim = fun_dim
        self.caster = torch.tensor(list(itertools.product(*([[0,1]]*fun_dim)))).float().T
        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.fun.weight, std=36.)
        nn.init.normal_(self.beta, std=9.)

    def forward(self, z):
        outp = self.fun(z).reshape(*z.shape[:(-1)], self.outp_dim, self.fun_dim)
        outp = (outp > self.beta.reshape(*([1]*(len(z.shape)-1)), self.outp_dim, self.fun_dim)).float()
        outp = torch.prod(outp.unsqueeze(-1)==self.caster.reshape(*([1]*(len(z.shape))), *self.caster.shape), dim=-2)
        return outp
    
    def quantile_init_beta(self, data, q_lims=0.5):
        outp = self.fun(data).reshape(-1, self.outp_dim*self.fun_dim)
        if isinstance(q_lims, (float,)):
            beta = torch.quantile(outp, q_lims, dim=0)
        elif q_lims[0] == q_lims[1]:
            beta = torch.quantile(outp, q_lims[0], dim=0)
        else:
            beta = torch.quantile(outp, Uniform(*q_lims).sample((self.outp_dim*self.fun_dim,)), dim=0)
            beta = torch.diagonal(beta, dim1=0, dim2=1)
        self.beta = beta.reshape(self.outp_dim, self.fun_dim)
