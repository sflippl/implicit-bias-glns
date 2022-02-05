import copy

import torch
import torch.nn as nn

class FRN(nn.Module):
    def __init__(self, inp_dim, hidden_dims, outp_dim, init_seed=1, bias=True):
        super().__init__()
        lst = []
        _bias = False
        for _inp_dim, _outp_dim in zip([inp_dim]+hidden_dims[:-1], hidden_dims):
            lst.append(nn.Linear(_inp_dim, _outp_dim, bias=_bias))
            _bias = bias
        lst.append(nn.Linear(hidden_dims[-1], outp_dim, bias=_bias))
        self.weights = nn.ModuleList(lst)
        self.gates = self.weights
        self.init_seed = init_seed
        self.init()
    
    def forward(self, x, return_gates=False):
        w_x = x
        g_x = x
        gates = []
        for i, (w_layer, g_layer) in enumerate(zip(self.weights, self.gates)):
            w_x = w_layer(w_x)
            if i < len(self.weights)-1:
                with torch.no_grad():
                    g_x = g_layer(g_x)
                    gate = (g_x >= 0.).float()
                    gates.append(gate)
                    g_x = gate*g_x
                w_x = gate*w_x
        if return_gates:
            return w_x, gates
        return w_x
    
    def freeze(self):
        self.gates = copy.deepcopy(self.gates)
    
    def init(self):
        torch.manual_seed(self.init_seed)
        for weight in self.weights:
            nn.init.kaiming_normal_(weight.weight)
