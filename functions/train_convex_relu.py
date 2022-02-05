import math

import numpy as np
import cvxpy as cp
import pandas as pd
import torch

def predict(var, train_data, objective):
    if objective in ['gates_only', 'hidden_layer']:
        return train_data['x']@var
    else:
        gated_var = train_data['context']@var
        return cp.sum(cp.multiply(gated_var, train_data['x']), axis=1)

def get_objective(var, objective):
    if objective in ['gates_only', 'hidden_layer']:
        return cp.Minimize(cp.sum_squares(var))
    else:
        norm = cp.norm2(var, axis=1)
        return cp.Minimize(cp.sum(norm))

def get_constraints(var, train_data, objective):
    y_hat = predict(var, train_data, objective)
    return [cp.multiply(train_data['y'], y_hat) >= 1]

def get_acc(var, data, objective):
    y_hat = predict(var, data, objective).value
    y_hat = np.sign(y_hat)
    return (data['y'] == y_hat).astype(float).mean()

def train_convex_relu(objective, train_data, val_data, max_iters=200, comparison_model=None):
    n_data, dim_x = train_data['x'].shape
    if objective in ['gates_only', 'hidden_layer']:
        var = cp.Variable((dim_x,))
    elif objective in ['learned_contexts', 'random_contexts']:
        _, n_contexts = train_data['context'].shape
        var = cp.Variable((n_contexts, dim_x))
    else:
        raise NotImplementedError()
    _constraints = get_constraints(var, train_data, objective)
    _objective = get_objective(var, objective)
    prob = cp.Problem(_objective, _constraints)
    if objective in ['gates_only', 'hidden_layer']:
        method = 'OSQP'
        kwargs = {}
    else:
        method = 'ECOS'
        kwargs = {'max_iters': max_iters}
    prob.solve(solver=method, verbose=True, **kwargs)
    if prob.status == 'infeasible':
        accs = [np.nan, np.nan, np.nan]
    else:
        train_acc = get_acc(var, train_data, objective)
        val_acc = get_acc(var, val_data, objective)
        accs = [train_acc, val_acc]
        if comparison_model is not None:
            y_hat = np.sign(predict(var, val_data, objective).value)
            with torch.no_grad():
                y_comp = np.sign(comparison_model(torch.from_numpy(val_data['comp_x']))[:,0].numpy())
            accs.append((y_hat==y_comp).astype(float).mean())
    return accs
