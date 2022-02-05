import itertools
import math

import numpy as np
import cvxpy as cp
import pandas as pd
import miniball as mb
import scipy.sparse as sp
import torch

def collapse_contexts(data):
    n_data, n_latent, n_contexts = data['context'].shape
    caster = np.array(
        list(itertools.product(*([list(range(n_contexts))]*n_latent)))
    ).astype(float)
    caster = (
        caster.reshape(*caster.shape, 1) == np.linspace(0, n_contexts-1, n_contexts)
    ).astype(float)
    new_context = (caster.reshape(1, *caster.shape)==data['context'].reshape(n_data, 1, n_latent, n_contexts)).astype(float)
    new_context = new_context.prod(axis=(2,3))
    data['context'] = new_context
    return data

def predict(var, train_data, constraints):
    return {
        'none': predict_none,
        'architecture': predict_architecture,
        'implicit_bias': predict_architecture
    }[constraints](var, train_data)

def predict_none(var, train_data):
    x, y, c = train_data['x'], train_data['y'], train_data['context']
    n_data, n_contexts = c.shape
    y_hat = cp.sum(cp.multiply(c@var, x), axis=1)
    return y_hat

def predict_architecture(var, train_data):
    x, y, c = train_data['x'], train_data['y'], train_data['context']
    n_data, n_latent, n_contexts = c.shape
    c = c.reshape(n_data, n_latent*n_contexts)
    y_hat = cp.sum(cp.multiply(c@var, x), axis=1)
    return y_hat

def get_constraints(var, train_data, constraints):
    y_hat = predict(var, train_data, constraints)
    return [cp.multiply(train_data['y'], y_hat) >= 1]

def get_none_objective(var, n_latent, n_contexts):
    return cp.Minimize(cp.sum_squares(var))

# To define the architecture objective, we use Prop. A.1.
# We must use the flattened variable, which represents the three dimensions in the order
# latent, context, x_dim

def get_architecture_objective(var, n_latent, n_contexts):
    x_dim = 785
    # We can express the norm by transforming zeta to zeta'_{h,c,d}=1/(HC)sum_{h',c'}zeta_{h',c'}-sum_{c'}zeta_{h,c'}+Czeta_{h',c'}.
    part_1 = cp.sum(var, axis=0)
    part_1 = cp.reshape(part_1, (1, x_dim))
    part_2 = cp.sum(cp.reshape(cp.transpose(var), (x_dim*n_latent, n_contexts)), axis=1)
    part_2 = cp.reshape(part_2, (x_dim*n_latent, 1))
    part_3 = cp.reshape(cp.transpose(var), (x_dim*n_latent, n_contexts))
    part_2_3 = n_contexts*part_3-part_2
    part_2_3 = cp.transpose(cp.reshape(part_2_3, (x_dim, n_latent*n_contexts)))
    zeta_tilde = part_2_3 + 1/(n_latent*n_contexts)*part_1
    norm = cp.sum_squares(zeta_tilde)
    return cp.Minimize(norm)

# We test that this is correct by creating a numpy tensor:
# rs = np.random.RandomState(seed=1)
# tensor = rs.randn(785, 3, 4)
# zeta_tilde = 1/12*tensor.sum(axis=(1,2), keepdims=True) - tensor.sum(axis=2, keepdims=True) + 4*tensor

def get_implicit_bias_objective(var, n_latent, n_contexts):
    # For this objective, we first compute the p-norm across the image dimension.
    norm = cp.norm2(var, axis=1)
    norm = cp.reshape(norm, (n_latent, n_contexts))
    norm = cp.norm2(norm, axis=1)
    return cp.Minimize(cp.sum(norm))

# To see that the implicit bias objective returns the correct output, we can run the following comparison,
# creating a numpy tensor (which is not possible for more than two dimensions in cvxpy):
# rs = np.random.RandomState(seed=1)
# tensor = rs.randn(785, 3, 4)
# Now the implicit bias objective is:
# norm_1 = np.sum(np.linalg.norm(tensor, axis=(0,2)))
# norm_1: 168.4071
# norm_2 = get_implicit_bias_objective(tensor.reshape(785, 12).T, 3, 4)
# norm_2: 168.4024

def get_acc(var, data, constraints):
    if data['y'].shape[0]==0:
        return 1.
    y_hat = predict(var, data, constraints).value
    y_hat = np.sign(y_hat)
    return (data['y'] == y_hat).astype(float).mean()

def get_contextwise_acc(var, data, constraints):
    y_hat = predict(var, data, constraints).value
    y_hat = np.sign(y_hat)
    acc = (data['y'] == y_hat).astype(float)
    return data['context'].T@acc

def select_context(data, i):
    n_context = data['context'].shape[1]
    selector = (data['context']@np.linspace(0, n_context-1, n_context))==i
    print(selector)
    return {
        'x': data['x'][selector],
        'y': data['y'][selector],
        'context': data['context'][selector]
    }

def get_radius(arr):
    if arr.shape[0]==0.:
        return 0.
    return math.sqrt(mb.Miniball(arr).squared_radius())

def get_loee(var, train_data, val_data, prob, compute_radius=True):
    norm = np.sqrt((var.value**2).sum(axis=1))
    n_contexts = train_data['context'].shape[1]
    n_per_context_train = np.stack([
        sum(train_data['context']@np.linspace(0, n_contexts-1, n_contexts)==i)\
        for i in range(n_contexts)
    ])
    n_per_context_val = np.stack([
        sum(val_data['context']@np.linspace(0, n_contexts-1, n_contexts)==i)\
        for i in range(n_contexts)
    ])
    n_support_vectors = np.stack([
        sum(prob.constraints[0].dual_value[train_data['context']@np.linspace(0, n_contexts-1, n_contexts)==i]!=0.)\
        for i in range(n_contexts)
    ])
    train_acc = get_contextwise_acc(var, train_data, 'none')
    val_acc = get_contextwise_acc(var, val_data, 'none')
    df = pd.DataFrame({
        'norm': norm,
        'n_train': n_per_context_train,
        'n_val': n_per_context_val,
        'n_supp': n_support_vectors,
        'train_acc': train_acc,
        'val_acc': val_acc
    })
    if compute_radius:
        df['radius'] = np.stack([
            get_radius(train_data['x'][train_data['context']@np.linspace(0, n_contexts-1, n_contexts)==i])\
            for i in range(n_contexts)
        ])
    return df

def train_convex(constraints, train_data, val_data, max_iters=200, comparison_model=None):
    n_data, n_latent, n_contexts = train_data['context'].shape
    if constraints == 'none':
        train_data = collapse_contexts(train_data)
        val_data = collapse_contexts(val_data)
    _, dim_x = train_data['x'].shape
    if constraints == 'none':
        var = cp.Variable((n_contexts**n_latent, dim_x))
    else:
        var = cp.Variable((n_latent*n_contexts, dim_x))
    _constraints = get_constraints(var, train_data, constraints)
    objective = {
        'none': get_none_objective,
        'architecture': get_architecture_objective,
        'implicit_bias': get_implicit_bias_objective
    }[constraints](var, n_latent, n_contexts)
    prob = cp.Problem(objective, _constraints)
    method = {
        'none': 'OSQP',
        'architecture': 'OSQP',
        'implicit_bias': 'ECOS'
    }[constraints]
    kwargs = {
        'none': {},
        'architecture': {},
        'implicit_bias': {'max_iters': max_iters}
    }[constraints]
    prob.solve(solver=method, verbose=True, **kwargs)
    train_acc = get_acc(var, train_data, constraints)
    val_acc = get_acc(var, val_data, constraints)
    accs = [train_acc, val_acc]
    if comparison_model is not None:
        y_hat = np.sign(predict(var, val_data, constraints).value)
        with torch.no_grad():
            y_comp = np.sign(comparison_model(torch.from_numpy(val_data['x']))[:,0].numpy())
        accs.append((y_hat==y_comp).astype(float).mean())
    return accs, [var, train_data, val_data, prob]
