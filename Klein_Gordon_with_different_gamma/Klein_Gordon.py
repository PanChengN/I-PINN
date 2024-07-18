'''
@Project ：Improved physics-informed neural network in mitigating gradient-related failur 
@File    ：Klein_Gordon.py.py
@IDE     ：PyCharm 
@Author  ：Pancheng Niu
@Date    ：2024/7/18 下午3:55 
'''
import numpy as np
import csv
import torch
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from Klein_Gordon_PINN import Klein_Gordon, Sampler

if __name__ == '__main__':
    def u(x):
        return x[:, 1:2] * np.cos(5 * np.pi * x[:, 0:1]) + (x[:, 0:1] * x[:, 1:2]) ** 3


    def u_tt(x):
        return - 25 * np.pi ** 2 * x[:, 1:2] * np.cos(5 * np.pi * x[:, 0:1]) + 6 * x[:, 0:1] * x[:, 1:2] ** 3


    def u_xx(x):
        return np.zeros((x.shape[0], 1)) + 6 * x[:, 1:2] * x[:, 0:1] ** 3


    def f(x, alpha, beta, gamma, k):
        return u_tt(x) + alpha * u_xx(x) + beta * u(x) + gamma * u(x) ** k


    def d(f, x):
        return torch.autograd.grad(f, x, grad_outputs=torch.ones_like(f), create_graph=True, only_inputs=True)[0]


    def operator(u, t, x, alpha, beta, gamma, k, sigma_t=1.0, sigma_x=1.0):
        u_t = d(u, t) / sigma_t
        u_x = d(u, x) / sigma_x
        u_tt = d(u_t, t) / sigma_t
        u_xx = d(u_x, x) / sigma_x
        residual = u_tt + alpha * u_xx + beta * u + gamma * u ** k
        return residual


    alpha = -1.0
    beta = 0.0
    gamma = 1.0
    k = 3

    ics_coords = np.array([[0.0, 0.0],
                           [0.0, 1.0]])
    bc1_coords = np.array([[0.0, 0.0],
                           [1.0, 0.0]])
    bc2_coords = np.array([[0.0, 1.0],
                           [1.0, 1.0]])
    dom_coords = np.array([[0.0, 0.0],
                           [1.0, 1.0]])

    ics_sampler = Sampler(2, ics_coords, lambda x: u(x), name='Initial Condition 1')

    bc1 = Sampler(2, bc1_coords, lambda x: u(x), name='Dirichlet BC1')
    bc2 = Sampler(2, bc2_coords, lambda x: u(x), name='Dirichlet BC2')
    bcs_sampler = [bc1, bc2]

    res_sampler = Sampler(2, dom_coords, lambda x: f(x, alpha, beta, gamma, k), name='Forcing')
    results = []
    error_list = []
    loss_list = []
    lam_threshold_list = [1, 10, 100, 1000, 10000, 100000, 1000000]

    for lam_threshold in lam_threshold_list:
        for depth in [7]:
            for widths in [50]:
                layer = [2] + [widths] * depth + [1]
                layer_str = f'{depth}x{widths}'
                for mode in ['I_PINN']:
                    model = Klein_Gordon(layer, operator, ics_sampler, bcs_sampler, res_sampler, alpha, beta, gamma, k,
                                         mode, lam_threshold)
                    model.train(nIter=40000, batch_size=128)
                    # Test data
                    nn = 100
                    t = np.linspace(dom_coords[0, 0], dom_coords[1, 0], nn)[:, None]
                    x = np.linspace(dom_coords[0, 1], dom_coords[1, 1], nn)[:, None]
                    t, x = np.meshgrid(t, x)
                    X_star = np.hstack((t.flatten()[:, None], x.flatten()[:, None]))

                    # Exact solution
                    u_star = u(X_star)
                    # Predicted solution
                    u_pred = model.predict_u(X_star)
                    u_pred = u_pred.detach().cpu().numpy()

                    error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
                    print('Relative L2 error_u: {:.2e}'.format(error_u))
                    error_list.append((lam_threshold, error_u))
                    results.append({
                        'model_type': mode,
                        'network_structure': layer_str,
                        'relative_error': error_u
                    })

                    # Test data
                    U_star = griddata(X_star, u_star.flatten(), (t, x), method='cubic')
                    U_pred = griddata(X_star, u_pred.flatten(), (t, x), method='cubic')

                    # Record loss values
                    loss_list.append((lam_threshold, model.loss_log))

                    # Save the loss logs corresponding to each lam_threshold.
                    with open(f'./pic/{mode}_{layer_str}_loss_log_lam_threshold_{lam_threshold}.csv', mode='w',
                              newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow(['Iteration', 'Loss'])  #  Write header
                        writer.writerows(enumerate(model.loss_log))  # Write data

    # Save relative error
    with open(f'./pic/{mode}_{layer_str}_relative_L2_error_vs_lam_threshold.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['lam_threshold', 'Relative L2 error'])  #  Write header
        writer.writerows(error_list)  # Write data