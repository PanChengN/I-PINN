'''
@Project ：Improved physics-informed neural network in mitigating gradient-related failur 
@File    ：Klein_Gordon_different_networks.py
@IDE     ：PyCharm 
@Author  ：Pancheng Niu
@Date    ：2024/7/18 下午3:51 
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

    lam_threshold = 1000000

    # Save relative error.
    results = []
    for depth in [3, 5, 7]:
        for widths in [30, 50, 70, 90]:
            layer = [2] + [widths] * depth + [1]
            layer_str = f'{depth}x{widths}'
            for mode in ['PINN', 'IA_PINN', 'IAW_PINN', 'I_PINN']:
                model = Klein_Gordon(layer, operator, ics_sampler, bcs_sampler, res_sampler, alpha, beta, gamma, k,
                                     mode, lam_threshold)
                model.train(nIter=40000, batch_size=128)
                # test data
                nn = 100
                t = np.linspace(dom_coords[0, 0], dom_coords[1, 0], nn)[:, None]
                x = np.linspace(dom_coords[0, 1], dom_coords[1, 1], nn)[:, None]
                t, x = np.meshgrid(t, x)
                X_star = np.hstack((t.flatten()[:, None], x.flatten()[:, None]))

                u_star = u(X_star)

                u_pred = model.predict_u(X_star)
                u_pred = u_pred.detach().cpu().numpy()

                error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
                print('Relative L2 error_u: {:.2e}'.format(error_u))

                results.append({
                    'model_type': mode,
                    'network_structure': layer_str,
                    'relative_error': error_u
                })

                U_star = griddata(X_star, u_star.flatten(), (t, x), method='cubic')
                U_pred = griddata(X_star, u_pred.flatten(), (t, x), method='cubic')

                fig_1 = plt.figure(1, figsize=(18, 5))
                plt.subplot(1, 3, 1)
                plt.imshow(U_star, extent=(t.min(), t.max(), x.min(), x.max()), origin='lower', cmap='jet',
                           aspect='auto')
                plt.colorbar()
                plt.xlabel('$t$')
                plt.ylabel('$x$')
                plt.title('Exact u(x)')

                plt.subplot(1, 3, 2)
                plt.imshow(U_pred, extent=(t.min(), t.max(), x.min(), x.max()), origin='lower', cmap='jet',
                           aspect='auto')
                plt.colorbar()
                plt.xlabel('$t$')
                plt.ylabel('$x$')
                plt.title('Predicted u(x)')

                plt.subplot(1, 3, 3)
                plt.imshow(np.abs(U_star - U_pred), extent=(t.min(), t.max(), x.min(), x.max()), origin='lower', cmap='jet',
                           aspect='auto')
                plt.colorbar()
                plt.xlabel('$t$')
                plt.ylabel('$x$')
                plt.title('Absolute error')
                plt.tight_layout()
                plt.savefig(f'./pic_w_{lam_threshold}/{mode}_{layer_str}_Exact_Predicted_Absolute_error.pdf', dpi=300, bbox_inches='tight')
                plt.close(fig_1)
                # plt.show()

                # loss_plotting
                loss_r = model.loss_res_log
                loss_b = model.loss_bcs_log
                loss_ic = model.loss_ics_log

                fig_2 = plt.figure(2)
                ax = fig_2.add_subplot(1, 1, 1)
                ax.plot(loss_r, label='$\mathcal{L}_{r}$')
                ax.plot(loss_b, label='$\mathcal{L}_{b}$')
                ax.plot(loss_ic, label='$\mathcal{L}_{ic}$')
                ax.set_yscale('log')
                ax.set_xlabel('iterations')
                ax.set_ylabel('Loss')
                plt.legend()
                plt.tight_layout()
                plt.savefig(f'./pic_w_{lam_threshold}/{mode}_{layer_str}_loss.pdf', dpi=300, bbox_inches='tight')
                plt.close(fig_2)
                # plt.show()

                lam_r = model.lam_r_log
                lam_b = model.lam_b_log
                lam_i = model.lam_i_log
                fig_3 = plt.figure(3)
                ax = fig_3.add_subplot(1, 1, 1)
                ax.plot(lam_r, label='$\lambda_{r}$')
                ax.plot(lam_b, label='$\lambda_{b}$')
                ax.plot(lam_i, label='$\lambda_{i}$')
                ax.set_yscale('log')
                ax.set_xlabel('iterations')
                ax.set_ylabel('$\lambda$')
                plt.legend()
                plt.tight_layout()

                plt.savefig(f'./pic_w_{lam_threshold}/{mode}_{layer_str}_lam_log.pdf', dpi=300, bbox_inches='tight')
                plt.close(fig_3)
                # plt.show()
            print("=======================================================================")

    # Relative error folder path
    save_dir = f'./pic_w_{lam_threshold}/'
    # Save relative error data to a CSV file.
    csv_file = os.path.join(save_dir, 'relative_errors100.csv')
    with open(csv_file, 'w', newline='') as csvfile:
        fieldnames = ['model_type', 'network_structure', 'relative_error']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)