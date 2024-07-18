'''
@Project ：Improved physics-informed neural network in mitigating gradient-related failur 
@File    ：Helmholtz2D_K.py
@IDE     ：PyCharm 
@Author  ：Pancheng Niu
@Date    ：2024/7/17 下午6:41 
'''
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from Helmholtz2D_PINN import Helmholtz2D_PINN, Sampler
import csv
import os

if __name__ == '__main__':

    a_1 = 1
    a_2 = 4


    def u(x, a_1, a_2):
        return np.sin(a_1 * np.pi * x[:, 0:1]) * np.sin(a_2 * np.pi * x[:, 1:2])


    def u_xx(x, a_1, a_2):
        return - (a_1 * np.pi) ** 2 * np.sin(a_1 * np.pi * x[:, 0:1]) * np.sin(a_2 * np.pi * x[:, 1:2])


    def u_yy(x, a_1, a_2):
        return - (a_2 * np.pi) ** 2 * np.sin(a_1 * np.pi * x[:, 0:1]) * np.sin(a_2 * np.pi * x[:, 1:2])


    def f(x, a_1, a_2, lam):
        return u_xx(x, a_1, a_2) + u_yy(x, a_1, a_2) + lam * u(x, a_1, a_2)


    def d(f, x):
        return torch.autograd.grad(f, x, grad_outputs=torch.ones_like(f), create_graph=True, only_inputs=True)[0]


    def operator(u, x1, x2, lam, sigma_x1=1.0, sigma_x2=1.0):
        u_x1 = d(u, x1) / sigma_x1
        u_x2 = d(u, x2) / sigma_x2
        u_xx1 = d(u_x1, x1) / sigma_x1
        u_xx2 = d(u_x2, x2) / sigma_x2
        residual = u_xx1 + u_xx2 + lam * u
        return residual


    lam = 1.0 ** 2

    bc1_coords = np.array([[-1.0, -1.0],
                           [1.0, -1.0]])
    bc2_coords = np.array([[1.0, -1.0],
                           [1.0, 1.0]])
    bc3_coords = np.array([[1.0, 1.0],
                           [-1.0, 1.0]])
    bc4_coords = np.array([[-1.0, 1.0],
                           [-1.0, -1.0]])
    dom_coords = np.array([[-1.0, -1.0],
                           [1.0, 1.0]])

    ics_sampler = None

    bc1 = Sampler(2, bc1_coords, lambda x: u(x, a_1, a_2), name='Dirichlet BC1')
    bc2 = Sampler(2, bc2_coords, lambda x: u(x, a_1, a_2), name='Dirichlet BC2')
    bc3 = Sampler(2, bc3_coords, lambda x: u(x, a_1, a_2), name='Dirichlet BC3')
    bc4 = Sampler(2, bc4_coords, lambda x: u(x, a_1, a_2), name='Dirichlet BC4')

    bcs_sampler = [bc1, bc2, bc3, bc4]

    res_sampler = Sampler(2, dom_coords, lambda x: f(x, a_1, a_2, lam), name='Forcing')
    lam_threshold = 100
    # Calculate and print relative error.
    results = []
    for depth in [3, 5, 7]:
        for widths in [30, 50, 70, 90]:
            layer = [2] + [widths] * depth + [1]
            layer_str = f'{depth}x{widths}'
            for mode in ['PINN', 'IA_PINN', 'IAW_PINN', 'I_PINN']:
                model = Helmholtz2D_PINN(mode, layer, ics_sampler, bcs_sampler, res_sampler, operator, lam,
                                         lam_threshold)
                model.train(nIter=40, batch_size=128)

                # test data
                nn = 100
                x1 = np.linspace(dom_coords[0, 0], dom_coords[1, 0], nn)[:,None]
                x2 = np.linspace(dom_coords[0, 1], dom_coords[1, 1], nn)[:,None]
                x1, x2 = np.meshgrid(x1, x2)
                X_star = np.hstack((x1.flatten()[:, None], x2.flatten()[:, None]))

                u_star = u(X_star, a_1, a_2)
                f_star = f(X_star, a_1, a_2, lam)

                u_pred = model.predict_u(X_star)
                u_pred = u_pred.detach().cpu().numpy()

                error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
                print('Relative L2 error_u: {:.2e}'.format(error_u))

                results.append({
                    'model_type': mode,
                    'network_structure': layer_str,
                    'relative_error': error_u
                })

                ### plotting ###
                # Exact solution data_interpolation
                U_star = griddata(X_star, u_star.flatten(), (x1, x2), method='cubic')
                # Predicted solution data_interpolation
                U_pred = griddata(X_star, u_pred.flatten(), (x1, x2), method='cubic')

                # ======================== fig 1 ==========================================
                fig_1 = plt.figure(1, figsize=(18, 5))
                plt.subplot(1, 3, 1)
                plt.imshow(U_star, extent=(x1.min(), x1.max(), x2.min(), x2.max()), origin='lower', cmap='jet',
                           aspect='auto')
                plt.colorbar()
                plt.xlabel(r'$x_1$')
                plt.ylabel(r'$x_2$')
                plt.title('Exact $u(x)$')

                plt.subplot(1, 3, 2)
                plt.imshow(U_pred, extent=(x1.min(), x1.max(), x2.min(), x2.max()), origin='lower', cmap='jet',
                           aspect='auto')
                plt.colorbar()
                plt.xlabel(r'$x_1$')
                plt.ylabel(r'$x_2$')
                plt.title('Predicted $u(x)$')

                plt.subplot(1, 3, 3)
                plt.imshow( np.abs(U_star - U_pred), extent=(x1.min(), x1.max(), x2.min(), x2.max()), origin='lower', cmap='jet',
                           aspect='auto')
                plt.colorbar()
                plt.xlabel(r'$x_1$')
                plt.ylabel(r'$x_2$')
                plt.title('Absolute error')
                plt.tight_layout()
                plt.savefig(f'./pic_k=1_w_{lam_threshold}/{mode}_{layer_str}_Exact_Predicted_Absolute_error.pdf', dpi=300, bbox_inches='tight')
                plt.close(fig_1)

                # ============================== fig 2 ==========================================

                loss_res = model.loss_res_log
                loss_bcs = model.loss_bcs_log
                fig_2 = plt.figure(2)
                ax = fig_2.add_subplot(1, 1, 1)
                ax.plot(loss_res, label='$\mathcal{Lid_driven_Cavity}_{r}$')
                ax.plot(loss_bcs, label='$\mathcal{Lid_driven_Cavity}_{b}$')
                ax.set_yscale('log')
                ax.set_xlabel('iterations')
                ax.set_ylabel('Loss')
                plt.legend()
                plt.tight_layout()
                plt.savefig(f'./pic_k=1_w_{lam_threshold}/{mode}_{layer_str}_loss.pdf', dpi=300, bbox_inches='tight')
                plt.close(fig_2)

                # ============================== fig 3 ==========================================

                lam_r = model.lam_r_log
                lam_u = model.lam_u_log
                fig_3 = plt.figure(3)
                ax = fig_3.add_subplot(1, 1, 1)
                ax.plot(lam_r, label='$\lambda_{r}$')
                ax.plot(lam_u, label='$\lambda_{b}$')
                ax.set_yscale('log')
                ax.set_xlabel('iterations')
                ax.set_ylabel('$\lambda$')
                plt.legend()
                plt.tight_layout()
                plt.savefig(f'./pic_k=1_w_{lam_threshold}/{mode}_{layer_str}_lam_log.pdf', dpi=300, bbox_inches='tight')
                plt.close(fig_3)
            print("===============================================================================================")

    # Save relative error.
    save_dir = f'./pic_k=1_w_{lam_threshold}/'
    # Save relative error data to a CSV file.
    csv_file = os.path.join(save_dir, 'relative_errors100.csv')
    with open(csv_file, 'w', newline='') as csvfile:
        fieldnames = ['model_type', 'network_structure', 'relative_error']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)