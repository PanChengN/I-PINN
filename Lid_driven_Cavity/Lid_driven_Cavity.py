'''
@Project ：Improved physics-informed neural network in mitigating gradient-related failur 
@File    ：Lid_driven_Cavity.py
@IDE     ：PyCharm 
@Author  ：Pancheng Niu
@Date    ：2024/7/18 下午3:43 
'''
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from Lid_driven_Cavity_PINN import Navier_Stokes2D, Sampler
import os
import csv

if __name__ == '__main__':
    def U_gamma_1(x):
        num = x.shape[0]
        return np.tile(np.array([1.0, 0.0]), (num, 1))


    def U_gamma_2(x):
        num = x.shape[0]
        return np.zeros((num, 2))


    def f(x):
        num = x.shape[0]
        return np.zeros((num, 2))


    def d(f, x):
        return torch.autograd.grad(f, x, grad_outputs=torch.ones_like(f), create_graph=True, only_inputs=True)[0]


    def operator(psi, p, x, y, Re, sigma_x=1.0, sigma_y=1.0):
        u = d(psi, y) / sigma_y
        v = - d(psi, x) / sigma_x

        u_x = d(u, x) / sigma_x
        u_y = d(u, y) / sigma_y

        v_x = d(v, x) / sigma_x
        v_y = d(v, y) / sigma_y

        p_x = d(p, x) / sigma_x
        p_y = d(p, y) / sigma_y

        u_xx = d(u_x, x) / sigma_x
        u_yy = d(u_y, y) / sigma_y

        v_xx = d(v_x, x) / sigma_x
        v_yy = d(v_y, y) / sigma_y

        Ru_momentum = u * u_x + v * u_y + p_x - (u_xx + u_yy) / Re
        Rv_momentum = u * v_x + v * v_y + p_y - (v_xx + v_yy) / Re

        return Ru_momentum, Rv_momentum


    Re = 100.0

    bc1_coords = np.array([[0.0, 1.0],
                           [1.0, 1.0]])
    bc2_coords = np.array([[0.0, 0.0],
                           [0.0, 1.0]])
    bc3_coords = np.array([[1.0, 0.0],
                           [1.0, 1.0]])
    bc4_coords = np.array([[0.0, 0.0],
                           [1.0, 0.0]])
    dom_coords = np.array([[0.0, 0.0],
                           [1.0, 1.0]])

    bc1 = Sampler(2, bc1_coords, lambda x: U_gamma_1(x), name='Dirichlet BC1')
    bc2 = Sampler(2, bc2_coords, lambda x: U_gamma_2(x), name='Dirichlet BC2')
    bc3 = Sampler(2, bc3_coords, lambda x: U_gamma_2(x), name='Dirichlet BC3')
    bc4 = Sampler(2, bc4_coords, lambda x: U_gamma_2(x), name='Dirichlet BC4')
    bcs_sampler = [bc1, bc2, bc3, bc4]

    res_sampler = Sampler(2, dom_coords, lambda x: f(x), name='Forcing')

    lam_threshold = 50

    # save relative error
    results = []
    for depth in [5]:
        for widths in [30]:
            layer = [2] + [widths] * depth + [2]
            layer_str = f'{depth}x{widths}'
            for mode in ['PINN','IA_PINN','IAW_PINN','I_PINN']:
                model = Navier_Stokes2D(mode, layer, bcs_sampler, res_sampler, operator, Re, lam_threshold)
                model.train(nIter=40000, batch_size=128)

                nx = 100
                ny = 100
                x = np.linspace(0.0, 1.0, nx)
                y = np.linspace(0.0, 1.0, ny)
                X, Y = np.meshgrid(x, y)

                X_star = np.hstack((X.flatten()[:, None], Y.flatten()[:, None]))

                psi_pred, p_pred = model.predict_psi_p(X_star)
                u_pred, v_pred = model.predict_uv(X_star)

                # Transform data format
                u_pred = u_pred.detach().cpu().numpy()
                v_pred = v_pred.detach().cpu().numpy()
                psi_pred = psi_pred.detach().cpu().numpy()
                p_pred = p_pred.detach().cpu().numpy()

                psi_star = griddata(X_star, psi_pred.flatten(), (X, Y), method='cubic')
                p_star = griddata(X_star, p_pred.flatten(), (X, Y), method='cubic')
                u_star = griddata(X_star, u_pred.flatten(), (X, Y), method='cubic')
                v_star = griddata(X_star, v_pred.flatten(), (X, Y), method='cubic')

                velocity = np.sqrt(u_pred ** 2 + v_pred ** 2)
                velocity_star = griddata(X_star, velocity.flatten(), (X, Y), method='cubic')

                u_ref = np.genfromtxt("reference_u.csv", delimiter=',')
                v_ref = np.genfromtxt("reference_v.csv", delimiter=',')
                velocity_ref = np.sqrt(u_ref ** 2 + v_ref ** 2)

                error = np.linalg.norm(velocity_star - velocity_ref.T, 2) / np.linalg.norm(velocity_ref, 2)
                print('l2 error: {:.2e}'.format(error))

                results.append({
                    'model_type': mode,
                    'network_structure': layer_str,
                    'relative_error': error
                })

                ### plotting ###
                # ============================== fig 1 ==========================================
                fig_1 = plt.figure(1, figsize=(18, 5))
                fig_1.add_subplot(1, 3, 1)
                plt.imshow(velocity_ref.T, extent=(X.min(), X.max(), Y.min(), Y.max()), origin='lower', cmap='jet',
                           aspect='auto')
                plt.colorbar()
                plt.xlabel('x')
                plt.ylabel('y')
                plt.title('Reference Velocity')
                fig_1.add_subplot(1, 3, 2)
                plt.imshow(velocity_star, extent=(x.min(), x.max(), Y.min(), Y.max()), origin='lower', cmap='jet',
                           aspect='auto')
                plt.colorbar()
                plt.xlabel('x')
                plt.ylabel('y')
                plt.title('Predicted Velocity')
                plt.tight_layout()
                fig_1.add_subplot(1, 3, 3)
                plt.imshow(np.abs(velocity_star - velocity_ref.T), extent=(X.min(), X.max(), Y.min(), Y.max()),
                           origin='lower', cmap='jet',
                           aspect='auto')
                plt.colorbar()
                plt.xlabel('x')
                plt.ylabel('y')
                plt.title('Absolute Error')
                plt.savefig(f'./pic{lam_threshold}/{mode}_{layer}_Exact_Predicted_Absolute_error.pdf', dpi=300,
                            bbox_inches='tight')
                plt.close(fig_1)
                # plt.show()

                # ============================== fig 2 ==========================================
                loss_res = model.loss_res_log
                loss_bcs = model.loss_bcs_log

                fig_2 = plt.figure(2)
                ax = fig_2.add_subplot(1, 1, 1)
                ax.plot(loss_res, label='$\mathcal{L}_{r}$')
                ax.plot(loss_bcs, label='$\mathcal{L}_{u_b}$')
                ax.set_yscale('log')
                ax.set_xlabel('iterations')
                ax.set_ylabel('Loss')
                plt.legend()
                plt.tight_layout()
                plt.savefig(f'./pic{lam_threshold}/{mode}_{layer}_loss.pdf', dpi=300, bbox_inches='tight')
                plt.close(fig_2)
                # plt.show()

                # ============================== fig 3 ==========================================

                lam_r = model.lam_r_log
                lam_u = model.lam_u_log
                fig_3 = plt.figure(3)
                ax = fig_3.add_subplot(1, 1, 1)
                ax.plot(lam_r, label='$\mathcal{w}_{r}$')
                ax.plot(lam_u, label='$\mathcal{w}_{u}$')
                ax.set_yscale('log')
                ax.set_xlabel('iterations')
                ax.set_ylabel('$W_i$')
                plt.legend()
                plt.tight_layout()
                plt.savefig(f'./pic{lam_threshold}/{mode}_{layer}_lam_log.pdf', dpi=300, bbox_inches='tight')
                plt.close(fig_3)
                # plt.show()

    save_dir = f'./pic{lam_threshold}/'
    csv_file = os.path.join(save_dir, 'relative_errors100.csv')
    with open(csv_file, 'w', newline='') as csvfile:
        fieldnames = ['model_type', 'network_structure', 'relative_error']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)
