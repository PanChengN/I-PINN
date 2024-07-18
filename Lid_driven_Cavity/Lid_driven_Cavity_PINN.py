'''
@Project ：Improved physics-informed neural network in mitigating gradient-related failur 
@File    ：Lid_driven_Cavity_PINN.py
@IDE     ：PyCharm 
@Author  ：Pancheng Niu
@Date    ：2024/7/18 下午3:41 
'''
from DNN import Net, Net_attention
import torch
import numpy as np
import timeit
import torch.optim
import torch.optim.lr_scheduler as lr_scheduler
import random
from tqdm import trange


def seed_torch(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


seed_torch(2341)

torch.cuda.empty_cache()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Current device:", device)


class Sampler:
    def __init__(self, dim, coords, func, name=None):
        self.dim = dim
        self.coords = coords
        self.func = func
        self.name = name

    def sample(self, N):
        x = self.coords[0:1, :] + (self.coords[1:2, :] - self.coords[0:1, :]) * np.random.rand(N, self.dim)
        y = self.func(x)
        return torch.tensor(x, requires_grad=True).float().to(device), torch.tensor(y, requires_grad=True).float().to(
            device)


class Navier_Stokes2D:
    def __init__(self, model_type, layers, bcs_sampler, res_sampler, operator, Re, lam_threshold):

        X, _ = res_sampler.sample(np.int32(1e5))
        self.mu_X = X.mean(0).detach()
        self.sigma_X = X.std(0).detach()
        self.sigma_x = self.sigma_X[0]
        self.sigma_y = self.sigma_X[1]

        self.model_type = model_type
        self.layers = layers

        self.operator = operator
        self.bcs_sampler = bcs_sampler
        self.res_sampler = res_sampler

        self.Re = torch.tensor(Re, dtype=torch.float32)

        if self.model_type in ['PINN', 'IAW_PINN']:
            self.dnn = Net(layers).to(device)
        elif self.model_type in ['IA_PINN', 'I_PINN']:
            self.dnn = Net_attention(layers).to(device)

        self.sigma_r = torch.tensor([1.], requires_grad=True).float().to(device)
        self.sigma_u = torch.tensor([1.], requires_grad=True).float().to(device)
        self.sigma_r = torch.nn.Parameter(self.sigma_r)
        self.sigma_u = torch.nn.Parameter(self.sigma_u)

        self.optimizer_1 = torch.optim.Adam(self.dnn.parameters(), lr=1e-3)
        self.optimizer_2 = torch.optim.Adam([self.sigma_r] + [self.sigma_u], lr=1e-3)
        self.scheduler = lr_scheduler.ExponentialLR(self.optimizer_1, gamma=0.9)
        self.iter = 0
        self.lam_threshold = lam_threshold

        self.loss_ics_log = []
        self.loss_bcs_log = []
        self.loss_res_log = []
        self.loss_log = []
        self.lam_r_log = []
        self.lam_u_log = []

    def d(self, f, x):
        return torch.autograd.grad(f, x, grad_outputs=torch.ones_like(f), create_graph=True, only_inputs=True)[0]

    def net_u(self, x1, x2):
        u = self.dnn(torch.cat([x1, x2], 1))
        return u

    def net_psi_p(self, x, y):
        psi_p = self.dnn(torch.cat([x, y], 1))
        psi = psi_p[:, 0:1]
        p = psi_p[:, 1:2]
        return psi, p

    def net_uv(self, x, y):
        psi, p = self.net_psi_p(x, y)
        u = self.d(psi, y) / self.sigma_y
        v = - self.d(psi, x) / self.sigma_x
        return u, v

    def net_r(self, x, y):
        psi, p = self.net_psi_p(x, y)
        u_momentum_pred, v_momentum_pred = self.operator(psi, p, x, y,
                                                         self.Re,
                                                         self.sigma_x, self.sigma_y)

        return u_momentum_pred, v_momentum_pred

    def fetch_minibatch(self, sampler, batch_size):
        X, Y = sampler.sample(batch_size)
        X = (X - self.mu_X) / self.sigma_X
        return X, Y

    def epoch_train(self, batch_size):
        X_bc1_batch, u_bc1_batch = self.fetch_minibatch(self.bcs_sampler[0], batch_size)
        X_bc2_batch, u_bc2_batch = self.fetch_minibatch(self.bcs_sampler[1], batch_size)
        X_bc3_batch, u_bc3_batch = self.fetch_minibatch(self.bcs_sampler[2], batch_size)
        X_bc4_batch, u_bc4_batch = self.fetch_minibatch(self.bcs_sampler[3], batch_size)
        X_res_batch, f_res_batch = self.fetch_minibatch(self.res_sampler, batch_size)

        u_bc1_pred, v_bc1_pred = self.net_uv(X_bc1_batch[:, 0:1], X_bc1_batch[:, 1:2])
        u_bc2_pred, v_bc2_pred = self.net_uv(X_bc2_batch[:, 0:1], X_bc2_batch[:, 1:2])
        u_bc3_pred, v_bc3_pred = self.net_uv(X_bc3_batch[:, 0:1], X_bc3_batch[:, 1:2])
        u_bc4_pred, v_bc4_pred = self.net_uv(X_bc4_batch[:, 0:1], X_bc4_batch[:, 1:2])

        U_bc1_pred = torch.cat((u_bc1_pred, v_bc1_pred), dim=1)
        U_bc2_pred = torch.cat((u_bc2_pred, v_bc2_pred), dim=1)
        U_bc3_pred = torch.cat((u_bc3_pred, v_bc3_pred), dim=1)
        U_bc4_pred = torch.cat((u_bc4_pred, v_bc4_pred), dim=1)

        u_momentum_pred, v_momentum_pred = self.net_r(X_res_batch[:, 0:1], X_res_batch[:, 1:2])

        loss_u_momentum = torch.mean(torch.square(u_momentum_pred))
        loss_v_momentum = torch.mean(torch.square(v_momentum_pred))
        loss_res = loss_u_momentum + loss_v_momentum

        loss_bcs = torch.mean(torch.square(U_bc1_pred - u_bc1_batch) +
                              torch.square(U_bc2_pred) +
                              torch.square(U_bc3_pred) +
                              torch.square(U_bc4_pred))
        return loss_res, loss_bcs

    def loss_fun(self, loss_bcs, loss_res):
        loss = loss_bcs + loss_res
        return loss

    def AW_loss_fun(self, loss_bcs, loss_res):
        loss = 1. / (self.sigma_u ** 2 + 1. / self.lam_threshold) * loss_bcs + 1. / (
                self.sigma_r ** 2 + 1. / self.lam_threshold) * loss_res + torch.log(
            self.sigma_u ** 2 + 1. / self.lam_threshold) + torch.log(self.sigma_r ** 2 + 1. / self.lam_threshold)
        return loss

    def train(self, nIter=10000, batch_size=128):
        start_time = timeit.default_timer()
        print(f"model: {self.model_type}, layer: {self.layers}")
        self.dnn.train()
        pbar = trange(nIter, ncols=140)
        for it in pbar:
            loss_bcs, loss_res = self.epoch_train(batch_size)
            self.optimizer_1.zero_grad()
            if self.model_type in ['PINN', 'IA_PINN']:
                loss = self.loss_fun(loss_bcs, loss_res)
                loss.backward()
            elif self.model_type in ['IAW_PINN', 'I_PINN']:
                self.optimizer_2.zero_grad()
                loss = self.AW_loss_fun(loss_bcs, loss_res)
                loss.backward()
                self.optimizer_2.step()
            self.optimizer_1.step()

            true_loss = loss_res + loss_bcs
            self.loss_bcs_log.append(loss_bcs.item())
            self.loss_res_log.append(loss_res.item())
            self.loss_log.append(true_loss.item())

            if self.iter % 1000 == 0:
                self.scheduler.step()
            if it % 100 == 0:
                if self.model_type in ['PINN', 'IA_PINN']:
                    pbar.set_postfix({'Iter': self.iter,
                                      'Loss': '{0:.3e}'.format(true_loss.item()),
                                      'lam_r': '{0:.2f}'.format(1),
                                      'lam_u': '{0:.2f}'.format(1)
                                      })
                    self.lam_r_log.append(1. / (self.sigma_r ** 2).item())
                    self.lam_u_log.append(1. / (self.sigma_u ** 2).item())
                elif self.model_type in ['IAW_PINN', 'I_PINN']:
                    pbar.set_postfix({'Iter': self.iter,
                                      'Loss': '{0:.3e}'.format(true_loss.item()),
                                      'lam_r': '{0:.2f}'.format(
                                          1. / (self.sigma_r ** 2 + 1. / self.lam_threshold).item()),
                                      'lam_u': '{0:.2f}'.format(
                                          1. / (self.sigma_u ** 2 + 1. / self.lam_threshold).item())
                                      })
                    self.lam_r_log.append(1. / (self.sigma_r ** 2 + 1. / self.lam_threshold).item())
                    self.lam_u_log.append(1. / (self.sigma_u ** 2 + 1. / self.lam_threshold).item())
            self.iter += 1
        elapsed = timeit.default_timer() - start_time
        print("Time: {0:.2f}s".format(elapsed))

    def predict_psi_p(self, X_star):
        X_star = torch.tensor(X_star, requires_grad=True).float().to(device)
        X_star = (X_star - self.mu_X) / self.sigma_X
        psi_star, p_star = self.net_psi_p(X_star[:, 0:1], X_star[:, 1:2])
        return psi_star, p_star

    def predict_uv(self, X_star):
        X_star = torch.tensor(X_star, requires_grad=True).float().to(device)
        X_star = (X_star - self.mu_X) / self.sigma_X
        u_star, v_star = self.net_uv(X_star[:, 0:1], X_star[:, 1:2])
        return u_star, v_star
