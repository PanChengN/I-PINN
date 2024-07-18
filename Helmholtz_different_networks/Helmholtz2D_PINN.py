'''
@Project ：Improved physics-informed neural network in mitigating gradient-related failur 
@File    ：Helmholtz2D_PINN_K.py
@IDE     ：PyCharm 
@Author  ：Pancheng Niu
@Date    ：2024/7/17 下午6:42 
'''
from DNN import Net, Net_attention
import torch
import torch.optim
import torch.optim.lr_scheduler as lr_scheduler
import random
import timeit
import numpy as np
from tqdm import trange

def seed_torch(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
seed_torch(1234)
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


class Helmholtz2D_PINN:

    def __init__(self, model_type, layers, ics_sampler, bcs_sampler, res_sampler, operator, lam, lam_threshold):

        X, _ = res_sampler.sample(np.int32(1e5))
        self.mu_X = X.mean(0).detach()
        self.sigma_X = X.std(0).detach()
        self.sigma_x1 = self.sigma_X[0]
        self.sigma_x2 = self.sigma_X[1]

        self.model_type = model_type
        self.layers = layers

        self.operator = operator
        self.ics_sampler = ics_sampler
        self.bcs_sampler = bcs_sampler
        self.res_sampler = res_sampler

        self.lambda_k = torch.tensor(lam, dtype=torch.float32)

        self.sigma_r = torch.tensor([1.], requires_grad=True).float().to(device)
        self.sigma_u = torch.tensor([1.], requires_grad=True).float().to(device)
        self.sigma_r = torch.nn.Parameter(self.sigma_r)
        self.sigma_u = torch.nn.Parameter(self.sigma_u)

        if self.model_type in ['PINN', 'IAW_PINN']:
            self.dnn = Net(layers).to(device)
        elif self.model_type in ['IA_PINN', 'I_PINN']:
            self.dnn = Net_attention(layers).to(device)

        self.optimizer_1 = torch.optim.Adam(self.dnn.parameters(), lr=1e-3)
        self.optimizer_2 = torch.optim.Adam([self.sigma_r] + [self.sigma_u], lr=1e-3)
        self.scheduler = lr_scheduler.ExponentialLR(self.optimizer_1, gamma=0.9)
        self.iter = 0
        self.lam_threshold = lam_threshold

        self.loss_bcs_log = []
        self.loss_res_log = []
        self.loss_log = []
        self.lam_r_log = []
        self.lam_u_log = []


    def net_u(self, x1, x2):
        u = self.dnn(torch.cat([x1, x2], 1))
        return u

    def net_r(self, x1, x2):
        u = self.net_u(x1, x2)
        residual = self.operator(u, x1, x2,
                                 self.lambda_k,
                                 self.sigma_x1,
                                 self.sigma_x2)
        return residual

    def fetch_minibatch(self, sampler, batch_size):
        X, Y = sampler.sample(batch_size)
        X = (X - self.mu_X) / self.sigma_X
        return X, Y

    def epoch_train(self, batch_size=128):
        X_bc1_batch, u_bc1_batch = self.fetch_minibatch(self.bcs_sampler[0], batch_size)
        X_bc2_batch, u_bc2_batch = self.fetch_minibatch(self.bcs_sampler[1], batch_size)
        X_bc3_batch, u_bc3_batch = self.fetch_minibatch(self.bcs_sampler[2], batch_size)
        X_bc4_batch, u_bc4_batch = self.fetch_minibatch(self.bcs_sampler[3], batch_size)
        X_res_batch, f_res_batch = self.fetch_minibatch(self.res_sampler, batch_size)

        u_bc1_pred = self.net_u(X_bc1_batch[:, 0:1], X_bc1_batch[:, 1:2])
        u_bc2_pred = self.net_u(X_bc2_batch[:, 0:1], X_bc2_batch[:, 1:2])
        u_bc3_pred = self.net_u(X_bc3_batch[:, 0:1], X_bc3_batch[:, 1:2])
        u_bc4_pred = self.net_u(X_bc4_batch[:, 0:1], X_bc4_batch[:, 1:2])
        r_pred = self.net_r(X_res_batch[:, 0:1], X_res_batch[:, 1:2])

        loss_bc1 = torch.mean(torch.square(u_bc1_pred - u_bc1_batch))
        loss_bc2 = torch.mean(torch.square(u_bc2_pred - u_bc2_batch))
        loss_bc3 = torch.mean(torch.square(u_bc3_pred - u_bc2_batch))
        loss_bc4 = torch.mean(torch.square(u_bc4_pred - u_bc2_batch))

        loss_bcs = loss_bc1 + loss_bc2 + loss_bc3 + loss_bc4
        loss_res = torch.mean(torch.square(r_pred - f_res_batch))

        return loss_bcs, loss_res

    def loss_fun(self, loss_bcs, loss_res):
        loss = loss_bcs+loss_res
        return loss

    def AW_loss_fun(self, loss_bcs, loss_res):
        loss = 1. / (self.sigma_u ** 2+ 1./self.lam_threshold) * loss_bcs + 1. / (self.sigma_r ** 2+ 1./self.lam_threshold) * loss_res + torch.log(
            self.sigma_u ** 2 + 1./self.lam_threshold) + torch.log(self.sigma_r ** 2 + 1./self.lam_threshold)
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
            self.lam_r_log.append(1. / (self.sigma_r ** 2+ 1./self.lam_threshold).item())
            self.lam_u_log.append(1. / (self.sigma_u ** 2+ 1./self.lam_threshold).item())

            if self.iter % 1000 == 0:
                self.scheduler.step()
            if it % 100 == 0:
                pbar.set_postfix({'Iter': self.iter,
                                  'Loss': '{0:.3e}'.format(true_loss.item()),
                                  'lam_r': '{0:.2f}'.format(1. / (self.sigma_r ** 2+ 1./self.lam_threshold).item()),
                                  'lam_u': '{0:.2f}'.format(1. / (self.sigma_u ** 2+ 1./self.lam_threshold).item())
                                  })
            self.iter += 1
        elapsed = timeit.default_timer() - start_time
        print("Time: {0:.2f}s".format(elapsed))

    def predict_u(self, X_star):
        X_star = torch.tensor(X_star, requires_grad=True).float().to(device)
        X_star = (X_star - self.mu_X) / self.sigma_X
        self.dnn.eval()
        u_pred = self.net_u(X_star[:, 0:1], X_star[:, 1:2])
        return u_pred

    def predict_r(self, X_star):
        X_star = torch.tensor(X_star, requires_grad=True).float().to(device)
        X_star = (X_star - self.mu_X) / self.sigma_X
        self.dnn.eval()
        r_pred = self.net_r(X_star[:, 0:1], X_star[:, 1:2])
        return r_pred
