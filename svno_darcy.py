import matplotlib.pyplot as plt
import math
import numpy as np
import torch
from tqdm import tqdm

from models import FNO2d
from train_utils.datasets import DarcyFlow
from train_utils.losses import LpLoss
from train_utils.utils import torch2dgrid


def load_data(data_config):
    dataset = DarcyFlow(data_config['datapath'],
                        nx=data_config['nx'],
                        sub=data_config['sub'],
                        offset=data_config['offset'],
                        num=data_config['n_sample'])
    
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=20, shuffle=True)
    return train_loader


def load_model(model_config):
    ckpt = torch.load('/central/groups/tensorlab/dllin/PINO-master/checkpoints/darcy-FDM/darcy-pretrain-pino-threshold.pt')
    
    model = FNO2d(modes1=model_config['modes1'],
                modes2=model_config['modes2'],
                fc_dim=model_config['fc_dim'],
                layers=model_config['layers'],
                act=model_config['act']).to(device)
    model.load_state_dict(ckpt['model'])
    return model


class RBF(torch.nn.Module):

    def __init__(self, sigma=1):
        super(RBF, self).__init__()
        self.sigma = sigma

    def forward(self, X, Y):
        n, s, _ = X.size()
        m, _, _ = Y.size()

        X_flat = X.view(n, s ** 2)
        Y_flat = Y.view(m, s ** 2)

        fnorms = torch.cdist(X_flat, Y_flat, p=2) / (s ** 2)

        return (-fnorms / (2 * self.sigma ** 2)).exp()


class SteinVariationNeuralOperator(torch.nn.Module):
    
    def __init__(self, model, s, A, u, device, sigma, epochs=10000):
        
        super(SteinVariationNeuralOperator, self).__init__()
        self.s = s
        self.model = model
        self.model.eval()
        self.A = A.to(device)
        self.A.requires_grad = True
        self.u = u.repeat(self.A.size(0), 1, 1).to(device)
        self.res = round(self.A.size(1) / self.u.size(1))
        self.device = device
        self.sigma = sigma
        self.epochs = epochs
        # define the mesh and mollifier for enforcing boundary conditions
        self.mesh = torch2dgrid(s, s).to(device)
        self.mollifier = torch.sin(np.pi * self.mesh[..., 0]) * torch.sin(np.pi * self.mesh[..., 1]) * 0.001
        self.mollifier = self.mollifier.to(device)
        # use RBF kernel
        self.K = RBF(sigma=sigma)
        self.myloss = LpLoss(size_average=True)
        self.optimizer = torch.optim.Adam([self.A], lr=0.1, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=2000, gamma=0.5)
    
    def phi(self):
        A_out = torch.zeros([self.A.size(0), self.s, self.s, 3], device=self.device)
        for i in range(self.A.size(0)):
            A_out[i] = self.get_a(self.darcy_mask1(self.A[i]))
        
        preds = self.model(A_out).reshape(self.A.size(0), self.s, self.s)
        preds = preds * self.mollifier
        preds = preds[::, ::self.res, ::self.res]
        
        log_prob = self.myloss(preds, self.u) + 0.01 * self.total_variance(self.A)
        log_prob.backward()
        score_func = -self.A.grad
        
        k_xx = self.K(self.A, self.A.detach())
        
        grad_k = -torch.autograd.grad(k_xx.sum(), self.A)[0]
        # grad_k = torch.zeros([self.A.size(0), self.s, self.s], device=self.device)
        # for i in range(self.A.size(0)):
        #     grad_k[i] = torch.autograd.grad(k_xx_sum[i], self.A, retain_graph=True)[0].sum(0)
        
        # kphi = torch.zeros([self.A.size(0), self.s, self.s], device=self.device)
        # for i in range(self.A.size(0)):
        #     for j in range(self.A.size(0)):
        #         kphi[i] += k_xx[i][j].detach() * score_func[j]
        kphi = torch.einsum('ij, j... -> i...', k_xx, score_func).to(self.device)

        return (kphi + grad_k)
    
    def step(self):
        self.A.grad = -self.phi()
        self.optimizer.step()
        
    def calculate_A_distribution(self):
        for _ in tqdm(range(self.epochs)):
            self.step()
            self.scheduler.step()
    
    def get_A_distribution(self):
        return self.darcy_mask2(self.A)
    
    """
    Additional methods for reformatting the input before passing into the model
    """
    def darcy_mask1(self, x):
        return 1 / (1 + torch.exp(-x)) * 9 + 3

    def darcy_mask2(self, x):
        x = 1 / (1 + torch.exp(-x))
        x[x>0.5] = 1
        x[x<=0.5] = 0
        return x * 9 + 3

    def total_variance(self, x):
        return torch.mean(torch.abs(x[...,:-1] - x[...,1:])) + torch.mean(torch.abs(x[...,:-1,:] - x[...,1:,:]))

    def get_a(self, x):
        return torch.cat([x.unsqueeze(2), self.mesh], dim=2).reshape(1, self.s, self.s, 3).to(self.device)
                

if __name__ == '__main__':
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f'Running on device {device}')
    
    data_config = {'name': 'Darcy',
                   'datapath': '/central/groups/tensorlab/dllin/PINO-master/data/piececonst_r421_N1024_smooth2.mat',
                   'total_num': 1024,
                   'offset': 0,
                    'n_sample': 1000,
                    'nx': 421,
                    'sub': 2}
    
    model_config = {'layers': [64, 64, 64, 64, 64],
                    'modes1': [20, 20, 20, 20],
                    'modes2': [20, 20, 20, 20],
                    'fc_dim': 128,
                    'act': 'gelu'}
    
    darcy_data = load_data(data_config)
    
    darcy_model = load_model(model_config)
    
    data = next(iter(darcy_data)) # get the first (a, u) pair from the data set
    u = data[1][0].to(device)
    u_full = u[::1, ::1].to(device)
    a_full = data[0][0][:, :, 0].to(device) # a is of the form [a(x, y), x, y] so we only want a(x, y)
    
    plt.imshow(a_full.detach().cpu().numpy())
    plt.savefig('a_particles/svno/actual_a.png', bbox_inches='tight')
    plt.imshow(u.detach().cpu().numpy())
    plt.savefig('a_particles/svno/actual_u.png', bbox_inches='tight')
    
    n = 20 # number of particles
    s = math.ceil(data_config['nx'] / data_config['sub']) # grid size
    
    svno = SteinVariationNeuralOperator(darcy_model, s, torch.rand([n, s, s], requires_grad=True, device=device), u_full, device, 1)
    svno.calculate_A_distribution()
    A_final = svno.get_A_distribution()

    images = []
    
    for i in range(n):
        img = A_final[i].detach().cpu().numpy()
        images.append(img)
        plt.imshow(img)
        plt.savefig(f'a_particles/svno/a_final_{str(i + 1).zfill(4)}.png', bbox_inches='tight')
    
    images = np.array(images)
    variance = np.var(images, axis=0)

    fig, ax = plt.subplots()
    im = ax.imshow(variance, cmap='viridis')
    cbar = ax.figure.colorbar(im, ax=ax)
    plt.savefig(f'a_particles/svno/variance.png', bbox_inches='tight')
