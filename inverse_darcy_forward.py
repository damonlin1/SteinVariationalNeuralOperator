import matplotlib.pyplot as plt
import math
import numpy as np
import torch
from scipy.fftpack import idct
from torch.utils.data import DataLoader

from models import FNO2d
from train_utils.datasets import DarcyFlow
from train_utils.losses import LpLoss, darcy_loss
from train_utils.utils import torch2dgrid

device = 0 if torch.cuda.is_available() else 'cpu'
print(f'Running on device {device}')

# configurations for loading in the dataset and model
data_config = {'name': 'Darcy',
               'datapath': '/central/groups/tensorlab/dllin/PINO-master/data/darcy_test.mat',
               'total_num': 100,
               'offset': 0,
               'n_sample': 100,
               'nx': 421,
               'sub': 2}
model_config = {'layers': [64, 64, 64, 64, 64],
                'modes1': [20, 20, 20, 20],
                'modes2': [20, 20, 20, 20],
                'fc_dim': 128,
                'act': 'gelu'}

# load in the dataset
dataset = DarcyFlow(data_config['datapath'],
                    nx=data_config['nx'],
                    sub=data_config['sub'],
                    offset=data_config['offset'],
                    num=data_config['n_sample'])
train_loader = DataLoader(dataset, batch_size=1, shuffle=True)

# load in the model
model = FNO2d(modes1=model_config['modes1'],
              modes2=model_config['modes2'],
              fc_dim=model_config['fc_dim'],
              layers=model_config['layers'],
              act=model_config['act']).to(device)
ckpt = torch.load('/central/groups/tensorlab/dllin/PINO-master/checkpoints/darcy-FDM/darcy-pretrain-pino.pt')
model.load_state_dict(ckpt['model'])

def total_variance(x):
    return torch.mean(torch.abs(x[...,:-1] - x[...,1:])) + torch.mean(torch.abs(x[...,:-1,:] - x[...,1:,:]))

# define the grid size
s = 211
ss = math.ceil(data_config['nx'] / data_config['sub'])
r = math.ceil(s / ss)

grid = torch2dgrid(ss, ss).cuda()

myloss = LpLoss(size_average=True)

mollifier = torch.sin(np.pi*grid[...,0]) * torch.sin(np.pi*grid[...,1]) * 0.001

def idct2(y):
    M, N = y.shape
    a = np.empty([M, N])
    b = np.empty([M, N])
    for i in range(M):
        a[i, :] = idct(y[i, :], norm='ortho')
    for j in range(N):
        b[:, j] = idct(a[:, j], norm='ortho')
    return b

def GRF(alpha, tau, s):
    xi = np.random.normal(0, 1, [s, s])
    x = np.linspace(0, s - 1, s)
    y = np.linspace(0, s - 1, s)
    K1, K2 = np.meshgrid(x, y)
    coef = tau ** (alpha - 1) * (np.pi ** 2 * (np.square(K1) + np.square(K2)) + tau ** 2) ** (-alpha / 2)
    L = s * coef * xi
    L[0, 0] = 0
    U = np.exp(idct2(L))
    return torch.from_numpy(U).type(torch.float32)

for _ in range(1):
    epochs = 10000
    a_error, u_error, pde_error = [], [], []
    x, y = next(iter(train_loader)) # get the first (a, u) pair from the data set
    y = y.cuda()
    x = x[0, :, :, 0].cuda() # a is of the form (a(x, y), x, y) but we only want a(x, y)
    
    xout = GRF(5, 6, s).cuda()
    xout.requires_grad = True
    
    # xout = torch.rand([s, s], requires_grad=True, device=device)

    optimizer = torch.optim.Adam([xout], lr=0.0005, weight_decay=1e-7)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)
        
    for ep in range(epochs):
        optimizer.zero_grad()
        
        out_masked = xout[::r, ::r]
        testx_l2 = myloss(out_masked, x)
        
        out_masked1 = torch.cat([out_masked.unsqueeze(2), grid], dim=2).reshape(1, ss, ss, 3)

        yout = model(out_masked1).reshape(1, ss, ss)
        yout = yout * mollifier
        yout = yout
        
        loss_data = myloss(yout, y)
        loss_TV = total_variance(xout)
        
        pde_loss = darcy_loss(y, out_masked)
        
        pino_loss = loss_data + 0.1 * pde_loss + 0.2 * loss_TV
        pino_loss.backward()
        optimizer.step()
        scheduler.step()
        
        a_error.append(testx_l2.item())
        u_error.append(loss_data.item())
        pde_error.append(pde_loss.item())
            
        print(ep, loss_data.item(), testx_l2.item(), pde_loss.item())
        
        if ep == epochs - 1:
            name_tag = 'a_particles/PINO-'
            plt.imshow(x.reshape(ss, ss).detach().cpu().numpy())
            plt.savefig(name_tag+'true-input.png',bbox_inches='tight')
            plt.imshow(out_masked.reshape(ss, ss).detach().cpu().numpy())
            plt.savefig(name_tag+'pred-input.png',bbox_inches='tight')

            plt.imshow(y.reshape(ss, ss).detach().cpu().numpy())
            plt.savefig(name_tag+'true-output.png',bbox_inches='tight')
            plt.imshow(yout.reshape(ss, ss).detach().cpu().numpy())
            plt.savefig(name_tag+'pred-output.png',bbox_inches='tight')

plt.figure()
plt.plot(a_error)
plt.xlabel('Epochs')
plt.ylabel('L2 error of a')
plt.savefig('error_a.jpg')

plt.figure()
plt.plot(u_error)
plt.xlabel('Epochs')
plt.ylabel('L2 error of u')
plt.savefig('error_u.jpg')

plt.figure()
plt.plot(pde_error)
plt.xlabel('Epochs')
plt.ylabel('L2 error of pde')
plt.savefig('error_pde.jpg')
