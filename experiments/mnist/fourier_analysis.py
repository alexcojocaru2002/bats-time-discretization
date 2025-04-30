import torch
import numpy as np
from torchvision import datasets, transforms

import matplotlib.pyplot as plt

transform=transforms.Compose([
        transforms.ToTensor()])

batch_size=128
dataset1 = datasets.MNIST('../data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(dataset1,batch_size=batch_size)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

n_samples=len(dataset1)

TIME_WINDOW = 100e-3
MAX_VALUE = 1.0

dt=0.0001
t_max=TIME_WINDOW
v_time=torch.arange(0,t_max,dt)
n_steps=len(v_time)

MNIST_H=28
MNIST_W=28
sptr=torch.zeros(batch_size, n_steps+1, MNIST_H, MNIST_W).to(device) # [batch x time_steps x H x W])
sum_fft=torch.zeros(n_steps+1,dtype=torch.complex128).to(device)

for batch_idx, (data, target) in enumerate(train_loader):
    data, target = data.to(device), target.to(device)
    spike_times = TIME_WINDOW * (1 - (data / MAX_VALUE))

    # Initialize/reset spike trains
    sptr[:,:,:,:]=0

    # Find spike indices for each batch x neuron
    sp_ind=spike_times*(n_steps/TIME_WINDOW)
    sp_ind=sp_ind.type(torch.int64)

    # Annotate the spikes
    idx=np.indices(sp_ind.shape)
    sptr[idx[0],sp_ind,idx[2],idx[3]]=1

    # Spike train per image
    sptr_im=sptr.sum(dim=(2,3))
    sum_fft+=torch.sum(torch.fft.fft(sptr_im,dim=1),dim=0)

    if batch_idx % 100 == 0:
        print(batch_idx)

sum_fft=sum_fft/n_samples

def freq_to_time():
    sr = 1 / dt
    n = np.arange(n_steps+1)
    T = n_steps / sr
    return n / T

# n2 = n_steps / 2
f_oneside = freq_to_time()
print(f_oneside)
plt.plot((1 / f_oneside[500:]), torch.abs(sum_fft[500:]))
plt.xlabel("DT")
plt.ylabel("Frequency")
plt.show()
