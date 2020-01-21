import torch
import torch.nn as nn
import numpy as np
from scipy.stats import truncnorm


class TruncationTrick(nn.Module):
    def __init__(self,version='basic',psi=0.7):
        super(TruncationTrick, self).__init__()
        self.version = version
        if version == 'basic':
        if version == 'lerp':
        if version == 'sigma':

    def truncated_noise_sample(self,batch_size=1, dim_z=512, truncation=1., seed=None):
        state = None if seed is None else np.random.RandomState(seed)
        #values = truncnorm.rvs(-2, 2, size=(batch_size, dim_z), random_state=state).astype(np.float32)
        values = truncnorm.rvs(-1.*truncation, truncation, size=(batch_size, dim_z), random_state=state).astype(np.float32)
        return values
        #if truncation < 1.:
        #    return values / (5 / truncation)
        #return values


class Distribution:
    def __init__(self, z):
        self.z = z
        self.metrics = {}
        self.metrics['mean'] = torch.mean(z) 
        self.metrics['std'] = torch.std(z) 
        self.metrics['max'] = torch.max(z) 
        self.metrics['min'] = torch.min(z) 
    def display_metrics(self):
        for k,v in self.metrics.items():
            print(k,':',v)

class GatedTNet(nn.Module):
    def __init__(self,nblocks,nz=512,residual=True,only_x_residual=False,arch='linear'):
        super(GatedTNet, self).__init__()
        insize=nz
        self.mlp = nn.Sequential(
                nn.Linear(insize,insize),
                nn.ReLU(),
                nn.Linear(insize,insize),
                nn.ReLU(),
                #nn.Linear(insize,insize),
                #nn.Sigmoid()
                )
        self.mlp.apply(init_weights)
        self.residual = residual
        self.nblocks = nblocks
        self.only_x_residual = only_x_residual
        for i in range(nblocks):
            if i == nblocks - 1:
                self.add_module(str(i), make_block(nz,arch,last=True) )
            else:
                self.add_module(str(i), make_block(nz,arch) )
    def forward(self, x):
        orig = x
        #print(x)
        x = x * self.mlp(x)
        #print(x)
        residual = x
        out = x
        for i in range(self.nblocks):
            out = self._modules[str(i)](out)
            if self.residual:
                out += residual
                if not self.only_x_residual:
                    residual = out
        out = orig + out
        #print(out)
        return out


class TTT(nn.Module):
    """
    a\item BPF + x
    b\item BPF-BF + x
    c\item BPF-BPF$_{bottleneck}$-BF + 
    d\item FBP + x
    e\item FBP-FB + x
    f\item FBP-F_${bottleneck}$BP-FB + 
    """
    def __init__(self,nblocks,nz=512,residual=True,only_x_residual=False,arch='linear'):
        super(TTT, self).__init__()
        self.residual = residual
        self.nblocks = nblocks
        self.only_x_residual = only_x_residual
        for i in range(nblocks):
            if i == nblocks - 1:
                self.add_module(str(i), make_block(nz,arch,last=True) )
            else:
                self.add_module(str(i), make_block(nz,arch) )
    def forward(self, x):
        residual = x
        out = x
        for i in range(self.nblocks):
            out = self._modules[str(i)](out)
            if self.residual:
                out += residual
                if not self.only_x_residual:
                    residual = out
        return out

class PDIP(nn.Module):
    def __init__(self, model=None, size=(1,100,1,1),nlayers=2,device="cpu"):
        super(PDIP, self).__init__()
        if model is None:
            return
        self.model = nn.Sequential(*list(model.children()))
        self.noise = []
        x_fake = torch.randn(size).to(device)
        for i,m in enumerate(self.model[:-1]):
            x_fake = m(x_fake)
            nch = x_fake.size()[1]
            #nn.PReLU(),
            net = nn.Sequential()
            for n in range(nlayers):
                net.add_module('conv'+str(n), nn.Conv2d(nch, nch, 1, stride=1, bias=False) )
                net.add_module('prelu'+str(n), nn.PReLU() )
            self.noise.append(net)
        self.noise = nn.ModuleList(self.noise)

    def forward(self, x):
        for i,m in enumerate(self.model[:-1]):
            x = m(x) 
            x = self.noise[i](x) + x
        x = self.model[-1](x)
        return x

def make_block(insize,arch,last=False):
    if arch == 'a' :#a\item BPF + x
        return nn.Sequential(
                nn.BatchNorm1d(insize)
                nn.PReLU(),
                nn.Linear(insize,insize),
                )
    elif arch =='b': #b\item BPF-BF + x
        return nn.Sequential(
                nn.BatchNorm1d(insize)
                nn.PReLU(),
                nn.Linear(insize,insize),

                nn.BatchNorm1d(insize)
                nn.Linear(insize,insize),
                )
    elif arch =='c': #c\item BPF-BPF$_{bottleneck}$-BF + 
        return nn.Sequential(
                nn.BatchNorm1d(insize)
                nn.PReLU(),
                nn.Linear(insize,insize),

                nn.BatchNorm1d(insize)
                nn.PReLU(),
                nn.Linear(insize,insize/2),

                nn.BatchNorm1d(insize/2)
                nn.Linear(insize/2,insize),
                )
    elif arch =='d': #d\item FBP + x
        return nn.Sequential(
                nn.Linear(insize,insize),
                nn.BatchNorm1d(insize)
                nn.PReLU(),
                )
    elif arch =='e': #e\item FBP-FB + x
        return nn.Sequential(
                nn.Linear(insize,insize),
                nn.BatchNorm1d(insize)
                nn.PReLU(),

                nn.Linear(insize,insize),
                nn.BatchNorm1d(insize)
                )
    elif arch =='f': #f\item FBP-F_${bottleneck}$BP-FB + 
        return nn.Sequential(
                nn.Linear(insize,insize),
                nn.BatchNorm1d(insize)
                nn.PReLU(),

                nn.Linear(insize,insize/2),
                nn.BatchNorm1d(insize/2)
                nn.PReLU(),

                nn.Linear(insize/2,insize),
                nn.BatchNorm1d(insize)
                )
    elif arch == 'linear'  or last:
        return nn.Sequential(
                nn.Linear(insize,insize),
                )
    elif arch == 'relu':
        return nn.Sequential(
                nn.Linear(insize,insize),
                nn.ReLU(),
                )
    elif arch == 'elu':
        return nn.Sequential(
                nn.Linear(insize,insize),
                nn.ELU(),
                )
    elif arch == 'prelu':
        return nn.Sequential(
                nn.Linear(insize,insize),
                nn.PReLU(),
                )
    elif arch == 'sigmoid':
        return nn.Sequential(
                nn.Linear(insize,insize),
                nn.Sigmoid(),
                )
    elif arch == 'tanh':
        return nn.Sequential(
                nn.Linear(insize,insize),
                nn.Tanh(),
                )
    elif arch == 'batchnorm':
        return nn.Sequential(
                nn.Linear(insize,insize),
                nn.BatchNorm1d(insize),
                )
    elif arch == 'groupnorm':
        return nn.Sequential(
                nn.Linear(insize,insize),
                nn.GroupNorm(6,1),
                )
    elif arch == 'instancenorm':
        return nn.Sequential(
                nn.Linear(insize,insize),
                nn.InstanceNorm1d(insize),
                )
    elif arch == 'layernorm':
        return nn.Sequential(
                nn.Linear(insize,insize),
                nn.LayerNorm(insize),
                )

def init_weights(m):
    if type(m) == nn.Linear:
        #kaiming_normal_
        #torch.nn.init.xavier_uniform(m.weight)
        #torch.nn.init.kaiming_normal_(m.weight)
        nn.init.normal_(m.weight, mean=0.0, std=0.00001)
        #nn.init.normal_(m.weight, mean=0.0, std=0.001)
        m.bias.data.fill_(0.00001)
    if type(m) == nn.Conv2d:
        #nn.init.normal_(m.weight, mean=0.0, std=0.0000000001) # works for 2layer
        nn.init.normal_(m.weight, mean=0.0, std=0.00001)
        #m.bias.data.fill_(0.00001)
"""
if __name__ == '__main__':
    z = torch.randn(120)
    d = Distribution(z)
    print('orig')
    d.display_metrics()
    nlayers = 100
    t2 = TNet(nlayers,arch='prelu')
    t2.apply(init_weights)
    #for layer in t.modules():
    #    if isinstance(layer, nn.Linear):
    #        print(layer.weight)
    zp = t2(z) 
    dp = Distribution(zp)
    dp.display_metrics()

    t2 = GatedTNet(nlayers,arch='prelu')
    t2.apply(init_weights)
    zp = t2(z) 
    dp = Distribution(zp)
    dp.display_metrics()
"""
