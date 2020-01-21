import torch
import torch.nn as nn
from torch.autograd import Variable

class PDIP(nn.Module):
    def __init__(self, model=None, arch='prelu', size=(1,100,1,1),nlayers=2,device="cpu"):
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
                net.add_module(make_pdip_block(nch,arch)
                #net.add_module('conv'+str(n), nn.Conv2d(nch, nch, 1, stride=1, bias=False) )
                #net.add_module('prelu'+str(n), nn.PReLU() )
            #net = nn.Sequential(nn.Conv2d(nch, nch, 3, 1, 1, bias=False),nn.BatchNorm2d(nch),nn.ReLU()) 
            #net = nn.Sequential(nn.Conv2d(nch, nch, 3, 1, 1, bias=False),nn.BatchNorm2d(nch),nn.ReLU(),nn.Conv2d(nch, nch, 3, 1, 1, bias=False),nn.BatchNorm2d(nch),nn.ReLU()) 
            self.noise.append(net)
            #self.noise.append( Variable(torch.randn(x_fake.size()).to(device), requires_grad=True) )
        self.noise = nn.ModuleList(self.noise)

    def forward(self, x):
        for i,m in enumerate(self.model[:-1]):
            x = m(x) 
            x = self.noise[i](x) + x
        x = self.model[-1](x)
        return x

def make_pdip_block(insize,arch,last=False):
    if arch == 'a' :#a\item BPF + x
        return nn.Sequential(
                nn.BatchNorm2d(insize)
                nn.PReLU(),
                nn.Conv2d(insize, insize, 1, stride=1, bias=False)
                )
    elif arch =='b': #b\item BPF-BF + x
        return nn.Sequential(
                nn.BatchNorm2d(insize)
                nn.PReLU(),
                nn.Conv2d(insize, insize, 1, stride=1, bias=False)

                nn.BatchNorm2d(insize)
                nn.Conv2d(insize, insize, 1, stride=1, bias=False)
                )
    elif arch =='c': #c\item BPF-BPF$_{bottleneck}$-BF + 
        return nn.Sequential(
                nn.BatchNorm2d(insize)
                nn.PReLU(),
                nn.Conv2d(insize, insize, 1, stride=1, bias=False)

                nn.BatchNorm2d(insize)
                nn.PReLU(),
                nn.Linear(insize,insize/2),

                nn.BatchNorm2d(insize/2)
                nn.Conv2d(insize/2, insize, 1, stride=1, bias=False)
                )
    elif arch =='d': #d\item FBP + x
        return nn.Sequential(
                nn.Conv2d(insize, insize, 1, stride=1, bias=False)
                nn.BatchNorm2d(insize)
                nn.PReLU(),
                )
    elif arch =='e': #e\item FBP-FB + x
        return nn.Sequential(
                nn.Conv2d(insize, insize, 1, stride=1, bias=False)
                nn.BatchNorm2d(insize)
                nn.PReLU(),

                nn.Conv2d(insize, insize, 1, stride=1, bias=False)
                nn.BatchNorm2d(insize)
                )
    elif arch =='f': #f\item FBP-F_${bottleneck}$BP-FB + 
        return nn.Sequential(
                nn.Conv2d(insize, insize, 1, stride=1, bias=False)
                nn.BatchNorm2d(insize)
                nn.PReLU(),

                nn.Conv2d(insize, insize/2, 1, stride=1, bias=False)
                nn.BatchNorm2d(insize/2)
                nn.PReLU(),

                nn.Conv2d(insize/2, insize, 1, stride=1, bias=False)
                nn.BatchNorm2d(insize)
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
                nn.Conv2d(insize, insize, 1, stride=1, bias=False)
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

"""
if __name__ == '__main__':
    g = Generator()
    w = PDIP(g)
    optimizerG = torch.optim.Adam(w.noise.parameters(), lr=0.0001, betas=(0.5, 0.999))
    
    x = torch.randn(1,100,1,1)
    out = w(x)
    print(out.size())
"""
