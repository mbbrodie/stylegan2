import torch
import torch.nn as nn
from torch.autograd import Variable

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

"""
if __name__ == '__main__':
    g = Generator()
    w = PDIP(g)
    optimizerG = torch.optim.Adam(w.noise.parameters(), lr=0.0001, betas=(0.5, 0.999))
    
    x = torch.randn(1,100,1,1)
    out = w(x)
    print(out.size())
"""
