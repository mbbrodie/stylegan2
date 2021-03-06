from abstract_ttt import TTTExperiment
import torch
import easydict
import numpy as np
from scipy.stats import truncnorm
from torch.nn import functional as F
import torch.nn as nn
from torchvision.utils import save_image
from ttt import TTT
from pdip import PDIP # do for generator.features
from model import Generator, Discriminator
import os
from os.path import join
from torch.autograd import Variable
from pdip_model import Generator as PG
from zipfile import ZipFile
import shutil

import random
seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


args = None
class StyleGAN2TTTExperiment(TTTExperiment):
    def __init__(self):
        super(StyleGAN2TTTExperiment, self)
        self.args = easydict.EasyDict()

    def set_args(self, a):
        global args
        args = a
    def calc_metrics(self):
        #just call fid.py with img_path
        if 'fid' in args.metrics:
            print('FID: ')
            print(metrics.fid(args))
        if 'lpips' in args.metrics:
            print('LPIPS: ')
            print(metrics.fid(args))

    def gen_from_z(self):
        args.w = args['g'].get_latent(args.z)
        #args.fake, _ = args.g([args.z]) # determinism not working out this way
        args.fake, _ = args.g([args.w], input_is_latent=True)
    def gen_from_w(self):
        args.fake, _ = args.g([args.w], input_is_latent=True)

    def setup(self, **kwargs):
        checkpoint = torch.load(args.checkpoint)

        g_ema = Generator(
            args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
        ).to(args.device)
        g_ema.load_state_dict(checkpoint['g_ema'],strict=False)
        #g_ema.load_state_dict(checkpoint['g'],strict=False)
        args['g'] = g_ema

        discriminator = Discriminator(
            args.size, channel_multiplier=args.channel_multiplier
        ).to(args.device)
        discriminator.load_state_dict(checkpoint['d'],strict=False)
        args['d'] = discriminator

    def save_results(self, **kwargs):
        for i,img in enumerate(args.fake):
            images = img.cpu()
            images = images.unsqueeze(0)
            images = torch.nn.functional.interpolate(images,size=(299,299))
            #save_image(images, join(args.savedir, str(kwargs['num']+i)+'.png'),normalize=True)
            save_image(images, join(args.tempdir, str(kwargs['num']+i)+'.png'),normalize=True)
    def transfer_results(self):
        if args.method in ['TTTz','TTTw','TNet','TNet+TTT']:
            zipname =  args.base_exp_name+  args.method+args.arch+str(args.nlayer)
        else:
            zipname =   args.base_exp_name+  args.method
        zipname = zipname 
        files = os.listdir(args.tempdir) 
        with ZipFile(zipname,'w') as z:
            for f in files:
                z.write(join(args.tempdir,f))
        shutil.move(zipname, join(args.savedir,zipname))
        
                        
    
    def sample_z(self, **kwargs):
        #sample_z = torch.randn(args.n_eval_samples, args.latent, device=args.device)
        #sample_z = torch.randn(args.batch_size, args.latent, device=args.device)
        #args['z'] = sample_z
        self.truncate_z()

    def sample_w(self, **kwargs):
        #if 'z' not in args.keys():
        #    self.sample_z()
        args['w'] = args['g'].get_latent(args.z)
    
    def lerp(self, a, b, t):
        return a + (b - a) * t

    def set_np_random_state_with_seed(self,seed):
        args.state = np.random.RandomState(seed)

    def truncate_z(self):
        # add existing scipy truncnorm code
        #args.z = truncnorm.rvs(-1*args.truncation, args.truncation, size=(args.n_eval_samples, args.latent), random_state=args.state).astype(np.float32)
        args.z = truncnorm.rvs(-1*args.truncation, args.truncation, size=(args.batch_size, args.latent)).astype(np.float32)
        args.z = torch.from_numpy(args.z).to('cuda')
        
    def truncate_w_with_lerp(self):
        mean_w = args.g.mean_latent(args.w.size(0))
        args['w'] = self.lerp(mean_w, args.w, args.truncation)

    def sample_n_stylegan_images_without_tt(self, **kwargs):
        self.sample_z()
        self.sample_w()
        self.gen_from_w()

    def sample_n_stylegan_images_with_w_ttl(self, **kwargs):
        self.sample_z()
        self.sample_w()
        self.truncate_w_with_lerp()
        self.gen_from_w()

    def sample_n_stylegan_images_with_z_tt(self, **kwargs):
        ### the paper says tt doesn't work with z
        self.sample_z()
        #self.truncate_z()
        self.gen_from_z()
    
    def sample_n_stylegan_images_with_coachgan(self, **kwargs):
        self.sample_z()
        self.coach_z()
        self.gen_from_z()
    
    def sample_n_stylegan_images_with_w_coachgan(self, **kwargs):
        self.sample_z()
        self.sample_w()
        self.coach_w()
        self.gen_from_w()

    def sample_n_stylegan_images_with_post_w_prenetwork_ttt(self, **kwargs):
        self.sample_z()
        self.sample_w()
        self.w_ttt()
        self.gen_from_w()

    def sample_n_stylegan_images_with_prenetwork_ttt(self, **kwargs):
        self.sample_z()
        self.ttt()
        self.gen_from_z()
    
    def sample_n_stylegan_images_with_intranetwork_ttt(self, **kwargs):
        #same as normal sample
        self.sample_z()
        self.sample_w()
        self.gen_from_w()
    
    def sample_n_stylegan_images_with_pre_and_intranetwork_ttt(self, **kwargs):
        self.sample_z()
        self.ttt()
        self.sample_w()        
        self.w_ttt()
        self.gen_from_w()

    def ttt(self):
        args.z = args.ttt( args.z )
    def w_ttt(self):
        args.w = args.w_ttt( args.w )

    def init_weights(self,m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, mean=0.0, std=0.000000001)
            m.bias.data.fill_(0.000000001)
        elif type(m) == nn.Conv2d:
            nn.init.normal_(m.weight, mean=0.0, std=0.000000001)
    
    def setup_prenetwork_w_ttt(self, **kwargs):
        ttt = TTT(args.nlayer, nz=args.latent, arch=args.arch)
        ttt.apply(self.init_weights)
        args['w_ttt'] = ttt.cuda()

    def setup_prenetwork_ttt(self, **kwargs):
        #add weight init!
        ttt = TTT(args.nlayer, nz=args.latent, arch=args.arch)
        ttt.apply(self.init_weights)
        args['ttt'] = ttt.cuda()
    
    def setup_intranetwork_ttt(self, **kwargs):
        checkpoint = torch.load(args.checkpoint)
        g_ema = PG(
            args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier,
            use_ttt=True,
            arch=args.arch,
            nlayer=args.nlayer
        ).to(args.device)
        g_ema.apply(self.init_weights)
        g_ema.load_state_dict(checkpoint['g_ema'],strict=False)
        args['g'] = g_ema
        #args['g'] = PDIP(model=args.g,arch=args.arch, nlayers=args.nlayer, size=(1, args.latent),device='cuda').cuda()
    
    ## OPTIMIZER SETUP ##
    def get_optimizer(self, net):
        return torch.optim.Adam(net.parameters(), lr=args.lr)

    def g_nonsaturating_loss(self):
        fake_pred = args.d(args.fake)
        loss = F.softplus(-fake_pred).mean()
        return loss
    
    def train_prenetwork_w_ttt(self, **kwargs):
        opt = self.get_optimizer(args.w_ttt)
        for i in range(args.niter):
            opt.zero_grad()
            self.sample_n_stylegan_images_with_post_w_prenetwork_ttt()
            g_loss = self.g_nonsaturating_loss()
            g_loss.backward()
            opt.step()

    def train_prenetwork_ttt(self, **kwargs):
        opt = self.get_optimizer(args.ttt)
        for i in range(args.niter):
            opt.zero_grad()
            self.sample_n_stylegan_images_with_prenetwork_ttt()
            g_loss = self.g_nonsaturating_loss()
            g_loss.backward()
            opt.step()
    
    def train_intranetwork_ttt(self, **kwargs):
        opt = self.get_optimizer(args.g.ttts)
        for i in range(args.niter):
            opt.zero_grad()
            self.sample_n_stylegan_images_with_intranetwork_ttt()
            g_loss = self.g_nonsaturating_loss()
            g_loss.backward()
            opt.step()

    #NOTE: this is z TTT right now. could adapt for w TTT if that performs well
    def train_prenetwork_and_intranetwork_ttt(self, **kwargs):
        opt = torch.optim.Adam(list(args.ttt.parameters()) + list(args.w_ttt.parameters())+list(args.g.ttts.parameters()), lr=args.lr)
        for i in range(args.niter):
            opt.zero_grad()
            self.sample_n_stylegan_images_with_pre_and_intranetwork_ttt()
            g_loss = self.g_nonsaturating_loss()
            g_loss.backward()
            opt.step()

    def save_models(self, **kwargs):
        pass
    
    def coach_z(self, **kwargs):
        args.z = Variable(args.z, requires_grad=True)
        opt = torch.optim.Adam([args.z], lr=0.01)
        for j in range(5):
            self.gen_from_z()
            g_loss = self.g_nonsaturating_loss()
            g_loss.backward()
            opt.step()

    def coach_w(self, **kwargs):
        args.w = Variable(args.w, requires_grad=True)
        opt = torch.optim.Adam([args.w], lr=0.01)
        for j in range(5):
            self.gen_from_w()
            g_loss = self.g_nonsaturating_loss()
            g_loss.backward()
            opt.step()
