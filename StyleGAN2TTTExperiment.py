from abstract_ttt import TTTExperiment
import torch
import easydict
import numpy as np
from scipy.stats import truncnorm
from torch.nn import functional as F
from torchvision.utils import save_image
from ttt import TTT
from pdip import PDIP # do for generator.features
from model import Generator, Discriminator
from os.path import join
from torch.autograd import Variable

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
            save_image(images, join(args.savedir, str(i)+'.png'),normalize=True)
    
    def sample_z(self, **kwargs):
        sample_z = torch.randn(args.n_eval_samples, args.latent, device=args.device)
        args['z'] = sample_z

    def sample_w(self, **kwargs):
        if 'z' not in args.keys():
            self.sample_z()
        args['w'] = args['g'].get_latent(args.z)
    
        #return a + (b - a) * t

    def set_np_random_state_with_seed(self,seed):
        args.state = np.random.RandomState(seed)

    def truncate_z(self):
        # add existing scipy truncnorm code
        args.z = truncnorm.rvs(-1*args.truncation, args.truncation, size=(args.batch_size, args.dim_z), random_state=args.state).astype(np.float32)
        
    def truncate_w_with_lerp(self):
        mean_w = args.g.mean_latent(args.w.size(0))
        args['w'] = self.lerp(mean_w, args.w, args.truncation)

    def sample_n_stylegan_images_without_tt(self, **kwargs):
        self.sample_w()
        self.gen_from_w()

    def sample_n_stylegan_images_with_w_ttl(self, **kwargs):
        self.sample_w()
        self.trucate_w_with_lerp()
        self.gen_from_w()

    def sample_n_stylegan_images_with_z_tt(self, **kwargs):
        ### the paper says tt doesn't work with z
        self.sample_z()
        self.truncate_z()
        self.gen_from_z()
    
    def sample_n_stylegan_images_with_coachgan(self, **kwargs):
        self.sample_z()
        self.coach_z()
        self.gen_from_z()
    
    def sample_n_stylegan_images_with_w_coachgan(self, **kwargs):
        self.sample_w()
        self.coach_w()
        self.gen_from_w()

    def sample_n_stylegan_images_with_post_w_prenetwork_ttt(self, **kwargs):
        self.sample_w()
        self.w_ttt()
        self.gen_from_w()

    def sample_n_stylegan_images_with_prenetwork_ttt(self, **kwargs):
        self.sample_z()
        self.ttt()
        self.gen_from_z()
    
    def sample_n_stylegan_images_with_intranetwork_ttt(self, **kwargs):
        #same as normal sample
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
        args.w = args.wttt( args.w )
    
    def setup_prenetwork_w_ttt(self, **kwargs):
        args['wttt'] = TTT(args.nlayer, nz=args.z.size(1),arch=args.arch)

    def setup_prenetwork_ttt(self, **kwargs):
        args['ttt'] = TTT(args.nlayer, nz=args.z.size(1),arch=args.arch)
    
    def setup_intranetwork_ttt(self, **kwargs):
        args['g'] = PDIP(model=g.features,arch=args.arch,nlayers=args.nlayer, size=args.w.size())
    
    ## OPTIMIZER SETUP ##
    def get_optimizer(self, net):
        return torch.optim.Adam(net.parameters(), lr=args.lr)

    def g_nonsaturating_loss(self):
        fake_pred = args.d(args.fake)
        loss = F.softplus(-fake_pred).mean()
        return loss
    
    def train_prenetwork_w_ttt(self, **kwargs):
        opt = get_optimizer(args.w_ttt)
        for i in range(args.niter):
            opt.zero_grad()
            self.sample_n_stylegan_images_with_post_w_prenetwork_ttt()
            g_loss = self.g_nonsaturating_loss()
            g_loss.backward()
            opt.step()

    def train_prenetwork_ttt(self, **kwargs):
        opt = get_optimizer(args.ttt)
        for i in range(args.niter):
            opt.zero_grad()
            self.sample_n_stylegan_images_with_prenetwork_ttt()
            g_loss = self.g_nonsaturating_loss()
            g_loss.backward()
            opt.step()
    
    def train_intranetwork_ttt(self, **kwargs):
        opt = get_optimizer(args.g.noise)
        for i in range(args.niter):
            opt.zero_grad()
            self.sample_n_stylegan_images_with_intranetwork_ttt()
            g_loss = self.g_nonsaturating_loss()
            g_loss.backward()
            opt.step()

    #NOTE: this is z TTT right now. could adapt for w TTT if that performs well
    def train_prenetwork_and_intranetwork_ttt(self, **kwargs):
        opt = torch.optim.Adam(list(args.ttt.parameters() + args.w_ttt.parameters()), lr=args.lr)
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
        for j in range(1):
            self.gen_from_z()
            g_loss = self.g_nonsaturating_loss()
            g_loss.backward()
            opt.step()

    def coach_w(self, **kwargs):
        args.w = Variable(args.w, requires_grad=True)
        opt = torch.optim.Adam([w], lr=0.01)
        for j in range(10):
            self.gen_from_w()
            g_loss = self.g_nonsaturating_loss()
            g_loss.backward()
            opt.step()
