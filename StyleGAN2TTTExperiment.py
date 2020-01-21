from abstract_ttt import TTTExperiment
import torch
import easydict
import numpy as np
from scipy.stats import truncnorm
from torch.nn import functional as F
from torchvision.utils import save_image
from ttt import TTT
from pdip import PDIP # do for generator.features

args = None
class StyleGAN2TTTExperiment(TTTExperiment):
    def __init__(self):
        super(StyleGAN2TTTExperiment, self)
        self.args = easydict.EasyDict()

    def set_args(self, a):
        global args
        args = a
    def gen_from_z(self):
        args.fake = args.g(args.z)
    def gen_from_w(self):
        args.fake = args.g(args.w, input_is_latent=True)
    
    def setup(self, **kwargs):
        """return G in train mode on CPU"""
        g_ema = Generator(
            args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
        ).to(device)
        checkpoint = torch.load(args.ckpt)
        g_ema.load_state_dict(checkpoint['g_ema'])
        args['g'] = g_ema
        """return D in train mode on CPU"""
        discriminator = Discriminator(
            args.size, channel_multiplier=args.channel_multiplier
        ).to(device)
        checkpoint = torch.load(args.ckpt)
        discriminator.load_state_dict(checkpoint['d'])
        args['d'] = discriminator
        return


    
    def save_results(self, **kwargs):
        for img in args.fake:
            images = img.cpu()
            images = torch.nn.functional.interpolate(images,size=(299,299))
            save_image(images, os.path.join(args.savedir, str(i)+'.png'),normalize=True)
    
    def sample_z(self, **kwargs):
        sample_z = torch.randn(args.sample, args.latent, device=device)
        args['z'] = sample_z

    def sample_w(self, **kwargs):
        if 'z' not in args.keys():
            self.sample_z()
        args['w'] = args['g'].get_latent(args.z)
    
        return a + (b - a) * t

    def set_np_random_state_with_seed(self,seed):
        args.state = np.random.RandomState(seed)

    def truncate_z(self):
        # add existing scipy truncnorm code
        args.z = truncnorm.rvs(-1*args.truncation, args.truncation, size=(args.batch_size, args.dim_z), random_state=args.state).astype(np.float32)
        
    def truncate_w_with_lerp(self):
        mean_w = args.g.mean_latent(args.w.size(0))
        args['w'] = self.lerp(mean_w, args.w, args.truncation)

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

    def g_nonsaturating_loss(self, fake_pred):
        fake_pred = args.d(args.fake)
        loss = F.softplus(-fake_pred).mean()
        return loss


    def train_prenetwork_w_ttt(self, **kwargs):
        opt = get_optimizer(args.w_ttt)
        for i in range(args.niter):
            opt.zero_grad()
            self.sample_n_stylegan_images_with_post_w_prenetwork_ttt()
            self.calc_g_loss()
            g_loss = g_nonsaturating_loss(fake_pred)
            g_loss.backward()
            opt.step()

    def train_prenetwork_ttt(self, **kwargs):
        opt = get_optimizer(args.ttt)
        for i in range(args.niter):
            opt.zero_grad()
            self.sample_n_stylegan_images_with_prenetwork_ttt()
            self.calc_g_loss()
            g_loss = g_nonsaturating_loss(fake_pred)
            g_loss.backward()
            opt.step()
    
    def train_intranetwork_ttt(self, **kwargs):
        opt = get_optimizer(pass)
        for i in range(args.niter):
            opt.zero_grad()
            self.sample_n_stylegan_images_with_intranetwork_ttt()
            self.calc_g_loss()
            g_loss = g_nonsaturating_loss(fake_pred)
            g_loss.backward()
            opt.step()

    #NOTE: this is z TTT right now. could adapt for w TTT if that performs well
    def train_prenetwork_and_intranetwork_ttt(self, **kwargs):
        opt = torch.optim.Adam(list(args.ttt.parameters() + args.w_ttt.parameters()), lr=args.lr)
        for i in range(args.niter):
            opt.zero_grad()
            self.sample_n_stylegan_images_with_pre_and_intranetwork_ttt()
            self.calc_g_loss()
            g_loss = g_nonsaturating_loss(fake_pred)
            g_loss.backward()
            opt.step()

    def save_models(self, **kwargs):
        pass
    
    def coach_z(self, **kwargs):
        args.z = Variable(args.z, requires_grad=True)
        opt = torch.optim.Adam([z], lr=0.01)
        for j in range(10):
            self.gen_from_z()
            self.calc_g_loss()
            g_loss = g_nonsaturating_loss(fake_pred)
            g_loss.backward()
            opt.step()

    def coach_w(self, **kwargs):
        args.w = Variable(args.w, requires_grad=True)
        opt = torch.optim.Adam([w], lr=0.01)
        for j in range(10):
            self.gen_from_w()
            self.calc_g_loss()
            g_loss = g_nonsaturating_loss(fake_pred)
            g_loss.backward()
            opt.step()
