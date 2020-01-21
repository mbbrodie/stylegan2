# code wrappers (for simple colab imports)
from model import Generator, Discriminator
import easydict
args = easydict.EasyDict()
from PIL import Image
def get_generator(device='cpu'):
    """return G in train mode on CPU"""
    g_ema = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    checkpoint = torch.load(args.ckpt)

    g_ema.load_state_dict(checkpoint['g_ema'])
    return g_ema

def get_discriminator(device='cpu'):
    """return D in train mode on CPU"""
    discriminator = Discriminator(
        args.size, channel_multiplier=args.channel_multiplier
    ).to(device)
    checkpoint = torch.load(args.ckpt)
    discriminator.load_state_dict(checkpoint['d'])
    return discriminator

def load_test_images():
    a1 = Image.open('a')
    b1 = Image.open('b')
    b2 = Image.open('b2')

    #PIL load
    #normalize [-1,1]
    return a1,b1,b2

def get_w_embeddings(z):
    return generator.get_latent(z)

def generate_from_w(w,truncation=1.):
    """
    def forward(
        self,
        styles,
        return_latents=False,
        inject_index=None,
        truncation=1,
        truncation_latent=None,
        input_is_latent=False,
        noise=None,
        randomize_noise=False
    ):
    """
    img, latents = generator(w, input_is_latent=True, truncation=truncation)
    return img
def coach():
    pass

def sample_z():
    sample_z = torch.randn(args.sample, args.latent, device=device)
    #latent_in = torch.randn(
    #    n_latent, self.style_dim, device=self.input.input.device
    #)
def sample_w():
    pass


