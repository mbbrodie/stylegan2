### this will be copied to colab
### everything else imported
from StyleGAN2TTTExperiment import StyleGAN2TTTExperiment
import sys
import os

import torch
import numpy as np
import random

def run(args):
    e = StyleGAN2TTTExperiment() 
    #print(args)
    e.set_args(args)
    e.setup()
    ## SETUP
    #if args.TTT:
    if args.method == 'TNet+TTT' or args.method == 'TTTz':
        e.setup_prenetwork_ttt()
    if args.method == 'TNet+TTT' or args.method == 'TTTw':
        e.setup_prenetwork_w_ttt()
    if args.method == 'TNet+TTT' or args.method == 'TNet':
        e.setup_intranetwork_ttt()
    print('finished setup')

    ## TRAIN
    # may need to set train_sample_size
    if args.method == 'TNet+TTT':
        e.train_prenetwork_and_intranetwork_ttt()
    elif args.method == 'TNet':
        e.train_intranetwork_ttt()
    elif args.method == 'TTTz':
        e.train_prenetwork_ttt()
    elif args.method == 'TTTw':
        e.train_prenetwork_w_ttt()
    print('finished train')
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    for i in range(args.n_eval_samples):
        ##comparison methods
        #if args.method in comparison_methods: #= ['normal','coachz','coachw','ttz','ttw']
        if args.method == 'normal':
            e.sample_n_stylegan_images_without_tt()
        if args.method == 'coachz':
            e.sample_n_stylegan_images_with_coachgan()
        if args.method == 'coachw':
            e.sample_n_stylegan_images_with_w_coachgan()
        if args.method == 'ttz':
            e.sample_n_stylegan_images_with_z_tt()
        if args.method == 'ttw':
            e.sample_n_stylegan_images_with_w_ttl()

        ## TTT and TNET  
        if args.method == 'TTTz':
            e.sample_n_stylegan_images_with_prenetwork_ttt()
        if args.method == 'TTTw':
            e.sample_n_stylegan_images_with_post_w_prenetwork_ttt()
        if args.method == 'TNet':
            e.sample_n_stylegan_images_with_intranetwork_ttt()
        if args.method == 'TNet+TTT':
            e.sample_n_stylegan_images_with_pre_and_intranetwork_ttt()

        e.save_results(num=i*args.batch_size)
    #e.calc_metrics()

#parser = argparse.ArgumentParser()
#parser.add_argument('--TT', action='store_true', help='use TT for z')
#parser.add_argument('--TTl', action='store_true', help='use TT-lerp for w')
#parser.add_argument('--TTT', action='store_true', help='use TTT for z')
#parser.add_argument('--w_TTT', action='store_true', help='use TTT for w')
#parser.add_argument('--coach_z', action='store_true', help='use CoachGAN for z')
#parser.add_argument('--coach_w', action='store_true', help='use CoachGAN for w')
from os.path import join
import easydict
args = easydict.EasyDict()

args.repo = './stylegan2'
sys.path.append(args.repo)

import dnnlib
from dnnlib import tflib

tflib.init_tf()

import metrics

#comparison methods
args.truncation = 0.7
args.lr =0.000000001
args.niter = 100
args.batch_size = 2
args.n_eval_samples = 5 
datasets = ['ffhq','cat','horse','church','car']
args.dataset = 'horse'
args.path ='/content/results'

args.base_exp_name='quicksampleshorse'
args.size = 1024 if args.dataset == 'ffhq' else 256
args.checkpoint = 'stylegan2-%s-config-f.pt' % args.dataset
from download import download_file_from_google_drive, gdrive_map
if not os.path.isfile(args.checkpoint):
    print('...Downloading checkpoint...')
    file_id = gdrive_map[args.dataset]
    download_file_from_google_drive(file_id, args.checkpoint)

args.channel_multiplier = 2
args.latent = 512
args.n_mlp = 8
args.device = 'cuda'

## TESTING
#methods = ['TTTz','TTTw','TNet']
methods = ['TTTw','TNet','TTTz']
#methods = ['TNet']
#methods = ['TTTw','TNet','TNet+TTT']
#methods = ['TNet','TNet+TTT']
#methods = ['TNet+TTT']
architectures = ['prelu']
layers = [2]
for m in methods:
    for arch in architectures:
        for nl in layers:
            args.nlayer = nl
            args.arch = arch
            args.method = m

            ## had some trouble with random seeds earlier
            ## forcing back to seed 0 here.
            seed = 0
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
            np.random.seed(seed)  # Numpy module.
            random.seed(seed)  # Python random module.
            torch.manual_seed(seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            ##

            args.arch = arch
            if args.method in ['TTTz','TTTw','TNet','TNet+TTT']:
                args.savedir = join( args.path, args.base_exp_name,  args.method+args.arch+str(args.nlayer))
            else:
                args.savedir = join( args.path, args.base_exp_name,  args.method)
            print(args.savedir)
            if not os.path.exists(args.savedir):
                os.makedirs(args.savedir)
            run(args)

#NOTE: All comparison methods work
#   coachz and ttz give different lookin images (as expected)
comparison_methods = ['normal','coachz','coachw','ttz','ttw']
#comparison_methods = ['normal']
for method in comparison_methods:
    print('method:',method)

    ## Reset random seed
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # multi-GPU.
    np.random.seed(seed)  
    random.seed(seed)  
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    ##

    args.method = method
    args.savedir = join( args.path, args.base_exp_name,  args.method)
    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)
    run(args)
