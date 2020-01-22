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
args.lr =0.001
args.niter = 50#100#0
args.batch_size = 2
args.n_eval_samples = 10#00
datasets = ['ffhq','cat','horse','church','car']
args.dataset = 'church'
args.path ='/content/results'

args.base_exp_name='testing50_iter_lr_001'
args.size = 1024 if args.dataset == 'ffhq' else 256
args.checkpoint = 'stylegan2-%s-config-f.pt' % args.dataset
args.channel_multiplier = 2
args.latent = 512
args.n_mlp = 8
args.device = 'cuda'

## TESTING
#methods = ['TTTz','TTTw','TNet','TNet+TTT']
#TTTw isn't working right now
methods = ['TTTw','TNet','TNet+TTT']
architectures = ['prelu','a','b','c','d','e','f']
layers = [2,4,8,16,32,64,128,256]
for m in methods:
    for arch in architectures:
        for nl in layers:
            args.nlayer = nl
            args.arch = arch
            args.method = m

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
            print(arch)
            if args.method in ['TTTz','TTTw','TNet','TNet+TTT']:
                args.savedir = join( args.path, args.base_exp_name,  args.method+args.arch+str(args.nlayer))
            else:
                args.savedir = join( args.path, args.base_exp_name,  args.method)
            if not os.path.exists(args.savedir):
                os.makedirs(args.savedir)
            run(args)

#NOTE: All comparison methods work
#   coachz and ttz give different lookin images (as expected)
comparison_methods = ['normal','coachz','coachw','ttz','ttw']
for method in comparison_methods:
    print('method:',method)
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    args.method = method
    args.savedir = join( args.path, args.base_exp_name,  args.method)
    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)
    run(args)


exit()

## SAMPLE
#a\item BPF + x
#b\item BPF-BF + x
#c\item BPF-BPF$_{bottleneck}$-BF + 
#d\item FBP + x
#e\item FBP-FB + x
#f\item FBP-F_${bottleneck}$BP-FB + 
#our methods
#if args.train ==> train then sample
