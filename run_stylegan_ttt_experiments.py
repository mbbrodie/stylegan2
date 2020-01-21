### this will be copied to colab
### everything else imported
from StyleGAN2TTTExperiment import StyleGAN2TTTExperiment as e
import os.path.join as join
import easydict
args = easydict.EasyDict()
args.n_eval_samples = 10
args.path = '' #gdrive probably
import metrics

#comparison methods
comparison_methods = ['normal','coachz','coachw','ttz','ttw']
for method in comparison_methods:
    args.method = method
    args.savedir = join( args.path, args.base_exp_name,  method)
    run(args)

## SAMPLE
experiments= [
        'normal',
        'coachz',
        'coachw',
        'ttz',
        'ttw',
        'TTTz',
        'TTTw',
        'TNet',
        'TNet+TTT',
        ]
layers = [2,4,8,16,32,64,128,256]
#a\item BPF + x
#b\item BPF-BF + x
#c\item BPF-BPF$_{bottleneck}$-BF + 
#d\item FBP + x
#e\item FBP-FB + x
#f\item FBP-F_${bottleneck}$BP-FB + 
blocks = ['a','b','c','d','e','f','prelu']
#our methods
#if args.train ==> train then sample
for nl in nlayer
    args.nlayer = nl
    for arch in architectures:
        args.arch = arch
        for method in methods:
            args.method = method
            args.savedir = join( args.base_exp_name,  method)
            run(args)

def run(args):
    e.set_args(args)
    ## SETUP
    if args.TTT:
        setup_prenetwork_ttt()
    if args.W_TTT:
        setup_prenetwork_w_ttt()
    if args.TNET:
        setup_intranetwork_ttt()

    ## TRAIN
    # may need to set train_sample_size
    if args.TNET and args.TTT and args.W_TTT: 
        train_prenetwork_and_intranetwork_ttt()
    elif args.TNET:
        train_intranetwork_ttt()
    elif args.TTT:
        train_prenetwork_ttt()
    elif args.W_TTT:
        train_prenetwork_w_ttt()
        
    for i in range(args.n_eval_samples):
        ##comparison methods
        #if args.method in comparison_methods: #= ['normal','coachz','coachw','ttz','ttw']
        if args.method == 'normal':
            gen_from_w() # normal
        if args.method == 'coachz':
            sample_n_stylegan_images_with_coachgan()
        if args.method == 'coachw':
            sample_n_stylegan_images_with_w_coachgan()
        if args.method == 'ttz':
            sample_n_stylegan_images_with_z_tt()
        if args.method == 'ttw':
            sample_n_stylegan_images_with_w_ttl()

        ## TTT and TNET  
        if args.method == 'TTTz':
            sample_n_stylegan_images_with_prenetwork_ttt()
        if args.method == 'TTTw':
            sample_n_stylegan_images_with_post_w_prenetwork_ttt()
        if args.method == 'TNet':
            sample_n_stylegan_images_with_intranetwork_ttt()
        if args.method == 'TNet+TTT':
            sample_n_stylegan_images_with_pre_and_intranetwork_ttt()

        save_results()
    metrics.calc_metrics()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--TT', action='store_true', help='use TT for z')
    parser.add_argument('--TTl', action='store_true', help='use TT-lerp for w')
    parser.add_argument('--TTT', action='store_true', help='use TTT for z')
    parser.add_argument('--w_TTT', action='store_true', help='use TTT for w')
    parser.add_argument('--coach_z', action='store_true', help='use CoachGAN for z')
    parser.add_argument('--coach_w', action='store_true', help='use CoachGAN for w')

