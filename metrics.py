# metrics
# given args.path to images
# given args.metrics can contain ['fid','lpips']
from fid import *
def fid(args):
    args.inception  
    features = extract_feature_from_saved_samples(
        args.img_path, args.inception, args.truncation, mean_latent, args.batch, args.n_sample, device
    ).numpy()
@torch.no_grad()
def extract_feature_from_saved_samples(
    path, inception, batch_size, n_sample, device
):
    n_batch = n_sample // batch_size
    resid = n_sample - (n_batch * batch_size)
    batch_sizes = [batch_size] * n_batch + [resid]
    features = []

    #for batch in tqdm(batch_sizes):
    #latent = torch.randn(batch, 512, device=device)
    #img, _ = g([latent], truncation=truncation, truncation_latent=truncation_latent)
    files = list(path.glob('*.jpg')) + list(path.glob('*.png'))
    for i in tqdm(range(n_batch)):
        if verbose:
            print('\rPropagating batch %d/%d' % (i + 1, n_batches),
                  end='', flush=True)
        start = i * batch_size
        end = start + batch_size

        images = np.array([imread(str(f)).astype(np.float32)
                           for f in files[start:end]])

        # Reshape to (n_images, 3, height, width)
        images = images.transpose((0, 3, 1, 2)) 
        images /= 255 

        img = torch.from_numpy(images).type(torch.FloatTensor)
        if cuda:
            batch = batch.cuda()

        feat = inception(img)[0].view(img.shape[0], -1)
        features.append(feat.to('cpu'))

    features = torch.cat(features, 0)

    return features


