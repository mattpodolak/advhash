import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import imagehash
import torch
from tqdm.notebook import trange
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = "True" # set env variable to deal with matplotlib and torch issue

def rgb2luma(x): # works, is differentiable
    """
        x: float tensor
        returns float tensor
    """
    mod = torch.ones(x.shape).cuda()
    add = torch.zeros(x.shape).cuda()
    add[:,:,2] = (torch.ones(add[:,:,2].shape).cuda() * 32768) / pow(2, 16)
    mod[:,:,0] = torch.mul(mod[:,:,0],19595)
    mod[:,:,1] = torch.mul(mod[:,:,1],38470) 
    mod[:,:,2] = torch.mul(mod[:,:,2],7471)
    mod = torch.div(mod, pow(2, 16))
    return torch.sum(torch.add(torch.mul(x, mod),add), dim=-1)

def box(x, boxmin=0,boxmax=255):
    """
        Convert [-inf,inf] to [min, max] using tanh
    """
    x = torch.tanh(x) # map [-inf,inf] to [-1,1]
    grad = 1 - torch.square(x)
    boxmul = (boxmax - boxmin) / 2.
    boxplus = (boxmin + boxmax) / 2.
    x = x*boxmul+boxplus # map [-1, +1] to [min, max]
    grad=grad*boxmul
    return x

def box_conv(x, boxmin=0,boxmax=255):
    """
        Convert [min, max] to [-inf,inf] using arctanh
    """
    boxmul = (boxmax - boxmin) / 2.
    boxplus = (boxmin + boxmax) / 2.
    x = (x - boxplus) / boxmul # map [min, max] to [-1, +1]
    return torch.arctanh(x*0.999999) # map [-1,1] to [-inf,inf]

def dist_check(X, Y):
    X = Image.fromarray((X).cpu().detach().numpy().astype('uint8'))
    Y = Image.fromarray((Y).cpu().detach().numpy().astype('uint8'))
    X=torch.tensor(imagehash.dhash(X, hash_size=16).hash, dtype=torch.float32).cuda()
    Y=torch.tensor(imagehash.dhash(Y, hash_size=16).hash, dtype=torch.float32).cuda()
    return torch.sum(torch.abs(X - Y))

def sinc(x):
    if (x == 0.0):
        return torch.tensor(1.0)
    x = x * np.pi
    return np.sin(x) / x

def lanczos3(x):
    if(-3.0 <= x and x < 3.0):
        return sinc(x)*sinc(x/3.0)
    else:
        return torch.tensor(0.0)
    
def precompute_coeffs(inSize, outSize):
    filterscale = scale = inSize / outSize
    filterscale = max(1, filterscale)
    support = 3 * filterscale # 3 for lanczos kernel size of 3
    ksize = int(np.ceil(support) * 2 + 1)
    kk = torch.zeros((outSize, inSize)).cuda()
    for i in range(outSize):
        center = (i+0.5)*filterscale
        ww = 0.0
        ss = 1.0 / filterscale
        _min = max(0, int(center-support+0.5))
        _max = min(inSize, int(center+support+0.5))
        for j in range(_min, _max):
            kk[i][j] = lanczos3((j-center+0.5)*ss)
        kk[i] = torch.div(kk[i], torch.sum(kk[i]))
    return kk

def resample(imIn, kkx, kky, xsize=17, ysize=16):
    im_y, im_x = imIn.shape
    if(im_x != xsize):
        imIn = torch.clip(torch.matmul(imIn, torch.transpose(kkx, 0, 1)), 0, 255)
    if(im_y != ysize):
        imIn = torch.clip(torch.matmul(kky, imIn), 0, 255)
    return imIn

def lanczos_resize(X, xsize=17, ysize=16):
    im_y, im_x = X.shape
    kkx = precompute_coeffs(im_x, xsize)
    kky = precompute_coeffs(im_y, ysize)
    return resample(X, kkx, kky)

def actual_hash_fn(X):
    X = rgb2luma(X)
    X = lanczos_resize(X)
    return (X[:, 1:] > X[:, :-1]).type(torch.float32)

def short_dist_check(X, Y):
    X_hash=actual_hash_fn(X)
    Y_hash=actual_hash_fn(Y)
    return torch.sum(torch.abs(X_hash - Y_hash))

def loss_hash_fn(X):
    X = rgb2luma(X)
    X = lanczos_resize(X)
    return X #X[:, 1:] - X[:, :-1]

def content_loss(X_orig, X_pert):
    return torch.mean(torch.abs(X_pert - X_orig)/X_orig)

def torch_norm(X): # works is differentiable
    avoid_zero_div = torch.tensor(1e-12)
    return torch.sqrt(torch.maximum(torch.sum(X**2), avoid_zero_div))

def loss_fn(X, X_orig, kkx, kky, Y, c=0.001):
    x = rgb2luma(X)
    X_hash = resample(x, kkx, kky)
    return torch.norm(X_hash-Y, 2) + c*torch.norm(X-X_orig, 2)

def train_fn(X, Y, boxmin=0, boxmax=255, epochs=1000, lr=5, c=0.001):
    metrics = {'distance': [], 'loss': [], 'content_loss':[]}
    w = box_conv(X, boxmin, boxmax)
    X = box(w, boxmin, boxmax)
    Y_hash = loss_hash_fn(Y)
    w.requires_grad = True
    im_y, im_x, _ = X.shape
    kkx = precompute_coeffs(im_x, 17)
    kky = precompute_coeffs(im_y, 16)
    opt = torch.optim.Adam([w], lr=lr, betas=(0.1,0.1))
    with trange(epochs) as t:
        for epoch in t:
            im_adv = box(w, boxmin, boxmax)
            opt.zero_grad()
            loss = loss_fn(im_adv, X, kkx, kky, Y_hash, c)
            dist = dist_check(im_adv, Y)
            c_loss = content_loss(X, im_adv)
            metrics['distance'].append(dist.item())
            metrics['loss'].append(loss.item())
            metrics['content_loss'].append(c_loss.item())
            t.set_description(f'Epoch {epoch+1}: loss={loss:.4f} dist={dist} c_loss={100*c_loss:.2f}% ')
            if(dist == 0):
                break
            loss.backward()
            opt.step()

    return box(w, boxmin, boxmax), metrics

def quick_plot(X):
    plt.imshow(X.cpu().detach().numpy(), cmap="gray")

Y_img = Image.open('./img/forest.jpg')
Y = torch.tensor(np.asarray(Y_img).astype('float32')).cuda()

X_img = Image.open('./img/cat.jpg')
X = torch.tensor(np.asarray(X_img).astype('float32')).cuda()

im_adv, metrics = train_fn(X, Y, 0, 255, lr=0.005, epochs=1000, c=0.001)

quick_plot(im_adv/255)

convertedAdv = torch.clip(torch.round(im_adv), 0, 255).cpu().detach().numpy().astype('uint8')

testRaw = Image.fromarray(convertedAdv)
hashRaw = torch.tensor(imagehash.dhash(testRaw, hash_size=16).hash, dtype=torch.float32).cuda()
hashTarget = torch.tensor(imagehash.dhash(Y_img, hash_size=16).hash, dtype=torch.float32).cuda()

torch.sum(torch.abs(hashRaw - hashTarget))