import torch
import numpy as np

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

def precompute_coeffs(inSize, outSize, device):
    filterscale = inSize / outSize
    filterscale = max(1, filterscale)
    support = 3 * filterscale # 3 for lanczos kernel size of 3
    kk = torch.zeros((outSize, inSize)).to(device)
    for i in range(outSize):
        center = (i+0.5)*filterscale
        ss = 1.0 / filterscale
        _min = max(0, int(center-support+0.5))
        _max = min(inSize, int(center+support+0.5))
        for j in range(_min, _max):
            kk[i][j] = lanczos3((j-center+0.5)*ss)
        kk[i] = torch.div(kk[i], torch.sum(kk[i]))
    return kk

def resample(imIn, kkx, kky, xsize=17, ysize=16, maxval=255):
    im_y, im_x = imIn.shape
    if(im_x != xsize):
        imIn = torch.clip(torch.matmul(imIn, torch.transpose(kkx, 0, 1)), 0, maxval)
    if(im_y != ysize):
        imIn = torch.clip(torch.matmul(kky, imIn), 0, maxval)
    return imIn

def lanczos_resize(X, xsize=17, ysize=16):
    im_y, im_x = X.shape
    kkx = precompute_coeffs(im_x, xsize, X.device)
    kky = precompute_coeffs(im_y, ysize, X.device)
    return resample(X, kkx, kky)