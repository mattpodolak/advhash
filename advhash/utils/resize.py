import torch
import numpy as np

def sinc(x):
    if x == 0.0:
        return torch.tensor(1.0)
    x = x * np.pi
    return np.sin(x) / x

def lanczos3(x):
    if(-3.0 <= x and x < 3.0):
        return sinc(x)*sinc(x/3.0)
    return torch.tensor(0.0)

def precompute_coeffs(in_size, out_size, device):
    filterscale = in_size / out_size
    filterscale = max(1, filterscale)
    support = 3 * filterscale # 3 for lanczos kernel size of 3
    kk = torch.zeros((out_size, in_size)).to(device)
    for i in range(out_size):
        center = (i+0.5)*filterscale
        ss = 1.0 / filterscale
        _min = max(0, int(center-support+0.5))
        _max = min(in_size, int(center+support+0.5))
        for j in range(_min, _max):
            kk[i][j] = lanczos3((j-center+0.5)*ss)
        kk[i] = torch.div(kk[i], torch.sum(kk[i]))
    return kk

def resample(im_in, kkx, kky, xsize=17, ysize=16, maxval=255):
    im_y, im_x = im_in.shape
    if im_x != xsize:
        im_in = torch.clip(torch.matmul(im_in, torch.transpose(kkx, 0, 1)), 0, maxval)
    if im_y != ysize:
        im_in = torch.clip(torch.matmul(kky, im_in), 0, maxval)
    return im_in

def lanczos_resize(X, xsize=17, ysize=16):
    im_y, im_x = X.shape
    kkx = precompute_coeffs(im_x, xsize, X.device)
    kky = precompute_coeffs(im_y, ysize, X.device)
    return resample(X, kkx, kky)
