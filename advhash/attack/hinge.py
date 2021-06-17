import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import imagehash
import torch
from tqdm.notebook import trange

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
    
def box(x, boxmin=0,boxmax=255):
    boxmul = (boxmax - boxmin) / 2.
    boxplus = (boxmin + boxmax) / 2.
    return torch.tanh((x)/boxmul) * boxmul + boxplus

def rgb2luma(x): # works, is differentiable
    """
        x: float tensor
        returns float tensor
    """
    return ((x[:,:,0] * 19595 + x[:,:,1] * 38470 + x[:,:,2] * 7471 + 32768)/ pow(2, 16))

def rgb2luma2(x): # works, is differentiable
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

def hamming_dist(X,Y):
    return torch.sum(torch.abs(X - Y))

def precompute_coeffs(inSize, in0, in1, outSize):
    filterscale = scale = (in1 - in0) / outSize
    support = 3 * filterscale # 3 from lanczos
    ksize = int(np.ceil(support) * 2 + 1)
#     kk = [0]*(ksize*(outSize))
    kk = []
    bounds = []
    for i in range(outSize):
        center = in0 + (i+0.5)*filterscale
        ww = 0.0
        ss = 1.0 / filterscale
        xmin = max(0, int(center-support+0.5))
        xmax = min(inSize, int(center+support+0.5))
        xmax -= xmin
        bounds.extend([xmin, xmax])
        k = []
        for x in range(xmax):
            w = lanczos3((x + xmin-center+0.5)*ss)
            ww+=w
            k.append(w)
        for x in range(xmax):
            if(ww != 0.0):
                k[x] /= ww
                
        for _ in range(xmax, ksize):
            k.append(0)

        kk.extend(k)
    return ksize, kk, bounds

def clip8(inVal):
    return torch.clip(inVal / pow(2,22), 0, 255)

def norm_coeffs_8bpc(outSize, ksize, prekk):
    kk = torch.zeros((outSize*ksize,)).cuda()
    for x in range(outSize*ksize):
        if (prekk[x] < 0):
            kk[x] = (int(-0.5 + prekk[x] * (1 << 22))) # PRECISON_BITS = 32-8-2
        else:
            kk[x] = (int(0.5 + prekk[x] * (1 << 22))) # PRECISON_BITS = 32-8-2
    return kk  


def resampleVertical_8bpc(imOut, imIn, offset, ysize, xsize, ksize, bounds, kk):
    for yy in range(ysize):
        ymin = bounds[yy * 2 + 0]
        ymax = bounds[yy * 2 + 1]
        for xx in range(xsize):
            a = imIn[ymin:ymin+ymax, xx]
            b = kk[(yy * ksize):(yy * ksize+ymax)]
            imOut[yy][xx] = clip8(pow(2, 21) + torch.sum(torch.mul(a,b)))
    return imOut

def resampleHorizontal_8bpc(imOut, imIn, offset, ysize, xsize, ksize, bounds, kk):
    for yy in range(ysize):
        for xx in range(xsize):
            xmin = bounds[xx * 2 + 0]
            xmax = bounds[xx * 2 + 1]
            a = imIn[yy + offset,xmin:(xmin+xmax)]
            b = kk[(xx*ksize):(xx*ksize+xmax)]
            imOut[yy][xx] = clip8(pow(2, 21) + torch.sum(torch.mul(a,b)))
    return imOut

def image_resample(imIn, xsize, ysize, kk_horiz, bounds_horiz, ksize_horiz, kk_vert, bounds_vert, ksize_vert):
    im_y, im_x = imIn.shape
    box = (0, 0, im_x, im_y)
    need_horizontal = xsize != im_x or box[0] or box[2] != xsize
    need_vertical = ysize != im_y or box[1] or box[3] != ysize
    imTemp = imOut = None # initialize image vars
    
    # First used row in the source image
    ybox_first = bounds_vert[0]
    # Last used row in the source image
    ybox_last = bounds_vert[ysize * 2 - 2] + bounds_vert[ysize * 2 - 1]
    
    # two-pass resize, horizontal pass
    if (need_horizontal):
        # Shift bounds for vertical pass
        for i in range(ysize):
            bounds_vert[i * 2] -= ybox_first
        
        imTemp = torch.zeros((ybox_last - ybox_first, xsize)).cuda()
        imTemp = resampleHorizontal_8bpc(imTemp, imIn, ybox_first, (ybox_last - ybox_first), xsize, ksize_horiz, bounds_horiz, kk_horiz)
        del bounds_horiz
        del kk_horiz
        imOut = imIn = imTemp
    else:
        # Free in any case
        del bounds_horiz
        del kk_horiz

    # vertical pass
    if (need_vertical):
        imOut = torch.zeros((ysize,xsize)).cuda()
        # imIn can be the original image or horizontally resampled one
        imOut = resampleVertical_8bpc(imOut, imIn, 0, ysize, xsize, ksize_vert, bounds_vert, kk_vert)
        del bounds_vert
        del kk_vert
    else:
        # Free in any case
        del bounds_vert
        del kk_vert
        
    # none of the previous steps are performed, copying
    if (imOut is None):
        imOut = imIn

    return imOut

def box(x, boxmin=0,boxmax=255):
    boxmul = (boxmax - boxmin) / 2.
    boxplus = (boxmin + boxmax) / 2.
    return torch.tanh((x)/boxmax) * boxmul + boxplus

def box_conv(x, boxmin=0,boxmax=255):
    boxmul = (boxmax - boxmin) / 2.
    boxplus = (boxmin + boxmax) / 2.
    return torch.arctanh((x - boxplus) / boxmul * 0.999999)

def loss_fn_final(X, Y, horiz_coeffs, vert_coeffs):
    avoid_zero_div = torch.tensor(1e-12)
    x = rgb2luma2(X)
    x = image_resample(x, 17, 16, *horiz_coeffs, *vert_coeffs)
    dx = x[:, 1:] - x[:, :-1]
    f = torch.sigmoid(dx)
    loss = torch.subtract(torch.abs(Y - f), 0.45)
    return torch.sum(torch.maximum(loss, avoid_zero_div))

def train_final(X, Y, epsilon=5, epochs=1000,lr=5):
    b = torch.tensor(np.random.uniform(-epsilon, epsilon, X.shape), requires_grad=True, device="cuda")
#     b = ((-epsilon - epsilon) * torch.rand(X.shape) + epsilon).clone().detach().requires_grad_(True).cuda()
    opt = torch.optim.Adam([b], lr=lr, betas=(0.1,0.1))
    # precompute coeffs
    im_y, im_x, im_z = X.shape
    ysize, xsize = Y.shape
    ksize_horiz, kk_horiz, bounds_horiz = precompute_coeffs(im_x, 0, im_x, xsize+1)
    ksize_vert, kk_vert, bounds_vert = precompute_coeffs(im_y, 0, im_y, ysize)
    # normalize coeffs
    kk_horiz = norm_coeffs_8bpc(xsize+1, ksize_horiz, kk_horiz)
    kk_vert = norm_coeffs_8bpc(ysize, ksize_vert, kk_vert)
    horiz_coeffs = (kk_horiz, bounds_horiz, ksize_horiz)
    vert_coeffs = (kk_vert, bounds_vert, ksize_vert)
    last_loss = torch.zeros((1,)).cuda()
    with trange(epochs) as t:
        for epoch in t:
            x = X+b
            loss = loss_fn_final(x, Y, horiz_coeffs, vert_coeffs)
            opt.zero_grad()
            t.set_description(f'Epoch {epoch+1}: loss={loss} ')
            loss.backward()
            opt.step()
            if(torch.abs(last_loss - loss) < 0.0001):
                return b
            else:
                last_loss = loss
    return b

Y_img = Image.open('./img/forest.jpg')
X_img = Image.open('./img/cat.jpg')
target = torch.tensor(imagehash.dhash(Y_img, hash_size=16).hash, dtype=torch.float32).cuda()
train = torch.tensor(np.array(X_img), dtype=torch.float32).cuda()

b = train_final(train, target, epochs=100,lr=10)

testRaw = train+b
print(f'max: {torch.max(testRaw)} min: {torch.min(testRaw)}')
testRaw = Image.fromarray(testRaw.cpu().detach().numpy().astype('uint8'))

rawHash=torch.tensor(imagehash.dhash(testRaw, hash_size=16).hash, dtype=torch.float32)
hamming_dist(rawHash,target.cpu())
testRaw.save("./img/raw_full.jpg")