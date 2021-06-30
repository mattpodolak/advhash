from PIL import Image
import torch
import numpy as np

def rgb2luma(x):
    """Convert a tensor from RGB to Luma

    Args:
        x: PyTorch tensor with shape (:, :, 3)
        device (optional): PyTorch device for the tensor - defaults
        to CPU

    Returns:
        PyTorch tensor with shape (:, :, 1)
    """

    mod = torch.ones(x.shape).to(x.device)
    add = torch.zeros(x.shape).to(x.device)
    add[:,:,2] = (torch.ones(add[:,:,2].shape) * 32768) / pow(2, 16)
    mod[:,:,0] = torch.mul(mod[:,:,0],19595)
    mod[:,:,1] = torch.mul(mod[:,:,1],38470)
    mod[:,:,2] = torch.mul(mod[:,:,2],7471)
    mod = torch.div(mod, pow(2, 16))
    return torch.sum(torch.add(torch.mul(x, mod),add), dim=-1)

def tensor2image(x):
    """Convert a tensor into an Image

    Args:
        x: PyTorch tensor

    Returns:
        Pillow Image

    """
    x = torch.clip(torch.round(x), 0, 255).cpu().detach().numpy().astype('uint8')
    return Image.fromarray(x)

def image2tensor(x, device=None):
    """Convert an Image to a tensor

    Args:
        x: Pillow Image
        device (optional): PyTorch device for the tensor - defaults
        to CPU

    Returns:
        PyTorch float32 tensor on the chosen device

    """
    if device is None:
        device = torch.device('cpu')

    return torch.tensor(np.asarray(x).astype('float32')).to(device)
