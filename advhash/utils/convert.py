import torch

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