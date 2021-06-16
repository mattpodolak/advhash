import torch

def box(x, boxmin=0,boxmax=255):
    """
        Convert [-inf,inf] to [min, max] using tanh
    """
    x = torch.tanh(x) # map [-inf,inf] to [-1,1]
    boxmul = (boxmax - boxmin) / 2.
    boxplus = (boxmin + boxmax) / 2.
    return x*boxmul+boxplus # map [-1, +1] to [min, max]

def box_conv(x, boxmin=0,boxmax=255):
    """
        Convert [min, max] to [-inf,inf] using arctanh
    """
    boxmul = (boxmax - boxmin) / 2.
    boxplus = (boxmin + boxmax) / 2.
    x = (x - boxplus) / boxmul # map [min, max] to [-1, +1]
    return torch.arctanh(x*0.999999) # map [-1,1] to [-inf,inf]
