import matplotlib.pyplot as plt
import torch

def plot_tensor(X, cmap="gray"):
    """Plot a tensor as an image.

    Args:
        X: PyTorch tensor to plot
        cmap (optional): String value of matplotlib color map to use
        - defaults to "gray".

    """
    if X.dtype == torch.float and torch.max(X) > 1:
        plt.imshow((X/255).cpu().detach().numpy(), cmap=cmap)

    else:
        plt.imshow(X.cpu().detach().numpy(), cmap=cmap)