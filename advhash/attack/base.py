import abc
import torch
from advhash import hashes
from tqdm import trange
from advhash.utils.constrain import box, box_conv
from advhash.utils.compare import avg_diff
from advhash.attack.hinge import hamming_dist

class Attack(object):
    """Base class for adversarial attacks

    Args:
        hash_fn: String or Hash function. Defines what hashing function to
        perform an attack against.
        hash_size (optional): Integer. Size of hash created by the hashing function 
        - default value is 16.
        split_point (optional): String. Defines what interior function to target as
        part of the attack - default value is None.
        device: String. Defines the device to store PyTorch tensors on.

    Raises:
        ValueError: For invalid arguments
    """

    def __init__(self, hash_fn, hash_size=16, split_point=None, device= 'cuda'):
        if device == "cuda":
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                raise ValueError('Cuda is not available on this computer')
        elif device == 'cpu':
            self.device = torch.device('cpu')
        else:
            raise ValueError(f'device should equal cuda or cpu, instead got {device}')

        self.metrics = {'distance': [], 'loss': [], 'avg_diff':[]}
        self.hash = self._get_hash(hash_fn, hash_size, split_point)

    def _get_hash(hash_fn, hash_size, split_point=None):
        hash_cls = hashes.get(hash_fn)
        if(hash_cls is None):
            raise ValueError(f'Failed to initialize hash function for {hash_fn}')
        return hash_cls(split_point=split_point, hash_size=hash_size)

    def _update_metrics(self, **kwargs):
        for metric in kwargs:
            if metric in self.metrics:
                self.metrics[metric] = kwargs[metric]

    @abc.abstractmethod
    def _loss_fn(self):
        """Attack loss function for optimization

        Returns:
            PyTorch tensor with the loss value
        """

        return None

    def attack(self, X, Y, epochs=1000, lr=0.005, betas=(0.1, 0.1), target_dist=0, **kwargs):
        """Perform an adversarial collision attack on an image hashing function

        Args:
            X: PyTorch tensor - source image 
            Y: PyTorch tensor - target image
            epochs (optional): number of iterations for generating the image - defaults to 1000
            lr (optional): learning rate for adam optimizer - defaults to 0.005
            betas (optional): tuple of betas for the adam optimizer - defaults to (0.1, 0.1)
            target_dist (optional): exit early if the hamming distance <= target_dist - defaults to 0

        Returns:
            PyTorch tensor of adversarial image

        Raise:
            ValueError: for invalid arguments

        """
        if type(X) != torch.tensor or type(Y) != torch.tensor:
            raise ValueError("Expected torch.tensor for X and Y")

        if X.device != self.device or Y.device != self.device:
            raise ValueError(f"Expected X and Y to be on {self.device}")

        w = box_conv(X)
        X_orig = box(w)
        Y_hash = self.hash.full_hash(Y)
        w.requires_grad = True
        self.opt = torch.optim.Adam([w], lr=lr, betas=betas)
        with trange(epochs) as t:
            for epoch in t:
                X_adv = box(w)
                self.opt.zero_grad()
                loss = self._loss_fn(X_adv, X_orig, Y, **kwargs)
                X_hash = self.hash.full_hash(X_adv)
                dist = hamming_dist(X_hash, Y_hash)
                diff = avg_diff(X_orig, X_adv)
                self._update_metrics(distance=dist.item(), loss=loss.item(), avg_diff=diff.item())
                t.set_description(f'Epoch {epoch+1}: loss={loss:.4f} dist={dist} diff={100*diff:.2f}% ')
                if(dist <= target_dist):
                    break
                loss.backward()
                self.opt.step()
        return box(w)
