import abc
import torch
from tqdm import trange
from advhash import hashes
from advhash.utils.constrain import box, box_conv
from advhash.utils.compare import avg_diff, hamming_dist

class Attack:
    """Base class for adversarial attacks

    Args:
        hash_fn: String or Hash function. Defines what hashing function to
        perform an attack against.
        other_args (optional): additional keyword arguments to accept

    Raises:
        ValueError: For invalid arguments
    """

    def __init__(self, hash_fn, other_args=None, **kwargs):
        allowed_args = {"hash_size", "split_point", "device"}
        if isinstance(other_args, set):
            allowed_args.update(other_args)
        for kw in kwargs:
            if kw not in allowed_args:
                raise ValueError(f"Invalid argument {kw}")

        self.metrics = {'distance': [], 'loss': [], 'avg_diff':[]}
        self.hash = self._get_hash(hash_fn, **kwargs)

    def _get_hash(self, hash_fn, **kwargs):
        hash_cls = hashes.get(hash_fn, kwargs)
        if hash_cls is None:
            raise ValueError(f'Failed to initialize hash function for {hash_fn}')

        config = hash_cls.get_config()
        for key in kwargs:
            if str(config[key]) != str(kwargs[key]):
                raise ValueError(f'Expected hash {key}={kwargs[key]}, instead got {config[key]}')

        return hash_cls

    def _update_metrics(self, **kwargs):
        for metric in kwargs:
            if metric in self.metrics:
                self.metrics[metric] = kwargs[metric]

    @abc.abstractmethod
    def _loss_fn(self, x_adv, x_orig, y, **kwargs):
        """Attack loss function for optimization

        Returns:
            PyTorch tensor with the loss value
        """

    def attack(self, x, y, epochs=1000, lr=0.005, betas=(0.1, 0.1), target_dist=0, **kwargs):
        """Perform an adversarial collision attack on an image hashing function

        Args:
            x: PyTorch tensor - source image
            y: PyTorch tensor - target image
            epochs (optional): number of iterations for generating the image - defaults to 1000
            lr (optional): learning rate for adam optimizer - defaults to 0.005
            betas (optional): tuple of betas for the adam optimizer - defaults to (0.1, 0.1)
            target_dist (optional): exit early if the hamming distance <= target_dist -
            defaults to 0

        Returns:
            PyTorch tensor of adversarial image

        Raise:
            ValueError: for invalid arguments

        """
        if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
            raise ValueError("Expected torch.tensor for x and y")

        x = x.to(self.hash.device)
        y = y.to(self.hash.device)

        w = box_conv(x)
        x_orig = box(w)
        y_hash = self.hash.full_hash(y)
        w.requires_grad = True
        self.opt = torch.optim.Adam([w], lr=lr, betas=betas)
        with trange(epochs) as t:
            for epoch in t:
                x_adv = box(w)
                self.opt.zero_grad()
                loss = self._loss_fn(x_adv, x_orig, y, **kwargs)
                x_hash = self.hash.full_hash(x_adv)
                dist = hamming_dist(x_hash, y_hash)
                diff = avg_diff(x_orig, x_adv)
                self._update_metrics(distance=dist.item(), loss=loss.item(), avg_diff=diff.item())
                t.set_description(f'Epoch {epoch+1}: loss={loss:.4f} \
                    dist={dist} diff={100*diff:.2f}% ')
                if dist <= target_dist:
                    break
                loss.backward()
                self.opt.step()
        return box(w)
