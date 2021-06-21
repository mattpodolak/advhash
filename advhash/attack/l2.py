from advhash.attack.base import Attack
import os
import torch

os.environ['KMP_DUPLICATE_LIB_OK'] = "True" # set env variable to deal with matplotlib and torch issue

class L2Attack(Attack):
    """Implementation of an L2 norm based attack.

    Reference:
        Dolhansky, B., & Canton-Ferrer, C. (2020, November). Adversarial collision attacks 
        on image hashing functions. https://arxiv.org/pdf/2011.09473.pdf

    
    Example:
        ```
        from advhash.attack.l2 import L2Attack
        import torch
        from PIL import Image

        target = Image.open('forest.jpg')
        source = Image.open('cat.jpg')

        l2 = L2Attack(hash_fn='dhash', split_point='resize')

        im_adv = l2.attack(target, source)

        ```

    """
    def __init__(self, hash_fn, split_point, hash_size=16, device='cuda'):
        super().__init__(hash_fn, hash_size, split_point, device)

    def _loss_fn(self, X_adv, X_orig, Y_img, c=0.001):
        """
            Loss function used during the creation of the adversarial image

            Args:
                X_adv: PyTorch tensor - adversarial image
                X_orig: PyTorch tensor - unperturbed image
                Y_img: PyTorch tensor - target image
                c (optional): content loss parameter - defaults to 0.001

        """
        X_hash = self.hash.partial_hash(X_adv)
        Y_hash = self.hash.partial_hash(Y_img)
        return torch.norm(X_hash-Y_hash, 2) + c*torch.norm(X_adv-X_orig, 2)