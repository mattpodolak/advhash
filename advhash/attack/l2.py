import torch
from advhash.attack.base import Attack

class L2Attack(Attack):
    """Implementation of an L2 norm based adversarial collision attack.

    Args:
        hash_fn: String or Hash function. Defines what hashing function to
        perform an attack against.
        split_point (optional): String. Defines what interior function to target as
        part of the attack - required if hash_fn is a string.
        hash_size (optional): Integer. Size of hash created by the hashing function
        - default value is 16.

        device (optional): String. Defines the device to store PyTorch tensors on.

    Raises:
        ValueError: For invalid arguments

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

    def __init__(self, hash_fn, split_point=None, **kwargs):
        if isinstance(hash_fn, str) and split_point is None:
            raise ValueError("split_point is required when specifying hash_fn by name")
        elif split_point is not None:
            kwargs["split_point"] = split_point
        super().__init__(hash_fn, **kwargs)

    def _loss_fn(self, x_adv, x_orig, y_img, c=0.001):
        """
            Loss function used during the creation of the adversarial image

            Args:
                x_adv: PyTorch tensor - adversarial image
                x_orig: PyTorch tensor - unperturbed image
                y_img: PyTorch tensor - target image
                c (optional): content loss parameter - defaults to 0.001

        """
        x_hash = self.hash.partial_hash(x_adv)
        y_hash = self.hash.partial_hash(y_img)
        return torch.norm(x_hash-y_hash, 2) + c*torch.norm(x_adv-x_orig, 2)
