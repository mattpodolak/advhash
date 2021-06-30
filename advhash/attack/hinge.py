import torch
from advhash.attack.base import Attack

class HingeAttack(Attack):
    """Implementation of a hinge loss based adversarial collision attack.

    Args:
        hash_fn: String or Hash function. Defines what hashing function to
        perform an attack against.
        hash_size (optional): Integer. Size of hash created by the hashing function
        - default value is 16.
        part of the attack - default value is None.
        device (optional): String. Defines the device to store PyTorch tensors on.

    Raises:
        ValueError: For invalid arguments

    Reference:
        Dolhansky, B., & Canton-Ferrer, C. (2020, November). Adversarial collision attacks
        on image hashing functions. https://arxiv.org/pdf/2011.09473.pdf


    Example:
        ```
        from advhash.attack.hinge import HingeAttack
        import torch
        from PIL import Image

        target = Image.open('forest.jpg')
        source = Image.open('cat.jpg')

        hinge = HingeAttack(hash_fn='dhash', split_point='resize')

        im_adv = hinge.attack(target, source)

        ```

    """
    def __init__(self, hash_fn, **kwargs):
        if "split_point" in kwargs:
            raise ValueError("Invalid argument split_point")
        super().__init__(hash_fn, split_point="last", **kwargs)

    def _loss_fn(self, x_adv, x_orig, y_img, d=0.45):
        """
            Loss function used during the creation of the adversarial image

            Args:
                x_adv: PyTorch tensor - adversarial image
                x_orig: PyTorch tensor - unperturbed image
                y_img: PyTorch tensor - target image
                d (optional): delta parameter - defaults to 0.45

        """
        avoid_zero_div = torch.tensor(1e-12) # avoid gradient issue at 0
        x_hash = torch.sigmoid(self.hash.partial_hash(x_adv))
        y_hash = self.hash.full_hash(y_img)
        loss = torch.subtract(torch.abs(y_hash - x_hash), d)
        return torch.sum(torch.maximum(loss, avoid_zero_div))
