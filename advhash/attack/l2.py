import torch
from advhash.attack.base import Attack

class L2Attack(Attack):
    """Implementation of an L2 norm based adversarial collision attack.

    Args:
        hash_fn: String or Hash function. Defines what hashing function to
        perform an attack against.
        hash_size (optional): Integer. Size of hash created by the hashing function
        - default value is 16.
        split_point (optional): String. Defines what interior function to target as
        part of the attack - default value is None.
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
