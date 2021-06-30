from collections import OrderedDict
import torch
from advhash.hash.base import Hash
from advhash.utils.convert import rgb2luma
from advhash.utils.resize import lanczos_resize

class DHash(Hash):
    """Hash function that implements Difference Hash.

    Difference Hash (dHash), uses the horizontal image gradient to binarize the image values
    after resizing and converting to the luma channel. Implementation is based on
    http://www.hackerfactor.com/blog/?/archives/529-Kind-of-Like-That.html

    Args:
        split_point: String. Defines the interior function to use as a split point when
        calling the hash function.
        hash_size: Integer between 1 and 16, inclusive. Defines the size of the resulting hash.

    Raises:
        ValueError: For invalid arguments

    """
    def __init__(self, split_point=None, hash_size=16, device=None):
        super().__init__(split_point, hash_size, device)

    def _horiz_grad(self, X):
        return X[:, 1:] - X[:,:-1]

    def _resize(self, X):
        return lanczos_resize(X, self.hash_size+1, self.hash_size)

    @property
    def interior_functions(self):
        return OrderedDict({"luma": rgb2luma, "resize": self._resize,
            "horiz_grad": self._horiz_grad})

    def get_config(self):
        """Returns the config of the Hash instance.

        The config is a Python dictionary (serializable)
        containing the configuration, and can be used to
        reinstantiate the hash function.

        Returns:
            Python dictionary.
        """
        config = super(DHash, self).get_config()
        return config

    def _hash(self, X, split_point=None):
        """Creates a hash of X.

        This method iterates through the interior functions
        of the hashing function, returning a partial or full
        hash.

        Args:
            X: A 2D PyTorch tensor

        Returns:
            A float32 PyTorch tensor containing the hash values
        """
        for fn_name, fn in self.interior_functions.items():
            X = fn(X)
            if split_point == fn_name:
                return X
        if split_point == "last":
            return X
        return (X > 0).type(torch.float32)
