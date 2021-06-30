import abc
from collections import OrderedDict
import torch

class Hash:
    """Base class for hashing algorithms

    Args:
        split_point: String. Defines the interior function to use as a split point when
        calling the hash function.
        hash_size: Integer between 1 and 16, inclusive. Defines the size of the resulting hash.
        device (optional): String or torch.device instance. Tensors will be stored on this device.

    Raises:
        ValueError: For invalid arguments
    """

    def __init__(self, split_point, hash_size, device=None):
        self.hash_size=hash_size
        self.split_point=split_point
        if device is None:
            self.device = torch.device('cpu')
        else:
            if isinstance(device, torch.device):
                self.device = device
            elif isinstance(device, str):
                self.device = torch.device(device)
            else:
                raise ValueError('device value is invalid, expected a string or \
                torch.device instance.')

    @abc.abstractproperty
    def interior_functions(self):
        return OrderedDict()

    @property
    def hash_size(self):
        return self._hash_size

    @property
    def split_point(self):
        return self._split_point

    @hash_size.setter
    def hash_size(self, new_size):
        if isinstance(new_size, int) and new_size > 0 and new_size <=16:
            self._hash_size = new_size
        else:
            raise ValueError(f'Hash size should be an integer between 1 and 16, \
                instead got {type(new_size)}')

    @split_point.setter
    def split_point(self, new_split=None):
        if isinstance(new_split, str):
            if new_split in self.interior_functions.keys():
                self._split_point = new_split
            elif new_split == "last":
                self._split_point = new_split
            else:
                raise ValueError(f'Invalid split point value {new_split}')
        elif new_split is None:
            self._split_point = None
        else:
            raise ValueError(f'Expected a string/None, instead got {type(new_split)}')

    @abc.abstractmethod
    def get_config(self):
        """Returns the config of the Hash instance.

        The config is a Python dictionary (serializable)
        containing the configuration, and can be used to
        reinstantiate the hash function.

        Returns:
            Python dictionary.
        """

        return {"hash_size": self.hash_size, "split_point": self.split_point, "device": self.device}

    @classmethod
    def from_config(cls, config):
        """Creates a Hash instance from its config.

        This method is the reverse of `get_config`,
        capable of instantiating the same hash function
        from the config dictionary.

        Args:
            config: A Python dictionary, typically the output of get_config.

        Returns:
            A Hash instance.
        """
        return cls(**config)

    def partial_hash(self, X):
        """Creates a partial hash of X.

        This method uses the split_point to create a partial hash. If no
        split point is set, it will return a full hash.

        Args:
            X: A 2D PyTorch tensor

        Returns:
            A float32 PyTorch tensor containing a partial/full hash
        """
        return self._hash(X, self.split_point)

    def full_hash(self, X):
        """Creates a full hash of X.

        This method returns the hash of the provided tensor.

        Args:
            X: A 2D PyTorch tensor

        Returns:
            A float32 PyTorch tensor containing the full hash
        """
        return self._hash(X, None)

    @abc.abstractmethod
    def _hash(self, X, split_point):
        return X
