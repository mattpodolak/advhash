"""
Built-in hashing classes
"""

from advhash.hash import dhash
from advhash.hash import base
from advhash.utils.generic import serialize_object
from advhash.utils.generic import deserialize_object


def serialize(hash_fn):
    """Serialize the Hash configuration to JSON compatible python dict.

    The configuration can be used for persistence and reconstruct the `Hash`
    instance again.

    Args:
        hash_fn: A `Hash` instance to serialize.

    Returns:
        Python dict which contains the configuration of the input hash_fn.
    """
    return serialize_object(hash_fn)


def deserialize(config):
    """Inverse of the `serialize` function.

    Args:
        config: Hash configuration dictionary.

    Returns:
        A Hash instance.
    """
    all_hashes = {
        'dhash': dhash.DHash,
    }

    # Make deserialization case-insensitive for built-in hashing functions.
    if config['class_name'].lower() in all_hashes:
        config['class_name'] = config['class_name'].lower()
    return deserialize_object(
        config,
        module_objects=all_hashes,
        module_type='Hash')


def get(hash_fn, config=None):
    """Retrieves a Hash instance.

    Args:
        hash_fn: one of the following
                - String: name of a hashing function
                - Dictionary: configuration dictionary
                - Hash instance: will be returned unchanged
        config (optional): Dictonary with config when using string for `hash_fn`

    Returns:
        A Hash instance.

    Raises:
        ValueError: If `hash_fn` cannot be interpreted.
    """
    if isinstance(hash_fn, base.Hash):
        return hash_fn
    elif isinstance(hash_fn, dict):
        return deserialize(hash_fn)
    elif isinstance(hash_fn, str):
        if config is None:
            config = {}
        config = {'class_name': str(hash_fn), 'config': config}
        return deserialize(config)
    else:
        raise ValueError(
            f'Could not interpret hash function : {hash_fn}')
