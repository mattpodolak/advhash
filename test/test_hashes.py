from advhash import hashes
from advhash.hash.dhash import DHash
import torch

def test_serialize():
    config = {'split_point': 'resize', 'hash_size':8, 'device': torch.device('cpu')}
    hash_fn = DHash(**config)
    serialized = hashes.serialize(hash_fn)

    assert serialized['config'] == config

def test_deserialize():
    config = {'split_point': 'resize', 'hash_size':8, 'device': torch.device('cpu')}
    full_config = {'class_name': 'dhash', 'config': config}
    deserialized = hashes.deserialize(full_config)

    assert isinstance(deserialized, DHash) and deserialized.get_config() == config

def test_get_str():
    dhash_fn = hashes.get('dhash')

    assert isinstance(dhash_fn, DHash)

def test_get_str_config():
    config = {'split_point': 'resize', 'hash_size':8, 'device': torch.device('cpu')}
    dhash_fn = hashes.get('dhash', config)

    assert type(dhash_fn) == DHash and dhash_fn.get_config() == config

def test_get_hash():
    dhash_fn = hashes.get(DHash())

    assert isinstance(dhash_fn, DHash)

def test_get_config():
    config = {'split_point': 'resize', 'hash_size':8}
    hash_fn = DHash(**config)
    hash_config = hashes.serialize(hash_fn)
    dhash_fn = hashes.get(hash_config)
    assert isinstance(dhash_fn, DHash) and dhash_fn.get_config() == hash_config['config']