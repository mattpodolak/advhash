from advhash import hashes
from advhash.hash.dhash import DHash


def test_serialize():
    config = {'split_point': 'resize', 'hash_size':8}
    hash_fn = DHash(**config)
    serialized = hashes.serialize(hash_fn)
    errors = []

    for key in config.keys():
        if(serialized['config'][key] != config[key]):
            errors.append(f'Expected {key} to be serialized as {config[key]}, instead got {serialized["config"][key]}')
    
    if serialized['class_name'] != 'DHash':
        errors.append(f'Expected class_name to be serialized as DHash, instead got {serialized["class_name"]}')

    assert not errors, "errors occured:\n{}".format("\n".join(errors))

def test_get_str():
    dhash_fn = hashes.get('dhash')

    assert type(dhash_fn) == DHash