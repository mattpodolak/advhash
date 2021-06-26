import advhash.utils.generic as generic
from advhash.hash.dhash import DHash


def test_serialize_object():
    config = {'split_point': 'resize', 'hash_size':8}
    hash_fn = DHash(**config)
    serialized = generic.serialize_object(hash_fn)
    errors = []

    for key in config.keys():
        if(serialized['config'][key] != config[key]):
            errors.append(f'Expected {key} to be serialized as {config[key]}, instead got {serialized["config"][key]}')
    
    if serialized['class_name'] != 'DHash':
        errors.append(f'Expected class_name to be serialized as DHash, instead got {serialized["class_name"]}')

    assert not errors, "errors occured:\n{}".format("\n".join(errors))

def test_deserialize_object():
    hash_config = {'split_point': 'resize', 'hash_size':8}
    config = {'class_name': 'dhash', 'config': hash_config}
    all_hashes = {
        'dhash': DHash,
    }
    deserialized = generic.deserialize_object(
      config,
      module_objects=all_hashes,
      module_type='Hash')

    errors = []

    for key in hash_config.keys():
        if(getattr(deserialized, key) != hash_config[key]):
            errors.append(f'Expected {key} to be serialized as {hash_config[key]}, instead got {getattr(deserialized, key)}')
    
    if type(deserialized) != DHash:
        errors.append(f'Expected class_name to be deserialized as DHash, instead got {type(deserialized)}')

    assert not errors, "errors occured:\n{}".format("\n".join(errors))

def test_deserialize_object_str():
    all_hashes = {
        'dhash': DHash,
    }
    deserialized = generic.deserialize_object(
      'dhash',
      module_objects=all_hashes,
      module_type='Hash')

    errors = []
    
    if type(deserialized) != DHash:
        errors.append(f'Expected class_name to be deserialized as DHash, instead got {type(deserialized)}')

    assert not errors, "errors occured:\n{}".format("\n".join(errors))