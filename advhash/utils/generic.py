def deserialize_object(identifier, module_objects=None, module_type=None):
    """Turns the serialized form of a AdvHash object back into an actual object.

    This function is for mid-level library implementers rather than end users.
    Importantly, this utility requires you to provide the dict of `module_objects`
    to use for looking up the object config; this is not populated by default.

    Args:
    identifier: the serialized form of the object.
    module_objects: A dictionary of built-in objects to look the name up in.
        Generally, `module_objects` is provided by midlevel library implementers.
    module_type: A string of the class name for the module_objects, used for
        logging purposes.

    Returns:
    The deserialized object.
    """
    if identifier is None:
        return None

    if isinstance(identifier, dict):
        config = identifier
        class_name = config['class_name']
        cls_config = config['config']
        cls = module_objects.get(class_name, None)

        if cls is None:
            raise ValueError(
                'Unknown {}: {}. Please ensure this object exists.'
                .format(module_type, class_name))
        elif hasattr(cls, 'from_config'):
            deserialized_obj = cls.from_config(cls_config)
        else:
        # `cls` may be a function returning a class, 
        # `config` is passed that contains the kwargs.
            deserialized_obj = cls(**cls_config)

        return deserialized_obj

    elif isinstance(identifier, str):
        obj = module_objects.get(identifier)
        if obj is None:
            raise ValueError(
                'Unknown {}: {}. Please ensure this object exists.'
                .format(module_type, identifier))
        return obj()
    else:
        raise ValueError('Could not interpret serialized %s: %s' %
                        (module_type, identifier))

def serialize_object(instance):
    """Serialize an AdvHash object into a JSON-compatible representation.

    Args:
        instance: The object to serialize.

    Returns:
        A dict-like, JSON-compatible representation of the object's config.
    """

    if hasattr(instance, 'get_config'):
        base_config = {'class_name': instance.__class__.__name__, 'config': {}}
        try:
            config = instance.get_config()
        except NotImplementedError as e:
            raise e
        for key, item in config.items():
            if isinstance(item, str):
                base_config['config'][key] = item
            else:
                try:
                    base_config['config'][key] = serialize_object(item)
                except ValueError:
                    base_config['config'][key] = item
        return base_config
    elif hasattr(instance, '__name__'):
        return instance.__name__
    else:
        raise ValueError('Cannot serialize', instance)
