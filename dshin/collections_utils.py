import collections


def namedtuple_with_defaults(typename, field_names, default_values=()):
    """
    http://stackoverflow.com/a/18348004

    :param typename: Same as the `typename` in `collections.namedtuple`.
    :param field_names: Same as the `field_names` in `collections.namedtuple`.
    :param default_values: A list or dictionary of default values.
    :return:
    """
    T = collections.namedtuple(typename, field_names)
    T.__new__.__defaults__ = (None,) * len(T._fields)
    if isinstance(default_values, collections.Mapping):
        prototype = T(**default_values)
    else:
        prototype = T(*default_values)
    T.__new__.__defaults__ = tuple(prototype)
    return T
