def merge_dicts(d1, d2):
    """
    Recursively merges dictionary d2 into dictionary d1.

    If both dictionaries have a nested dictionary at the same key, this function
    will recursively merge those nested dictionaries. If there is a conflict
    where d1 has a dictionary and d2 has a non-dictionary value at the same key,
    the value from d2 will overwrite the one in d1.

    Args:
        d1 (dict): The dictionary to be updated.
        d2 (dict): The dictionary with updates to merge into d1.

    Returns:
        None: The function updates d1 in place.

    Example:
        d1 = {'a': 1, 'b': {'x': 10}}
        d2 = {'b': {'y': 20}, 'c': 3}
        merge_dicts(d1, d2)
        # d1 is now {'a': 1, 'b': {'x': 10, 'y': 20}, 'c': 3}
    """
    for k, v in d2.items():
        if isinstance(v, dict) and k in d1 and isinstance(d1[k], dict):
            merge_dicts(d1[k], v)
        else:
            d1[k] = v
