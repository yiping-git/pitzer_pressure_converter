# -*- coding: utf-8 -*-
# Author: Yiping Liu
# Description:
# Version: 1.0
# Last Modified: Mar 14, 2024

import itertools
import functools
import joblib


def count_calls(func):
    """
    A wrapper used for counting the how many times a function is called.
    :param func: the target function.
    :return:
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        wrapper.calls += 1  # Increment the call count
        print(wrapper.calls)
        return func(*args, **kwargs)

    wrapper.calls = 0  # Initialize the call count to 0
    return wrapper


def cache_dict_array(func):
    """
     Cache dict result of a function, where, the value of a corresponding key is an array, e.g., {'a':np.array([...]),
     'b':...}.
     :param func: target function.
     :return:
     """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        cache_key = (func.__name__, args, tuple(sorted(kwargs.items())))
        cached_result = joblib.Memory('cache_dir').cache(func)
        return cached_result(*args, **kwargs)

    return wrapper


def cache_dict(func):
    """
    Cache dict result of a function.
    :param func: the target function
    :return:
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        cache_key = (func.__name__, args, tuple(sorted(kwargs.items())))
        memory = joblib.Memory(location='cache_dir', verbose=0)
        cached_func = memory.cache(func)
        return cached_func(*args, **kwargs)

    return wrapper


def cache_array(func):
    """
    Cache array result.
    :param func: target function
    :return:
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Generating a cache key using function name and all arguments
        cache_key = (func.__name__,) + args + tuple(sorted(kwargs.items()))
        # Creating a joblib.Memory cache with a specific cache directory
        cache = joblib.Memory('cache_dir').cache(func)
        # Returning the result of the cached function
        return cache(*args, **kwargs)

    return wrapper


def find_pair(x, pairs):
    """
    Compare and determine if x tuple is in the list of the "pairs" tuples.
    :param x: a tuple
    :param pairs: a list of tuples
    :return: a dict contains a boolean value and the match (if there is).
    """
    pair_length = len(pairs)
    i = 0
    has = False
    target = ''
    while i < pair_length:
        pair = pairs[i]
        if set(x) == set(pair):
            has = True
            target = pair
        i += 1
    return {
        "has": has,
        "target": target
    }


def get_charge_number(ion):
    """
    Get the charge number of an ion (str).
    :param ion: ion name, a string with "+" or "-" sign followed by number of charge.
    :return: charge number.
    """

    if "+" in ion:
        lis = ion.split("+")
        if lis[1]:
            result = lis[1]
        else:
            result = 1
    elif "-" in ion:
        lis = ion.split("-")
        if lis[1]:
            lis[1] = '-' + lis[1]
            result = lis[1]
        else:
            result = -1
    else:
        result = 0
    return int(result)


def ion_to_element(ion):
    element = ion
    if "+" in ion:
        lis = ion.split("+")
        element = lis[0]
    elif "-" in ion:
        lis = ion.split("-")
        element = lis[0]
    return element


def is_neutral(ion):
    charge = get_charge_number(ion)
    if charge == 0:
        return True
    return False


def is_cation(ion):
    charge = get_charge_number(ion)
    if charge > 0:
        return True
    return False


def is_anion(ion):
    charge = get_charge_number(ion)
    if charge < 0:
        return True
    return False


def type_of_species(ion):
    """
    determine the type of a ion (string).
    :param ion:
    :return: 0-neutral, 1-cation, -1-anion
    """
    type_value = 0

    charge = get_charge_number(ion)
    if charge == 0:
        type_value = 0
    if charge > 0:
        type_value = 1
    if charge < 0:
        type_value = -1
    return type_value


def is_2v2_salt(pair):
    """
    Determine whether a salt is a 2-2 type or not.
    :param ion1: ion1.
    :param ion2: ion2.
    :return: return True if it is a 2-2 type of salt.
    """

    ion1 = pair[0]
    ion2 = pair[1]
    if ion1.find("+") != -1:
        result1 = ion1.split("+")
    else:
        result1 = ion1.split("-")
    if ion2.find("+") != -1:
        result2 = ion2.split("+")
    else:
        result2 = ion2.split("-")
    if result1[1] == '2' and result2[1] == '2':
        return True
    else:
        return False


def calculate_ionic_strength(molalities):
    """
    Calculate ionic strength by inputting molalities.
    :param molalities: dict, e.g., {'Na+':1}.
    :return: ionic strength.
    """

    data = molalities
    ions = data.keys()
    sum_value = 0
    for ion in ions:
        charge_number = get_charge_number(ion)
        sum_value += data[ion] * (charge_number ** 2)
    return sum_value / 2


def group_components(components):
    """
    Find groups from components of ions and neutral species
    :param components: str, compsotions of a solution, can be cations (e.g., 'Na+'), anions (e.g., 'Cl-') and neutral species
    (e.g., 'NaCl0').
    :return: a dictionary of groups containing only cations, only anions or unique pairs of all the particles.
    """
    cations = [c for c in components if '+' in c]
    anions = [a for a in components if '-' in a]
    neutrals = [n for n in components if '+' not in n and '-' not in n]

    cation_anion_pairs = list(itertools.product(cations, anions))
    cation_pairs = list(itertools.combinations(cations, 2)) if len(cations) >= 2 else []
    anion_pairs = list(itertools.combinations(anions, 2)) if len(anions) >= 2 else []
    neutral_pairs = list(itertools.combinations(neutrals, 2))

    neutral_ion_pairs = list(itertools.product(neutrals, cations + anions))

    neutral_cation_anion_pairs = [(a, *b) for a in neutrals for b in cation_anion_pairs]

    return {
        'cations': cations,
        'anions': anions,
        'neutrals': neutrals,
        'cation_anion_pairs': cation_anion_pairs,
        'cation_pairs': cation_pairs,
        'anion_pairs': anion_pairs,
        'neutral_pairs': neutral_pairs,
        'neutral_ion_pairs': neutral_ion_pairs,
        'neutral_cation_anion_pairs': neutral_cation_anion_pairs
    }
