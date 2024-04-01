import numpy as np
from numpy.polynomial import Polynomial


def mg12(t):
    """
    R² = 0.9881
    :param t: temperature (°C).
    :return: solubility (mol·kg⁻¹).
    """
    coefficients = np.array([
        -2.93081407e-06,
        -2.80313203e-04,
        -8.67515964e-03,
        -5.75509650e-02,
        1.64949201e+00,
        2.58821154e+01
    ])
    return np.polyval(coefficients, t)


def mg8(t):
    """
    R² = 8.004E-01
    :param t:temperature (°C).
    :return: solubility (mol·kg⁻¹).
    """
    return -3.902E-04 * t ** 2 + 3.525E-02 * t + 5.563E+00


def mg6(t):
    """
    R² = 9.9368E-0
    :param t:temperature (°C).
    :return: solubility (mol·kg⁻¹).
    """
    coefficients = np.array([
        1.85134659e-11,
        -5.49012940e-09,
        6.13045040e-07,
        -3.05754371e-05,
        7.14408810e-04,
        5.63462708e-03,
        5.54506413e+00,
    ])

    return np.polyval(coefficients, t)


def mg4(t):
    """
    R² = 9.889E-01
    :param t:temperature (°C).
    :return:solubility (mol·kg⁻¹).
    """
    coefficients = np.array([
        1.28218499e-09,
        - 1.09800170e-06,
        3.90076470e-04,
        - 7.35747236e-02,
        7.76983605e+00,
        - 4.35526232e+02,
        1.01307536e+04
    ])
    return np.polyval(coefficients, t)


def mg2(t):
    """
    R² = 9.871E-01
    :param t:temperature (°C).
    :return:solubility (mol·kg⁻¹).
    """
    return 3.721E-04 * t ** 2 - 1.043E-01 * t + 1.988E+01


def MgCl2_solubility(t, solid=None):
    """
    Calculate the solubility of MgCl₂.
    :param solid: hydrate of MgCl₂.
    :param t: temperature (°C).
    :return: solubility (mol·kg⁻¹).
    """
    if solid:
        if solid == 'MgCl2-12H2O':
            return mg12(t)
        elif solid == 'MgCl2-8H2O':
            return mg8(t)
        elif solid == 'MgCl2-6H2O':
            return mg6(t)
        elif solid == 'MgCl2-4H2O':
            return mg4(t)
        elif solid == 'MgCl2-2H2O':
            return mg2(t)
    else:
        if -31.6 <= t <= -16.35:
            return mg12(t)
        elif -16.35 < t <= -3:
            return mg8(t)
        elif -3 < t <= 116.67:
            return mg6(t)
        elif 116.67 < t <= 181:
            return mg4(t)
        elif 181 < t:
            return mg2(t)


