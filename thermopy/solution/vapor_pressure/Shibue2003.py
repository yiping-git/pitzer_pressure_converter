# -*- coding: utf-8 -*-
# Author: Yiping Liu
# Description:
# Version: 1.0
# Last Modified: Mar 14, 2024

import numpy as np

pc_db = {
    'NaCl': {
        'q1': 9.00404E2,
        'q2': -2.92542E4,
        'q3': 1.39806E6,
        'q4': -2.80756E7,
        'q5': 2.41637E8,
        'q6': -7.18726E8,
    },
    'CaCl2': {
        'q1': -6.47588E-3,
        'q2': 8.58985E0,
        'q3': 5.97149E3,
        'q4': -2.22618E7,
        'q5': 0,
        'q6': 0,
    }
}

h_db = {
    'NaCl': {
        'a1': 1.28746E-1,
        'a2': -7.31097E-1,
        'a3': -3.15058E2,
        'b1': 3.92767E2,
        'b2': -2.46440E3,
        'u': 0.024
    },
    'CaCl2': {
        'a1': 5.89403E-2,
        'a2': -3.40230E-1,
        'a3': -4.47233E2,
        'b1': -1.24733E3,
        'b2': 2.43253E3,
        'u': 0.012,
    }
}


def h(salt, x):
    """
    Calculating h value defined in Ref[1].
    :param salt: 'NaCl' or 'CaCl2'
    :param x: mole fraction.
    :return: h_value
    """
    p = h_db[salt]
    return (p['a1'] ** 2 * p['a2'] / (p['u'] + p['a1'] ** 2) ** 2 + 2 * p['a3'] * p['u']) * (x - p['u']) + p['b1'] * (
            x - p['u']) ** 2 + p['b2'] * (x - p['u']) * (x ** 2 - p['u'] ** 2) + p['a2'] * p['u'] / (
                   p['u'] + p['a1'] ** 2) + p['a3'] * p['u'] ** 2


def pc(salt, x):
    """
    Function for calculating critical pressure, defined in Ref[1].
    :param salt: 'NaCl' or 'CaCl2'
    :param x: mole fraction.
    :return: critical pressure, MPa
    """
    b = pc_db[salt]
    return 22.064 \
           + b['q1'] * x \
           + b['q2'] * x ** 2 \
           + b['q3'] * x ** 3 \
           + b['q4'] * x ** 4 \
           + b['q5'] * x ** 5 \
           + b['q6'] * x ** 6


def g(T):
    """
    The g function defined in Ref[1], which is a variation of function for vapor pressure of pure water by Ref[2].
    :param T: temperature,K
    :return: g value
    """
    result = 647.096 / T * (
            -7.85951783 * (1 - T / 647.096) + 1.84408259 * (1 - T / 647.096) ** (3 / 2)
    ) + 647.096 / T * (
                     -11.7866497 * (1 - T / 647.096) ** 3 + 22.6807411 * (1 - T / 647.096) ** (7 / 2)
             ) + 647.096 / T * (
                     -15.9618719 * (1 - T / 647.096) ** 4 + 1.80122502 * (1 - T / 647.096) ** (15 / 2)
             )
    return result


def molal_to_mole_fraction(m):
    """
    convert molality to mole fraction
    :param m:molality
    :return:mole fractions
    """

    m_w = 1000 / 18.015257
    frac = m / (m + m_w)
    return frac


def pressure_fraction(salt, T, x):
    """
    mole fraction based pressure calculation.
    :param salt: NaCl or CaCl2
    :param T: temperature, K
    :param x: mole fraction of salt-H2O system.
    :return: vapor pressure in MPa
    """
    return np.exp(np.log(pc(salt, x)) + g(T) + h(salt, x))


def pressure_molality(salt, T, m):
    """
    mole fraction based pressure calculation.
    :param salt: NaCl or CaCl2
    :param T: temperature, K
    :param x: mole fraction of salt-H2O system.
    :return: vapor pressure in MPa
    """
    x = molal_to_mole_fraction(m)
    return pressure_fraction(salt, T, x)


"""
REFERENCES

[1] Shibue, Y. (2003). Vapor pressures of aqueous NaCl and CaCl2 solutions at elevated temperatures. Fluid phase equilibria, 213(1-2), 39-51.
[2] Wagner, W., & Pruss, A. (1993). International equations for the saturation properties of ordinary water substance. Revised according to the international temperature scale of 1990. Addendum to J. Phys. Chem. Ref. Data 16, 893 (1987). Journal of physical and chemical reference data, 22(3), 783-787.
"""

