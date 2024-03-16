# -*- coding: utf-8 -*-
# Author: Yiping Liu
# Description:
# Version: 1.0
# Last Modified: Mar 14, 2024

import numpy as np

pc_db = {
    'KCl': {
        'q1': 2.92000E3,
        'q2': 2.58266E4,
        'q3': 7.24841E2,
        'q4': 1.05057E4,
    },
    'MgCl2': {
        'q1': 3.16499E3,
        'q2': 1.58355E4,
        'q3': 9.59996E2,
        'q4': 1.01998E4,
    }
}
h_db = {
    'KCl': {
        'a1': -3.52372E1,
        'a2': 1.24680E2,
        'a3': 0,
    },
    'MgCl2': {
        'a1': -4.44219E1,
        'a2': 2.86908E2,
        'a3': -1.33485E3,
    }
}


def t_c(salt, x):
    q1 = pc_db[salt]['q1']
    q2 = pc_db[salt]['q2']
    return 647.096 + q1 * x + q2 * x ** 2


def p_c(salt, x):
    q3 = pc_db[salt]['q3']
    q4 = pc_db[salt]['q4']
    return 22.064 + q3 * x + q4 * x ** 2


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


def h(salt, x):
    a1 = h_db[salt]['a1']
    a2 = h_db[salt]['a2']
    a3 = h_db[salt]['a3']

    return a1 * x + a2 * x ** 2 + a3 * x ** 3


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
    Mole fraction based pressure calculation
    :param salt:
    :param T:
    :param x:
    :return:
    """
    return np.exp(np.log(p_c(salt, x)) + g(T) + h(salt, x))


def pressure_molality(salt, T, m):
    """
    Molality based pressure calculation
    :param salt:
    :param T:
    :param x:
    :return:
    """
    x = molal_to_mole_fraction(float(m))
    return pressure_fraction(salt=salt, T=T, x=x)


