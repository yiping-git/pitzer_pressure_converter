# -*- coding: utf-8 -*-
# Author: Yiping Liu
# Description:
# Version: 1.0
# Last Modified: Mar 15, 2024
from .dielectric_constant import Bradley_and_Pltzer1978, Fernandez1997


def dielectric(T, P, method):
    if method == 'B&P':
        return Bradley_and_Pltzer1978.dielectric(T=T, P=P)
    elif method == 'Fernandez':
        return Fernandez1997.dielectric_p(T=T, P=P)
    else:
        print('Currently no such method.')


def a_phi(T, P, method):
    """
    Calculate the Debye–Hückel coefficient for osmotic coefficient.
    :param T: temperature (K);
    :param P: pressure (MPa).
    :return: Aphi, (kg·mol⁻¹)^(1/2)
    """
    if method == 'B&P':
        return Bradley_and_Pltzer1978.a_phi(T=T, P=P)
    elif method == 'Fernandez':
        return Fernandez1997.a_phi(T=T, P=P)
    else:
        print('Currently no such method.')


def a_h(T, P, method):
    """
    Calculate the Debye–Hückel coefficient for enthalpy.
    :param T: temperature (K);
    :param P: pressure (MPa).
    :return: Ah, (kg·mol⁻¹)^(1/2)
    """
    if method == 'B&P':
        return Bradley_and_Pltzer1978.a_h(T=T, P=P)
    elif method == 'Fernandez':
        return Fernandez1997.a_h(T=T, P=P)
    else:
        print('Currently no such method.')


def a_j(T, P, method):
    """
    Calculate the Debye–Hückel coefficient for heat capacity.
    :param T: temperature (K);
    :param P: pressure (MPa).
    :return: Aj, (kg·mol⁻¹)^(1/2)
    """
    if method == 'B&P':
        return Bradley_and_Pltzer1978.a_j(T=T, P=P)
    elif method == 'Fernandez':
        return Fernandez1997.a_j(T=T, P=P)
    else:
        print('Currently no such method.')


def a_v(T, P, method):
    """
    Calculate the Debye–Hückel coefficient for volume.
    :param T: temperature (K);
    :param P: pressure (MPa).
    :return: Av (cm³·kg^(1/2)·mol^(-3/2))
    """
    if method == 'B&P':
        return Bradley_and_Pltzer1978.a_v(T=T, P=P)
    elif method == 'Fernandez':
        return Fernandez1997.a_v(T=T, P=P)
    else:
        print('Currently no such method.')
