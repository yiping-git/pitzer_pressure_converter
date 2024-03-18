# -*- coding: utf-8 -*-
# Author: Yiping Liu
# Description:
# Version: 1.0
# Last Modified: Mar 15, 2024


import iapws
import numpy as np
import thermopy as tp
from utilities.universal_functions import lagrange, partial_lagrange, partial_lagrange_2ed


def dielectric(T, P):
    """
    Calculate the dielectric constant of water at given temperature.
    :param T:  temperature in Kelvin.
    :param P:  pressure, MPa.
    :return: dielectric constant.
    """

    P = P * 10  # MPa to bar

    U1 = 3.4279E2
    U2 = -5.0866E-3
    U3 = 9.4690E-7
    U4 = -2.0525
    U5 = 3.1159E3
    U6 = -1.8289E2
    U7 = -8.0325E3
    U8 = 4.2142E6
    U9 = 2.1417
    D1000 = U1 * np.exp(U2 * T + U3 * T ** 2)
    C = U4 + U5 / (U6 + T)
    B = U7 + U8 / T + U9 * T
    return D1000 + C * np.log((B + P) / (B + 1000))


def pdt_p(T, P):
    """
    Calculate  [∂D/∂T]P at constant P (pressure).
    :param T: temperature in Kelvin
    :param P: pressure, MPa.
    :return: [∂D/∂T]P
    """

    # MPa to bar
    P = P * 10

    U1 = 3.4279E2
    U2 = -5.0866E-3
    U3 = 9.4690E-7
    U4 = -2.0525
    U5 = 3.1159E3
    U6 = -1.8289E2
    U7 = -8.0325E3
    U8 = 4.2142E6
    U9 = 2.1417

    return U1 * (2 * T * U3 + U2) * np.exp(T ** 2 * U3 + T * U2) - U5 * np.log(
        (P + T * U9 + U7 + U8 / T) / (T * U9 + U7 + 1000 + U8 / T)) / (T + U6) ** 2 + (U4 + U5 / (T + U6)) * (
                   (-U9 + U8 / T ** 2) * (P + T * U9 + U7 + U8 / T) / (T * U9 + U7 + 1000 + U8 / T) ** 2 + (
                   U9 - U8 / T ** 2) / (T * U9 + U7 + 1000 + U8 / T)) * (T * U9 + U7 + 1000 + U8 / T) / (
                   P + T * U9 + U7 + U8 / T)


def pdp_t(T, P):
    """
    Calculate  [∂D/∂P]T at constant P (pressure).
    :param T: temperature (K).
    :param P: pressure (MPa).
    :return: [∂D/∂P]T
    """
    U4 = -2.0525
    U5 = 3.1159E3
    U6 = -1.8289E2
    U7 = -8.0325E3
    U8 = 4.2142E6
    U9 = 2.1417
    return 10 * (U4 + U5 / (T + U6)) / (10 * P + T * U9 + U7 + U8 / T)


def a_phi(T, P=None):
    """
    Calculate the Debye–Hückel coefficient for osmotic coefficient.


    :param T: temperature (K);
    :param P: pressure (MPa).
    :return: Ah, (kg·mol⁻¹)^(1/2)
    """
    if not P or np.isnan(P):
        P = iapws.iapws97._PSat_T(T=T)

    water = iapws.iapws97._Region1(T=T, P=P)

    d = dielectric(T=T, P=P)
    d_si = d * 4 * np.pi * tp.epsilon0

    rho = 1 / water['v']

    dhc = (1 / 3) * (2 * np.pi * tp.Na * rho) ** (1 / 2) * (tp.e ** 2 / (d_si * tp.k * T)) ** (3 / 2)

    return dhc


def partial_ln_dt(T, P):
    """
    Calculate the ∂(lnD)/∂T
    :param T: temperature in Kelvin
    :return: ∂(lnD)/∂T
    """
    d = dielectric(T=T, P=P)
    pd = pdt_p(T=T, P=P)
    return pd / d


def a_l(T, P=None):
    """
    Calculate the Debye-Hukel constant A(H).
    :param t: temperature in K.
    :return: Ah, kg^(1/2)·mol^(1/2)
    """
    if not P or np.isnan(P):
        P = iapws.iapws97._PSat_T(T)

    water = iapws.iapws97._Region1(T=T, P=P)
    p_lnd_t = partial_ln_dt(T=T, P=P)

    aphi = a_phi(T=T, P=P)
    alpha_w = water['alfav']

    dhc = -6 * aphi * tp.R * T * (1 + T * p_lnd_t + T / 3 * alpha_w)
    return dhc


def a_j(T, P=None):
    """
    Calculate the Debye-Hukel constant A(J).
    :param T: temperature (K).
    :return: A(J), in kg^(1/2)·mol^(1/2)
    """
    if not P:
        P = iapws.iapws97._PSat_T(T)

    step = 10

    Tm1 = T - step
    Tm2 = T - 2 * step
    Tp1 = T + step
    Tp2 = T + 2 * step

    al = a_l(T=T, P=P)
    al_m1 = a_l(T=Tm1, P=P)
    al_m2 = a_l(T=Tm2, P=P)
    al_p1 = a_l(T=Tp1, P=P)
    al_p2 = a_l(T=Tp2, P=P)

    return partial_lagrange(
        x=T,
        Lx=[Tm2, Tm1, T, Tp1, Tp2],
        Ly=[al_m2, al_m1, al, al_p1, al_p2]
    )


def a_v(T, P=None):
    """
    Calculate the Debye–Hückel coefficient for volume.
    :param T: temperature (K);
    :param P: pressure (MPa).
    :return: Av (cm³·kg^(1/2)·mol^(-3/2))
    """
    if not P:
        P = iapws.iapws97._PSat_T(T=T)

    water = iapws.iapws97._Region1(T=T, P=P)
    kappa = water['kt']

    pepp_t = pdp_t(T=T, P=P)
    aphi = a_phi(T=T, P=P)
    epsilon = dielectric(T=T, P=P)
    return 2 * aphi * tp.R * T * (
            3 * pepp_t / epsilon - kappa
    )



