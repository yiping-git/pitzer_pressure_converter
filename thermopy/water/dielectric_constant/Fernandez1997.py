# -*- coding: utf-8 -*-
# Author: Yiping Liu
# Description:
# Version: 1.0
# Last Modified: Mar 15, 2024

import thermopy as tp
import iapws
import numpy as np
from utilities.universal_functions import lagrange, partial_lagrange, partial_lagrange_2ed


def density(T, P):
    water = iapws.iapws97._Region1(T=T, P=P)
    rho = (1 / water['v']) / (tp.Mw / 1000)
    return rho


Li = [1, 1, 1, 2, 3, 3, 4, 5, 6, 7, 10]
Lj = [0.25, 1, 2.5, 1.5, 1.5, 2.5, 2, 2, 5, 0.5, 10]
Ln = [0.978224486826, -0.957771379375,
      0.237511794148, 0.714692244396,
      -0.298217036956, -0.108863472196,
      0.949327488264E-1, -0.980469816509E-2,
      0.165167634970E-4, 0.937359795772E-4,
      -0.123179218720E-9, 0.196096504426E-2,
      ]


def g_func_ana(T, rho):
    """
    Analytical expression of the g function defined in Ref[1].
    :param T: temperature (K)
    :param rho: density (mol·m⁻³)
    :return: g value
    """
    epsilon = dielectric(T, rho)
    return (2 + 1 / epsilon) * tp.k * T / (3 * tp.mu ** 2) * (
            3 * tp.epsilon0 * (epsilon - 1) / (tp.Na * rho) - tp.alfa * (epsilon + 2)
    )


def g_func(T, rho):
    """
    Calculate value (numerically fitted correlation) of g function defined in Ref[1]. Verification: at T=573.11K,
    P=8.6MPa, g(exp)=2.538 957, g(cal)=2.544602415366809.
    :param T: temperature, K
    :param rho: density, mol·m⁻³
    :return: g value
    """

    g = 1 + Ln[11] * (rho / tp.rhoc * (tp.Mw / 1000)) * (T / 228 - 1) ** (-1.2)
    for i, j, n in zip(Li, Lj, Ln):
        g += n * (rho / tp.rhoc * (tp.Mw / 1000)) ** i * (tp.Tc / T) ** j
    return g


def A_func(T, rho):
    """
    Calculate A value, defined in Ref[1].
    :param T: temperature, K
    :param rho: density, mol·m⁻³
    :return: A value
    """
    g = g_func(T=T, rho=rho)
    return tp.Na * tp.mu ** 2 * rho * g / (tp.epsilon0 * tp.k * T)


def B_func(rho):
    """
    Calculate B value, defined in Ref[1].
    :param T: temperature, K
    :param rho: density, mol·m⁻³
    :return: B value
    """
    return tp.Na * tp.alfa * rho / (3 * tp.epsilon0)


def C_func(T, rho):
    """
    Calculate C value, defined in Ref[1].
    :param T: temperature, K
    :param rho: density, mol·m⁻³
    :return: C value
    """
    A = A_func(T=T, rho=rho)
    B = B_func(rho)
    return 9 + 2 * A + 18 * B + A ** 2 + 10 * A * B + 9 * B ** 2


def dielectric(T, rho):
    """
    Calculate dielectric constant of water.
    :param T: temperature (K).
    :param rho: density of water (ρ, mol·m⁻³)
    :return:
    """
    A = A_func(T=T, rho=rho)
    B = B_func(rho=rho)
    return (1 + A + 5 * B + np.sqrt(9 + 2 * A + 18 * B + A ** 2 + 10 * A * B + 9 * B ** 2)) / (4 - 4 * B)


def lagrange_ept(T, P):
    """
    Partial derivative of dielectric constant to temperature at constant pressure [∂D/∂T]P with lagrange-5-point method.
    :param T: temperature (K)
    :param P: pressure (Mpa)
    :return: first derivative of dielectric constant to pressure, [∂D/∂T]P.
    """

    step = 5  # k, distance between x variable, maybe affect by the limit of the iapws package.
    Tm1 = T - step
    Tm2 = T - step * 2
    Tp1 = T + step
    Tp2 = T + step * 2

    rho = density(T, P)
    rho_m1 = density(Tm1, P)
    rho_m2 = density(Tm2, P)
    rho_p1 = density(Tp1, P)
    rho_p2 = density(Tp2, P)

    epsilon = dielectric(T, rho)
    epsilon_m1 = dielectric(Tm1, rho_m1)
    epsilon_m2 = dielectric(Tm2, rho_m2)
    epsilon_p1 = dielectric(Tp1, rho_p1)
    epsilon_p2 = dielectric(Tp2, rho_p2)

    return partial_lagrange(
        x=T,
        Lx=[Tm2, Tm1, T, Tp1, Tp2],
        Ly=[epsilon_m2, epsilon_m1, epsilon, epsilon_p1, epsilon_p2]
    )


def lagrange_epp(T, P):
    """
    Partial derivative of dielectric constant to pressure at constant temperature [∂D/∂P]T with lagrange-5-point method.
    :param T: temperature (K)
    :param P: pressure (Mpa)
    :return: first derivative of dielectric constant to pressure, [∂D/∂P]T.
    """
    step = P / 10  # Mpa, distance between x variable, maybe affect by the limit of the iapws package.
    Pm1 = P - step
    Pm2 = P - step * 2
    Pp1 = P + step
    Pp2 = P + step * 2

    rho = density(T, P)
    rho_m1 = density(T, Pm1)
    rho_m2 = density(T, Pm2)
    rho_p1 = density(T, Pp1)
    rho_p2 = density(T, Pp2)

    epsilon = dielectric(T, rho)
    epsilon_m1 = dielectric(T, rho_m1)
    epsilon_m2 = dielectric(T, rho_m2)
    epsilon_p1 = dielectric(T, rho_p1)
    epsilon_p2 = dielectric(T, rho_p2)

    return partial_lagrange(
        x=P,
        Lx=[Pm2, Pm1, P, Pp1, Pp2],
        Ly=[epsilon_m2, epsilon_m1, epsilon, epsilon_p1, epsilon_p2]
    )


def lagrange_eptt(T, P):
    """
    The second partial derivative of dielectric constant to temperature at constant pressure [∂²D/∂T²]P with
    lagrange-5-point method.
    :param T: temperature (K)
    :param P: pressure (Mpa)
    :return: second derivative of dielectric constant to pressure, [∂²D/∂T²]P.
    """
    step = 5  # k, distance between x variable, maybe affect by the limit of the iapws package.
    Tm1 = T - step
    Tm2 = T - step * 2
    Tp1 = T + step
    Tp2 = T + step * 2

    rho = density(T, P)
    rho_m1 = density(Tm1, P)
    rho_m2 = density(Tm2, P)
    rho_p1 = density(Tp1, P)
    rho_p2 = density(Tp2, P)

    epsilon = dielectric(T, rho)
    epsilon_m1 = dielectric(Tm1, rho_m1)
    epsilon_m2 = dielectric(Tm2, rho_m2)
    epsilon_p1 = dielectric(Tp1, rho_p1)
    epsilon_p2 = dielectric(Tp2, rho_p2)

    return partial_lagrange_2ed(
        x=T,
        Lx=[Tm2, Tm1, T, Tp1, Tp2],
        Ly=[epsilon_m2, epsilon_m1, epsilon, epsilon_p1, epsilon_p2]
    )


def lagrange_eppp(T, P):
    """
    The second partial derivative of dielectric constant to pressure at constant temperature [∂²D/∂P²]T with
    lagrange-5-point method.
    :param T: temperature (K)
    :param P: pressure (Mpa)
    :return: second derivative of dielectric constant to pressure, [∂²D/∂P²]T.
    """
    step = P / 10  # Mpa, distance between x variable, maybe affect by the limit of the iapws package.
    Pm1 = P - step
    Pm2 = P - step * 2
    Pp1 = P + step
    Pp2 = P + step * 2

    rho = density(T, P)
    rho_m1 = density(T, Pm1)
    rho_m2 = density(T, Pm2)
    rho_p1 = density(T, Pp1)
    rho_p2 = density(T, Pp2)

    epsilon = dielectric(T, rho)
    epsilon_m1 = dielectric(T, rho_m1)
    epsilon_m2 = dielectric(T, rho_m2)
    epsilon_p1 = dielectric(T, rho_p1)
    epsilon_p2 = dielectric(T, rho_p2)

    return partial_lagrange_2ed(
        x=P,
        Lx=[Pm2, Pm1, P, Pp1, Pp2],
        Ly=[epsilon_m2, epsilon_m1, epsilon, epsilon_p1, epsilon_p2]
    )


def a_gamma(T, P=None):
    """
    Calculate the Debye–Hückel coefficient for activity coefficient.


    :param T: temperature (K);
    :param P: pressure (MPa).
    :return: Ah, (kg·mol⁻¹)^(1/2)
    """
    if not P:
        P = iapws.iapws97._PSat_T(T)

    water = iapws.iapws97._Region1(T=T, P=P)

    # convert to mol·m⁻³
    rho = (1 / water['v']) / (tp.Mw / 1000)

    epsilon = dielectric(T=T, rho=rho)

    return (2 * np.pi * tp.Na * rho * tp.Mw / 1000) ** (1 / 2) * (
            tp.e ** 2 / (4 * np.pi * epsilon * tp.epsilon0 * tp.k * T)
    ) ** (3 / 2)


def a_phi(T, P=None):
    """
    Calculate the Debye–Hückel coefficient for osmotic coefficient.


    :param T: temperature (K);
    :param P: pressure (MPa).
    :return: Ah, (kg·mol⁻¹)^(1/2)
    """
    return (1 / 3) * a_gamma(T=T, P=P)


def a_h(T, P=None):
    """
    Calculate the Debye–Hückel coefficient for enthalpy.


    :param T: temperature (K);
    :param P: pressure (MPa).
    :return: Ah, (kg·mol⁻¹)^(1/2)
    """
    if not P:
        P = iapws.iapws97._PSat_T(T)

    water = iapws.iapws97._Region1(T=T, P=P)
    alfav = water['alfav']
    rho = (1 / water['v']) / (tp.Mw / 1000)
    aphi = a_phi(T=T, P=P)
    pepp_t = lagrange_ept(T=T, P=P)
    epsilon = dielectric(T=T, rho=rho)

    return -6 * aphi * tp.R * T * (
            1 + T * pepp_t / epsilon + T * alfav / 3
    )


def a_j(T, P=None):
    """
    Calculate the Debye–Hückel coefficient for heat capacity.


    :param T: temperature (K);
    :param P: pressure (MPa).
    :return: Aj, (kg·mol⁻¹)^(1/2)
    """
    if not P:
        P = iapws.iapws97._PSat_T(T)
    step = 10

    Tm1 = T - step
    Tm2 = T - 2 * step
    Tp1 = T + step
    Tp2 = T + 2 * step

    al = a_h(T=T, P=P)
    al_m1 = a_h(T=Tm1, P=P)
    al_m2 = a_h(T=Tm2, P=P)
    al_p1 = a_h(T=Tp1, P=P)
    al_p2 = a_h(T=Tp2, P=P)

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
    water = iapws.iapws97._Region1(T=T, P=P)
    rho = (1 / water['v']) / (tp.Mw / 1000)
    kappa = water['kt']

    pepp_t = lagrange_epp(T=T, P=P)
    aphi = a_phi(T=T, P=P)
    epsilon = dielectric(T=T, rho=rho)
    return 2 * aphi * tp.R * T * (
            3 * pepp_t / epsilon - kappa
    )

T = 300
P = 10
print('a_v:', a_v(T=T,P=P))

"""
REFERENCES

[1] Fernandez, D. P., Goodwin, A. R. H., Lemmon, E. W., Levelt Sengers, J. M. H., & Williams, R. C. (1997). A formulation for the static permittivity of water and steam at temperatures from 238 K to 873 K at pressures up to 1200 MPa, including derivatives and Debye–Hückel coefficients. Journal of Physical and Chemical Reference Data, 26(4), 1125-1166.
"""
