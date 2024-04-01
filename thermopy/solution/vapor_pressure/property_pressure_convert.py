# -*- coding: utf-8 -*-
# Author: Yiping Liu
# Description:
# Version: 1.0
# Last Modified: Mar 14, 2024

import warnings
from scipy.integrate import quad
import numpy as np
import thermopy as tp
import iapws
from scipy.optimize import minimize


def vp_pure_water(t):
    """
    Calculating vapor pressure of pure water at different temperatures.
    :param t: temperaure in °C.
    :return: vapor pressure of pure water, in int.atm.
    """
    print(t)
    if not (50 <= t <= 374.11):
        warnings.warn("The temperature {} °C is outside the recommended range (50, 374.11).".format(t), Warning)

    t = 273.16 + t  # to Kelvin
    x = 647.27 - t  # 647.27 is critical temperature
    p_c = 218.167  # in int.atm.
    a = 3.3463130
    b = 4.14113E-2
    c = 7.515484E-9
    d = 1.3794481E-2
    e = 6.56444E-11

    mid_item = (x / t) * ((a + b * x + c * x ** 3 + e * x ** 4) / (1 + d * x))

    p = p_c / (
            10 ** (mid_item)
    )
    return p


def molar_vol_water(t):
    """
    Calculate the molar volume of pure water.
    :param t: temperature, in °C.
    :return: molar volumne of pure water, in cm^3.
    """

    v_c = 3.1975
    t_c = 374.11  # °C (critical temperature)
    x = t_c - t
    a = -0.3151548
    b = -1.203374E-3
    c = 7.48908E-13
    d = 0.1342489
    e = -3.946263E-3
    v_0 = tp.Mw * ((v_c + a * x ** (1 / 3) + b * x + c * x ** 4) / (1 + d * x ** (1 / 3) + e * x))
    return v_0


def xi(t, p):
    T = t + 273.16
    # p = vp_pure_water(t)
    tau = 1 / T
    b0 = 1.89 - 2641.62 * tau * 10 ** (80870 * tau ** 2)
    g_1 = 82.546 * tau - 1.6246E5 * tau ** 2
    g_2 = 0.21828 - 1.2697E5 * tau ** 2
    g_3 = 3.635E-4 - 6.768E64 * tau ** (24)
    result = b0 + b0 ** 2 * g_1 * tau * p + b0 ** 4 * g_2 * tau ** 3 * p ** 3 - b0 ** 13 * g_3 * tau ** 12 * p ** 12
    return 18.015268 * result / 10 ** 6


# def activity_of_water(t, p, m, nu):
#     """
#     Calculating water activity of a solution with measured vapor pressure of p at temperature t.
#     :param t: temperature in °C
#     :param p: vapor pressure of the solution, in int.atm.
#     :param m: molality of the electrolyte
#     :param nu: stoichiometric number of the electrolyte, e.g., for nu(NaCl) = 2, nu(MgCl2) = 3
#     :return:
#     """
#     tk = 273.16 + t
#     tau = 1 / tk
#
#     p0 = vp_pure_water(t)  # in int.atm
#     # p_m = iapws.iapws97._PSat_T(t + 273.16)  # in mPa
#     # p0 = p_m * 9.86923  # in int.atm
#
#     delta_p = (p0 - p) * 101325  # atm to Pa
#     v_w = molar_vol_water(t) / (10 ** 6)  # cm^3 to m^3
#
#     def integ(tau):
#         p1 = p0
#         p2 = p
#
#         b0 = 1.89 - 2641.62 * tau * 10 ** (80870 * tau ** 2)
#         g_1 = 82.546 * tau - 1.6246E5 * tau ** 2
#         g_2 = 0.21828 - 1.2697E5 * tau ** 2
#         g_3 = 3.635E-4 - 6.768E64 * tau ** (24)
#
#         item0 = b0 * (p2 - p1)
#         item1 = b0 ** 2 * g_1 * tau * (1 / 2) * (p2 ** 2 - p1 ** 2)
#         item2 = b0 ** 4 * g_2 * tau ** 3 * (1 / 4) * (p2 ** 4 - p1 ** 4)
#         item3 = b0 ** 13 * g_3 * tau ** 12 * (1 / 13) * (p2 ** 13 - p1 ** 13)
#
#         result = tp.Mw * (item0 + item1 + item2 - item3) * 101325 / (10 ** 6)
#         return result
#
#     inte2 = integ(tau)
#
#     term1 = - 1000 / (tp.Mw * nu * m) * np.log(p / p0)
#     term2 = - 1000 / (tp.Mw * nu * m) * (inte2 / (tp.R * tk))
#     term3 = - 1000 / (tp.Mw * nu * m) * v_w / (tp.R * tk) * delta_p
#     ln_a_w = np.log(p / p0) + inte2 / (tp.R * tk) + v_w / (tp.R * tk) * delta_p
#     os = term1 + term2 + term3
#     # os = -1000 / (nu * m * 18.015268) * ln_a_w
#     return {
#         'a_w': np.exp(ln_a_w),
#         'os': os,
#     }


def b(T, P):
    r = 8.314
    water = iapws.IAPWS97(T=T, P=P, x=1)
    v = water.Vapor.v / 1000 * 18.015268

    return r * 10 ** (-6) * T / P - v


def water_activity(T, P):
    """
    Calculate water activity from vapor pressure.
    :param T: temperature, in K.
    :param P: pressure, MPa
    :param m: molality of the salute
    :param nu: number of ions of the salute molecular.
    :return:
    """

    water = iapws.iapws97.IAPWS97(T=T, x=0)
    v_w = water.v * tp.Mw / 1000
    P0 = water.P
    delta_p = (P0 - P) * 10 ** 6

    def inte(p):
        water = iapws.IAPWS97(T=T, P=p, x=1)
        v = water.Vapor.v / 1000 * tp.Mw

        return tp.R * 10 ** (-6) * T / p - v

    result, error = quad(inte, P, P0)
    int_b = result * 10 ** 6

    ln_a_w = np.log(P / P0) + int_b / (tp.R * T) + v_w / (tp.R * T) * delta_p

    return ln_a_w


def osmotic_coefficient(T, P, m, nu):
    ln_a_w = water_activity(
        T=T,
        P=P,
    )

    os = -1000 / (nu * m * tp.Mw) * ln_a_w
    # print(P, np.exp(ln_a_w), os)
    return os


def vapor_pressure_objective_function(x, T, m, nu, phi):
    """
    Objective function to find the root (pressure) that satisfies the given osmotic coefficient.
    :param x: vapor pressure.
    :param T: temperature (K).
    :param m: molality (mol/kg).
    :param nu: number of moles.
    :param phi: target osmotic coefficient.
    :return:  squared residual between calculated osmotic coefficient and target osmotic coefficient.
    """

    calculated_phi = osmotic_coefficient(
        T=T,
        P=x,
        m=m,
        nu=nu
    )
    return (calculated_phi - phi) ** 2


def vapor_pressure(x0, T, m, nu, phi):
    """
    Solve for vapor pressure using scipy.optimize.root_scalar.
    :param T: temperature (K).
    :param m: molality (mol/kg).
    :param nu:  number of moles.
    :param phi: osmotic coefficient.
    :param x_guess: initial guess for pressure (MPa).
    :return: Vapor pressure (MPa).
    """

    # guess of pressure should always be less than Psat of pure water.
    p_max = iapws.iapws97._PSat_T(T)
    print('p_max:', p_max)
    bounds = [(0.0, p_max)]
    result = minimize(
        fun=vapor_pressure_objective_function,
        x0=x0,
        args=(T, m, nu, phi),
        method='Nelder-Mead',
        bounds=bounds,
        tol=1e-6
    )
    if result.success:
        return result.x[0]
    else:
        raise ValueError("Minimization failed to converge.")


def second_virial_LeFevre(T):
    """
    Calculate the second virial coefficient of ordinary water with the method of Le Fevre et al. (1975)
    :param T: temperature in K.
    :return: the second virial coefficient (B), in m^3/kg
    """
    m_w = 18.015257
    alpha = 10000  # K
    beta = 1500  # K
    a1 = 0.0015  # m^3/kg
    a2 = -0.000942  # m^3/kg
    a3 = -0.0004882  # m^3/kg

    f1 = 1 / (1 + T / alpha)
    f2 = (1 - np.exp(-beta / T)) ** (5 / 2) * (T / beta) ** (1 / 2) * np.exp(beta / T)
    f3 = beta / T

    b_t = a1 * f1 + a2 * f2 + a3 * f3

    b_t = m_w * b_t / 1000
    return b_t


def second_virial_hill(t):
    m_w = 18.015257
    alpha = 10000
    beta = 1500
    a1 = 0.160946868952524E-2
    a2 = -0.1270588577231565E-2
    a3 = -0.1947567798197269E-3
    a6 = 0.2960495174903245E-4
    f1 = (1 + t / alpha) ** (-1)
    f2 = (1 - np.exp(-beta / t)) ** (5 / 2) * (t / beta) ** (1 / 2) * np.exp(beta / t)
    f3 = beta / t
    f4 = (beta / t) ** 2
    f5 = (beta / t) ** 3
    f6 = (beta / t) ** 4
    b_t = a1 * f1 + a2 * f2 + a3 * f3 + a6 * f6
    b_t = m_w * b_t / 1000
    return b_t


def second_virial_harvey(t):
    a1 = 0.34404
    a2 = -0.75826
    a3 = -24.219
    a4 = -3978.2
    b1 = -0.5
    b2 = -0.8
    b3 = -3.35
    b4 = -8.3
    b_t = 1000 * (a1 * (t / 100) ** b1 + a2 * (t / 100) ** b2 + a3 * (t / 100) ** b3 + a4 * (t / 100) ** b4)
    b_t = b_t / 10 ** 6
    return b_t

# def osmotic_coefficient(t, p, nu, m, ):
#     """
#     Calculate the osmotic coefficient of water in a solution with the measure vapor pressure of this solution.
#     :param t: temperature in K.
#     :param p: pressure in kPa.
#     :param nu: stoichimetric number of the electrolyte.
#     :param m: molality.
#     :param b: the second virial coefficient of the water.
#     :return:
#     """
#
#     m_w = 18.015257
#     p_m = iapws.iapws97._PSat_T(t)  # mPa to kPa
#     p0 = p_m * 1000
#     v0 = iapws.iapws97._Region1(t, p_m)['v'] * m_w / 1000
#
#     # b = second_virial_LeFevre(t)
#     b = humidAir._virial(t)['Bww']
#     # b = second_virial_hill(t)
#     # b = second_virial_harvey(t)
#     delta_p = (p - p0) * 1000
#
#     # ln_a_w = np.log(p / p0) + 18.015268 * b /1000 * delta_p / (r * t)
#
#     ln_a_w = np.log(p / p0) + b * delta_p / (R * t) + v0 * delta_p / (R * t)
#     # ln_a_w = np.log(p / p0) + b * delta_p / (r * t)
#     os = -1000 / (nu * m * m_w) * ln_a_w
#     return np.exp(ln_a_w), os
