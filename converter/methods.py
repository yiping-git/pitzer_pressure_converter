import numpy as np
import itertools
import iapws
import functools
import warnings

from database.solid_data import solids
from public.low_level import find_pair, get_charge_number, cache_array, cache_dict, cache_dict_array
from public.j_x import compute_j_jp


R = 8.314

import numpy as np
import iapws


def dielectric(T,P):
    """
    Calculate the dielectric constant of water at given temperature.
    :param T:  temperature in Kelvin.
    :param P:  pressure, MPa.
    :return: dielectric constant.
    """

    # MPa to bar
    P = P  * 10

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




def partial_dielectric(T,P):
    """
    Calculate  ∂D/∂T at constant P (pressure).
    :param T: temperature in Kelvin
    :param P: pressure, MPa.
    :return: ∂D/∂T
    """

    P = P  * 10  # MPa to bar

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



def partial_ln_dt(T,P):
    """
    Calculate the ∂(lnD)/∂T
    :param T: temperature in Kelvin
    :return: ∂(lnD)/∂T
    """
    d = dielectric(T,P)
    pd = partial_dielectric(T,P)
    return pd / d




def get_alphas(x):
    return {
        'alpha': 2,  # kg^(1/2)⋅mol^(-1/2)
        'alpha_1': x[-2],  # kg^(1/2)⋅mol^(-1/2)
        'alpha_2': x[-1],  # kg^(1/2)⋅mol^(-1/2)
    }


"""
the Pitzer parameters
"""


def partial_derivative(f, t, dt=1e-5):
    """
    Calculate the partial derivative of a function f with respect to t using finite differences.

    Parameters:
    - f: Function that takes a single argument.
    - t: Value at which to calculate the partial derivative.
    - dt: Small increment in t.

    Returns:
    - Approximate value of the partial derivative at t.
    """
    return (f(t + dt) - f(t)) / dt


def a_phi_iapws(t, p):
    a = 6.0221367E23  # mol−1
    k = 1.380658E-23  # J·K−1
    e = 1.602176634E-19  # C
    ep_0 = 8.854187817E-12  # C^2·J−1·cm^−1

    if not p or np.isnan(p):
        water = iapws.iapws97.IAPWS97(T=t, x=0)
        p = water.P
    else:
        water = iapws.iapws97.IAPWS97(T=t, P=p)

    d = dielectric(T=t, P=p)
    d_si = d * 4 * np.pi * ep_0  # convert to SI unit
    rho = water.rho  # kg/m^3 to g/cm^3
    dhc = (1 / 3) * (2 * np.pi * a * rho) ** (1 / 2) * (e ** 2 / (d_si * k * t)) ** (3 / 2)

    return dhc


@cache_array
def a_phi_iapws_array(t_data, p_data):
    a = 6.0221367E23  # mol−1
    k = 1.380658E-23  # J·K−1
    e = 1.602176634E-19  # C
    ep_0 = 8.854187817E-12  # C^2·J−1·m−1 to C^2·J−1·m^−1

    # Initialize an empty NumPy array to collect the results
    dhc_results = np.empty_like(t_data, dtype=float)


    for i, t in enumerate(t_data):
        p = p_data[i]
        if not p or np.isnan(p):
            water = iapws.iapws97.IAPWS97(T=t, x=0)
            p = water.P
        else:
            water = iapws.iapws97.IAPWS97(T=t, P=p)

        d = dielectric(T=t, P=p)
        d_si = d * 4 * np.pi * ep_0

        rho = water.rho  # kg/m^3 to g/cm^3
        dhc = (1 / 3) * (2 * np.pi * a * rho) ** (1 / 2) * (e ** 2 / (d_si * k * t)) ** (3 / 2)
        dhc_results[i] = dhc
    return dhc_results


def a_phi_moller(t):
    a1 = 3.36901532e-01
    a2 = -6.32100430e-04
    a3 = 9.14252359e00
    a4 = -1.35143986e-02
    a5 = 2.26089488e-03
    a6 = 1.92118597e-06
    a7 = 4.52586464e01
    a8 = 0
    return a1 + a2 * t + a3 / t + a4 * np.log(t) + a5 / (t - 263) + a6 * t ** 2 + a7 / (680 - t) + a8 / (t - 227)


def a_l_iapws(t, p):
    if not p or np.isnan(p):
        water = iapws.iapws97.IAPWS97(T=t, x=0)
        p = water.P
    else:
        water = iapws.iapws97.IAPWS97(T=t, P=p)
    alpha_w = water.alfav
    a_phi = a_phi_iapws(t=t, p=p)

    p_lnd_t = partial_ln_dt(T=t, P=p)
    dhc = -6 * a_phi * R * t * (1 + t * p_lnd_t + t / 3 * alpha_w)
    return dhc


def a_j_iapws(t, p):
    if not p or np.isnan(p):
        water = iapws.iapws97.IAPWS97(T=t, x=0)
        p = water.P
    dhc_results = partial_derivative(lambda t: a_l_iapws(t, p), t)
    return dhc_results


@cache_array
def a_j_iapws_array(t_data, p_data):
    dhc_results = np.empty_like(t_data, dtype=float)
    for i, t, in enumerate(t_data):
        p = p_data[i]
        dhc_results[i] = partial_derivative(lambda t: a_l_iapws(t, p), t)
    return dhc_results


def g_func(a):
    result = 2 * (
            1 - (1 + a) * np.exp(-a)
    ) / a ** 2
    return result


def g_func_prime(a):
    g_prime = -2 * (
            1 - (1 + a + a ** 2 / 2) * np.exp(-a)
    ) / a ** 2
    return g_prime


def get_c(c_phi, z_m, z_x):
    c_mx = c_phi / (2 * (abs(z_m * z_x)) ** 0.5)
    return c_mx


def get_c_gamma(c_phi):
    c_gamma = 3 * c_phi / 2
    return c_gamma


def get_f(a_phi, i):
    """
    :param a_phi: A_phi (Debye-Hukel constant)
    :param i: ionic strength
    :return: expression of "f" function
    """
    b = 1.2
    f = - (4 * i * a_phi / b) * np.log(1 + b * np.sqrt(i))
    return f


def get_f_gamma(a_phi, i):
    """
    :param a_phi:A_phi (Debye-Hukel constant)
    :param i: ionic strength
    :return: the "f^gamma" function in Pitzer's model
    """
    b = 1.2
    f_gamma = - a_phi * (
            np.sqrt(i) / (1 + b * np.sqrt(i)) + (2 / b) * np.log(
        1 + b * np.sqrt(i))
    )
    return f_gamma


"""
*** End of the Pitzer parameters dealing *** 
"""


def get_x_mn(z_m: int, z_n: int, a_phi: float, i: float):
    """
    calculate the 'x' value of ions 'm' and 'n'.
    :param z_m: charge number of ion 'm'
    :param z_n: charge number of ion 'n'
    :param a_phi: Avogedral's number of this solution
    :param i: Ionic strength of this solution
    :return: 'x_mn' for calculating the 'J' value
    :reference: [2] p9
    """
    x_mn = 6 * z_m * z_n * a_phi * i ** 0.5
    return x_mn


def get_e_theta(z_m, z_n, a_phi, i):
    """
    :param z_m: charge number of species m
    :param z_n: charge number of species n
    :param a_phi:
    :param i: ionic strength
    :return: e_theta and e_theta_prime
    :reference: [1] p123
    """

    x_mn = get_x_mn(z_m, z_n, a_phi, i)
    x_mm = get_x_mn(z_m, z_m, a_phi, i)
    x_nn = get_x_mn(z_n, z_n, a_phi, i)

    mn = compute_j_jp(x_mn)
    mm = compute_j_jp(x_mm)
    nn = compute_j_jp(x_nn)
    j_mn = mn['j_x']
    j_mn_prime = mn['j_xp']
    j_mm = mm['j_x']
    j_mm_prime = mm['j_xp']
    j_nn = nn['j_x']
    j_nn_prime = nn['j_xp']

    e_theta = (z_m * z_n / (4 * i)) * (j_mn - 0.5 * j_mm - 0.5 * j_nn)
    e_theta_prime = -(e_theta / i) + (z_m * z_n / (8 * i ** 2)) * (
            x_mn * j_mn_prime - 0.5 * x_mm * j_mm_prime - 0.5 * x_nn * j_nn_prime)
    return {
        "e_theta": e_theta,
        "e_theta_prime": e_theta_prime,
    }


def calculate_ionic_strength(molalities):
    data = molalities
    ions = data.keys()
    sum_value = 0
    for ion in ions:
        charge_number = get_charge_number(ion)
        sum_value += data[ion] * (charge_number ** 2)
    return sum_value / 2


def calculate_charge_balance(molalities):
    balance = 0
    for species in molalities.keys():
        balance += get_charge_number(species) * molalities[species]
    return balance


def get_hydrate_data(solid):
    data = {}
    if solid in solids.keys():
        data = solids[solid]
    return data


"""
Pitzer-model-only methods
"""


def thermal_property_function(x, t):
    return x[0] \
           + x[1] * t \
           + x[2] * t ** 2 \
           + x[3] * t ** 3 \
           + x[4] * t ** 4 \
           + x[5] * t ** 5 \
           + x[6] * t ** 6


def heat_capacity_function(x, t):
    # return (10.98 + 0.0039 * t)*4.184 \
    return x[0] \
           + x[1] * t \
           + x[2] * t ** 2 \
           + x[3] * t ** 3 \
           + x[4] * t ** 4 \
           + x[5] * t ** 5 \
           + x[6] * t ** 6


# p = np.array([
#     -1.11253832e-03, -4.18974580e-03, 9.30123650e-04, 2.48134099e-04,
#     -1.42984003e-06, 2.88241623e-09, -1.98968628e-12,
# ])
# print(heat_capacity_function(p, 373.16))


def parameter_function_archer_2000(a, t_data, p_data):
    result = np.zeros_like(t_data)
    for i, t in enumerate(t_data):
        p = p_data[i]
        if not p:
            p = iapws.iapws97._PSat_T(t)
        result[i] = a[0] \
                    + a[1] * t / 1000 \
                    + a[2] * (t / 500) ** 2 \
                    + a[3] / (t - 215) \
                    + a[4] * 1 * 10 ** 4 * (1 / (t - 215)) ** 3 \
                    + a[5] * 100 * (1 / (t - 215)) ** 2 \
                    + a[6] * 200 * (1 / t) ** 2 \
                    + a[7] * (t / 500) ** 3 \
                    + a[8] * (1 / (650 - t)) ** (1 / 2) \
                    + a[9] * 1 * 10 ** (-5) * p \
                    + a[10] * 2 * 10 ** (-4) * p * (1 / (t - 225)) \
                    + a[11] * 100 * p * (1 / (650 - t)) ** 3 \
                    + a[12] * 1 * 10 ** (-5) * p * (t / 500) \
                    + a[13] * 2 * 10 ** (-4) * p * (1 / (650 - t)) \
                    + a[14] * 1 * 10 ** (-7) * p ** 2 \
                    + a[15] * 2 * 10 ** (-6) * p ** 2 * (1 / (t - 225)) \
                    + a[16] * p ** 2 * (1 / (650 - t)) ** 3 \
                    + a[17] * 1 * 10 ** (-7) * p ** 2 * (t / 500) \
                    + a[18] * 1 * 10 ** (-7) * p ** 2 * (t / 500) ** 2 \
                    + a[19] * 4 * 10 ** (-2) * p * (1 / (t - 225)) ** 2 \
                    + a[20] * 1 * 10 ** (-5) * p * (t / 500) ** 2 \
                    + a[21] * 2 * 10 ** (-8) * p ** 3 * (1 / (t - 225)) \
                    + a[22] * 1 * 10 ** (-2) * p ** 3 * (1 / (650 - t)) ** 3 \
                    + a[23] * 200 * (1 / (650 - t)) ** 3
    return result


def d_parameter_function_archer_2000(a, t_data, p_data):
    result = np.zeros_like(t_data)
    for i, t in enumerate(t_data):
        p = p_data[i]
        if not p:
            # if no pressure provided, use P(sat) of pure water.
            p = iapws.iapws97._PSat_T(t)
        result[i] = a[1] / 1000 \
                    - a[10] * 0.0002 * p / (t - 225) ** 2 \
                    + a[11] * 300 * p / (650 - t) ** 4 \
                    + a[12] * 2.0e-8 * p \
                    + a[13] * 0.0002 * p / (650 - t) ** 2 \
                    - a[15] * 2.0e-6 * p ** 2 / (t - 225) ** 2 \
                    + a[16] * 3 * p ** 2 / (650 - t) ** 4 \
                    + a[17] * 2.0e-10 * p ** 2 \
                    + a[18] * 8.0e-13 * p ** 2 * t \
                    - a[19] * 0.08 * p / (t - 225) ** 3 \
                    + a[2] * t / 125000 \
                    + a[20] * 8.0e-11 * p * t \
                    - a[21] * 2.0e-8 * p ** 3 / (t - 225) ** 2 \
                    + a[22] * 0.03 * p ** 3 / (650 - t) ** 4 \
                    + a[23] * 600 / (650 - t) ** 4 \
                    - a[3] / (t - 215) ** 2 \
                    - a[4] * 30000 / (t - 215) ** 4 \
                    - a[5] * 200 / (t - 215) ** 3 \
                    - a[6] * 400 / t ** 3 \
                    + a[7] * 3 * t ** 2 / 125000000 \
                    + a[8] * 0.5 * (1 / (650 - t)) ** 0.5 / (650 - t)
    return result


def dd_parameter_function_archer_2000(a, t_data, p_data):
    result = np.zeros_like(t_data)
    for i, t in enumerate(t_data):
        p = p_data[i]
        if not p:
            # if no pressure provided, use P(sat) of pure water.
            p = iapws.iapws97._PSat_T(t)
        result[i] = a[10] * 0.0004 * p / (t - 225) ** 3 \
                    + a[11] * 1200 * p / (650 - t) ** 5 \
                    + a[13] * 0.0004 * p / (650 - t) ** 3 \
                    + a[15] * 4.0e-6 * p ** 2 / (t - 225) ** 3 \
                    + a[16] * 12 * p ** 2 / (650 - t) ** 5 \
                    + a[18] * 8.0e-13 * p ** 2 \
                    + a[19] * 0.24 * p / (t - 225) ** 4 \
                    + a[2] / 125000 \
                    + a[20] * 8.0e-11 * p \
                    + a[21] * 4.0e-8 * p ** 3 / (t - 225) ** 3 \
                    + a[22] * 0.12 * p ** 3 / (650 - t) ** 5 \
                    + a[23] * 2400 / (650 - t) ** 5 \
                    + a[3] * 2 / (t - 215) ** 3 \
                    + a[4] * 120000 / (t - 215) ** 5 \
                    + a[5] * 600 / (t - 215) ** 4 \
                    + a[6] * 1200 / t ** 4 \
                    + a[7] * 3 * t / 62500000 \
                    + a[8] * 0.75 * (1 / (650 - t)) ** 0.5 / (650 - t) ** 2
    return result


v0_KCl = np.array([
    1.56152E03,
    -1.69234E05,
    -4.29918E00,
    4.59233E-03,
    -3.25686E04,
    -6.86887E00,
    7.35220E02,
    2.02245E-02,
    -2.15779E-05,
    1.03212E02,
    5.34941E-03,
    -5.73121E-01,
    -1.57862E-05,
    1.66987E-08,
    -7.22012E-02,
])
bv_KCl = np.array([
    0.0,
    0.0,
    9.45015E-08,
    -2.90741E-10,
    3.26205E-03,
    8.39662E-07,
    0.0,
    -4.41638E-09,
    6.71235E-12,
    -4.42327E-05,
    -7.97437E-10,
    0.0,
    4.12771E-12,
    -6.24996E-15,
    4.16221E-08,
])


def bv(b, T, P):
    """

    :param b: parameter
    :param T: temperature in K
    :param P: pressure in bar
    :return:
    """
    return b[0] \
           + b[1] / T \
           + b[2] * T \
           + b[3] * T ** 2 \
           + b[4] / (647 - T) \
           + P * (b[5]
                  + b[6] / T
                  + b[7] * T
                  + b[8] * T ** 2
                  + b[9] / (647 - T)
                  ) \
           + P ** 2 * (b[10]
                       + b[11] / T
                       + b[12] * T
                       + b[13] * T ** 2
                       + b[14] / (647 - T)
                       )


def p_bv_t(b, T, P):
    """

    :param b: parameter
    :param T: temperature in K
    :param P: pressure in bar
    :return:
    """
    return - b[1] / T ** 2 \
           + b[2] \
           + b[3] * 2 * T \
           + b[4] / (647 - T) ** 2 \
           + P * (
                   - b[6] / T ** 2
                   + b[7]
                   + b[8] * 2 * T
                   + b[9] / (647 - T) ** 2
           ) \
           + P ** 2 * (
                   - b[11] / T ** 2
                   + b[12]
                   + b[13] * 2 * T
                   + b[14] / (647 - T) ** 2
           )


from scipy.integrate import quad

def int_p_bv_t(P1, P2, T):
    b = bv_KCl
    """

    :param b: parameter
    :param T: temperature in K
    :param P1: start pressure in bar
    :param P2: end pressure in bar
    :return:
    """
    # return -P1 ** 3 * (
    #         2 * T ** 5 * b[13] + T ** 4 * b[12] - 2588 * T ** 4 * b[13] - 1294 * T ** 3 * b[12] + 837218 * T ** 3 * b[
    #     13] - T ** 2 * b[11] + 418609 * T ** 2 * b[12] + T ** 2 * b[14] + 1294 * T * b[11] - 418609 * b[11]) \
    #        / (
    #                3 * T ** 4 - 3882 * T ** 3 + 1255827 * T ** 2) - P1 ** 2 * (
    #                2 * T ** 5 * b[8] + T ** 4 * b[7] - 2588 * T ** 4 * b[8] - 1294 * T ** 3 * b[7] + 837218 * T ** 3 *
    #                b[8] - T ** 2 * b[6] + 418609 * T ** 2 * b[7] + T ** 2 * b[9] + 1294 * T * b[6] - 418609 * b[6]) / (
    #                2 * T ** 4 - 2588 * T ** 3 + 837218 * T ** 2) - P1 * (
    #                2 * T ** 5 * b[3] + T ** 4 * b[2] - 2588 * T ** 4 * b[3] - 1294 * T ** 3 * b[2] + 837218 * T ** 3 *
    #                b[3] - T ** 2 * b[1] + 418609 * T ** 2 * b[2] + T ** 2 * b[4] + 1294 * T * b[1] - 418609 * b[1]) / (
    #                T ** 4 - 1294 * T ** 3 + 418609 * T ** 2) + P2 ** 3 * (
    #                2 * T ** 5 * b[13] + T ** 4 * b[12] - 2588 * T ** 4 * b[13] - 1294 * T ** 3 * b[
    #            12] + 837218 * T ** 3 * b[13] - T ** 2 * b[11] + 418609 * T ** 2 * b[12] + T ** 2 * b[14] + 1294 * T * b[
    #                    11] - 418609 * b[11]) / (
    #                3 * T ** 4 - 3882 * T ** 3 + 1255827 * T ** 2) + P2 ** 2 * (
    #                2 * T ** 5 * b[8] + T ** 4 * b[7] - 2588 * T ** 4 * b[8] - 1294 * T ** 3 * b[7] + 837218 * T ** 3 *
    #                b[8] - T ** 2 * b[6] + 418609 * T ** 2 * b[7] + T ** 2 * b[9] + 1294 * T * b[6] - 418609 * b[6]) / (
    #                2 * T ** 4 - 2588 * T ** 3 + 837218 * T ** 2) + P2 * (
    #                2 * T ** 5 * b[3] + T ** 4 * b[2] - 2588 * T ** 4 * b[3] - 1294 * T ** 3 * b[2] + 837218 * T ** 3 *
    #                b[3] - T ** 2 * b[1] + 418609 * T ** 2 * b[2] + T ** 2 * b[4] + 1294 * T * b[1] - 418609 * b[1]) / (
    #                T ** 4 - 1294 * T ** 3 + 418609 * T ** 2)
    result, error = quad(lambda P: p_bv_t(b, T, P), P1, P2)
    return result


def get_beta_012(x, t, p, para_num, has_b2):
    b0 = parameter_function_archer_2000(
        a=x[:para_num],
        t_data=t,
        p_data=p
    )
    b1 = parameter_function_archer_2000(
        a=x[para_num:2 * para_num],
        t_data=t,
        p_data=p
    )
    if has_b2:
        b2 = parameter_function_archer_2000(
            a=x[2 * para_num:3 * para_num],
            t_data=t,
            p_data=p
        )
    else:
        b2 = 0
    return {
        'b0': b0,
        'b1': b1,
        'b2': b2,
    }


def get_beta_l_012(x, t, p, para_num, has_b2):
    b0 = d_parameter_function_archer_2000(
        a=x[:para_num],
        t_data=t,
        p_data=p
    )
    b1 = d_parameter_function_archer_2000(
        a=x[para_num:2 * para_num],
        t_data=t,
        p_data=p
    )
    if has_b2:
        b2 = d_parameter_function_archer_2000(
            a=x[2 * para_num:3 * para_num],
            t_data=t,
            p_data=p
        )
    else:
        b2 = 0
    return {
        'b0': b0,
        'b1': b1,
        'b2': b2,
    }


def get_beta_j_012(x, t, p, para_num, has_b2):
    b0 = dd_parameter_function_archer_2000(
        a=x[:para_num],
        t_data=t,
        p_data=p
    )
    b1 = dd_parameter_function_archer_2000(
        a=x[para_num:2 * para_num],
        t_data=t,
        p_data=p
    )
    if has_b2:
        b2 = dd_parameter_function_archer_2000(
            a=x[2 * para_num:3 * para_num],
            t_data=t,
            p_data=p
        )
    else:
        b2 = 0
    return {
        'b0': b0,
        'b1': b1,
        'b2': b2,
    }
