import numpy as np
import iapws

# test

import thermopy as tp
from thermopy.database import DbStoichio
from thermopy.utilities.functions import find_pair, get_charge_number, cache_array, cache_dict, cache_dict_array

from PitzerBase.functions import compute_j_jp


def get_alphas(x):
    return {
        'alpha': 2,  # kg^(1/2)⋅mol^(-1/2)
        'alpha_1': x[-2],  # kg^(1/2)⋅mol^(-1/2)
        'alpha_2': x[-1],  # kg^(1/2)⋅mol^(-1/2)
    }



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
    Partial differentiation of Bv to T [∂Bv/∂T]P.
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


def int_pbvt(P1, P2, T):
    b = bv_KCl
    """
    Intgration of partial differentiation of Bv to T: ∫[∂Bv/∂T]P from P1 to P2.
    :param b: parameter
    :param T: temperature in K
    :param P1: start pressure in MPa
    :param P2: end pressure in Mpa
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
