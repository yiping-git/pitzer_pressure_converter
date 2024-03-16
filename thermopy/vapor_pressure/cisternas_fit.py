# -*- coding: utf-8 -*-
# Author: Yiping Liu
# Description:
# Version: 1.0
# Last Modified: Mar 14, 2024

import pandas as pd
import numpy as np
from scipy.optimize import minimize

from utilities import get_charge_number, calculate_ionic_strength
from vapor_pressure.database import electrolyte_data


class VaporPressure:
    def __init__(self, x0, t_data, m_data, f_data, electrolyte):
        self.x0 = x0
        self.t_data = t_data
        self.m_data = m_data
        self.f_data = f_data
        self.electrolyte = electrolyte

    def ionic_strength(self, m):
        """
        Calculating the total ionic strength.
        :return: return the value of the total ionic strength.
        """
        e = self.electrolyte
        m = m

        e_data = electrolyte_data[e]
        comps = e_data['comp']
        keys = list(comps.keys())
        if get_charge_number(keys[0]) > 0:
            cation = keys[0]
            anion = keys[1]
        else:
            cation = keys[1]
            anion = keys[0]

        z_c = get_charge_number(cation)
        z_a = get_charge_number(anion)

        i = 1 / 2 * m * (comps[cation] * z_c ** 2 + comps[anion] * z_a ** 2)
        return i

    def k(self, x):
        k = x[0]
        return k

    def x_value(self, m):
        """
        Calculating x values every electrolyte.
        :return: a dict contains ionic strength of every electrolyte.
        """

        e = self.electrolyte
        m = m

        e_data = electrolyte_data[e]
        comps = e_data['comp']
        keys = list(comps.keys())

        if get_charge_number(keys[0]) > 0:
            cation = keys[0]
            anion = keys[1]
        else:
            cation = keys[1]
            anion = keys[0]
        z_c = get_charge_number(cation)
        z_a = get_charge_number(anion)

        x_value = 2 * m * (comps[cation] + comps[anion]) / (m * comps[cation] * z_c ** 2 + m * comps[anion] * z_a ** 2)

        return x_value

    def _vapor_pressure(self, x, m, t):
        m_w = 18.015257
        a_s = -0.021302
        # b_s = -5.390915
        b_s = x[3]
        # c_s = 7.2873
        c_s = x[1]
        # d_s = 1789.6279
        d_s = x[2]
        e_s = 39.53

        i = self.ionic_strength(m=m)

        k = self.k(x=x)
        x_value = self.x_value(m=m)

        a = a_s + 3.60591E-4 * i + m_w / 2303
        b = b_s + 1.382982 * i - 0.031185 * i ** 2
        c = c_s - 3.99334E-3 * i - 1.11614E-4 * i ** 2 + m_w * i * (1 - x_value) / 2303
        d = d_s - 0.138481 * i + 0.07511 * i ** 2 - 1.79277E-3 * i ** 3

        # a = a_s + x[6] * i + m_w / 2303
        # b = b_s + x[7] * i + x[8] * i ** 2
        # c = c_s + x[9] * i + x[10] * i ** 2 + m_w * i * (1 - x_value) / 2303
        # d = d_s + x[11] * i + x[12] * i ** 2 + x[13] * i ** 3

        result = 10 ** (
                k * i * (a - b / (t - e_s)) + (c - d / (t - e_s))
        ) * 0.01

        self._P = result
        return result

    def objective_function(self, x, t_data, m_data, f_data):
        print(x)

        cal_data = self._vapor_pressure(x, t=t_data, m=m_data)
        # cal_data = self.empirical(x=x, t=t_data, m=m_data)

        weighted_diffs = cal_data - f_data

        # Calculate the weighted sum of squared differences
        sigma = np.sqrt(np.mean(np.square(weighted_diffs)))

        return sigma

    def optimize(self):
        obj_fun = self.objective_function
        t_data = self.t_data
        m_data = self.m_data
        f_data = self.f_data

        res = minimize(
            fun=obj_fun,
            x0=self.x0,
            method='Nelder-Mead',
            args=(t_data, m_data, f_data,),
            options={'maxfev': 10000}
        )

        return res


data = pd.read_csv(r"E:\work\data\high-T\vapor pressure\KCl-H2O.csv")

solution = VaporPressure(
    x0=[0.1, 7.2873, 1789.6279,-5.390915],
    t_data=data['TK'],
    m_data=data['mKCl'],
    f_data=data['Pbar'],
    electrolyte='KCl'
)
# print(solution.f_data)
result = solution.optimize()
print(result)
