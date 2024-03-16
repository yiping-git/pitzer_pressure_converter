# -*- coding: utf-8 -*-
# Author: Yiping Liu
# Description:
# Version: 1.0
# Last Modified: Mar 14, 2024


from utilities import get_charge_number, calculate_ionic_strength
from vapor_pressure.database import electrolyte_data
import numpy as np


class VaporPressure:
    def __init__(self, t, electrolytes):
        self.t = t
        self.electrolytes = electrolytes
        self._P = 0

    def ionic_strength(self):
        """
        Calculating the total ionic strength.
        :return: return the value of the total ionic strength.
        """
        i = 0

        for key in self.electrolytes.keys():
            e = key
            m = self.electrolytes[key]
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

            i += 1 / 2 * m * (comps[cation] * z_c ** 2 + comps[anion] * z_a ** 2)
        return i

    def el_ionic_strength(self):
        """
        Calculating ionic strength of every electrolyte.
        :return: a dict contains ionic strength of every electrolyte.
        """
        total_i = self.ionic_strength()
        result = {}
        i = self.ionic_strength()
        for key in self.electrolytes.keys():
            e = key
            m = self.electrolytes[key]
            e_data = electrolyte_data[e]
            comps = e_data['comp']
            k = e_data['k']
            # k = 1.22717611-0.15985779*i + 0.00519674*i**2
            keys = list(comps.keys())

            if get_charge_number(keys[0]) > 0:
                cation = keys[0]
                anion = keys[1]
            else:
                cation = keys[1]
                anion = keys[0]
            z_c = get_charge_number(cation)
            z_a = get_charge_number(anion)

            y = 1 / 2 * m * (comps[cation] * z_c ** 2 + comps[anion] * z_a ** 2) / total_i

            # {'electrolyte':(k value, ionic strength of this electrolyte)}
            result[e] = (k, y)

        return result

    def x_values(self):
        """
        Calculating x values every electrolyte.
        :return: a dict contains ionic strength of every electrolyte.
        """
        total_i = self.ionic_strength()
        result = {}

        for key in self.electrolytes.keys():
            e = key
            m = self.electrolytes[key]

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

            x = 2 * m * (comps[cation] + comps[anion]) / (m * comps[cation] * z_c ** 2 + m * comps[anion] * z_a ** 2)
            y = 1 / 2 * m * (comps[cation] * z_c ** 2 + comps[anion] * z_a ** 2) / total_i
            result[e] = (x, y)
        return result

    def _vapor_pressure(self):
        m_w = 18.015257
        a_s = -0.021302
        b_s = -5.390915
        # c_s = 7.2873
        c_s = 7.18068809e+00
        # d_s = 1789.6279
        d_s = 1.73447867e+03
        e_s = 39.53

        i = self.ionic_strength()

        k_and_is = self.el_ionic_strength()
        x_and_ys = self.x_values()
        km = 0
        xm = 0

        keys = self.electrolytes.keys()

        for key in keys:
            values = k_and_is[key]
            k, y = values
            km += k * y
        for key in keys:
            values = x_and_ys[key]
            x, y = values
            xm += x * y

        a = a_s + 3.60591E-4 * i + m_w / 2303
        b = b_s + 1.382982 * i - 0.031185 * i ** 2
        c = c_s - 3.99334E-3 * i - 1.11614E-4 * i ** 2 + m_w * i * (1 - xm) / 2303
        d = d_s - 0.138481 * i + 0.07511 * i ** 2 - 1.79277E-3 * i ** 3

        result = 10 ** (
                km * i * (a - b / (self.t - e_s)) + (c - d / (self.t - e_s))
        ) * 0.01

        self._P = result
        return result

    @property
    def P(self):
        self._vapor_pressure()
        return self._P




