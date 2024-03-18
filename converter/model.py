import sys
import os

import numpy as np

# test
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import converter.methods as pm

import thermopy as tp
from thermopy.database import Db
from thermopy.utilities.functions import get_charge_number, calculate_ionic_strength


class PressureConverter:
    def __init__(self, t_data, p1_data, p2_data, m_data, salt):
        """

        :param t_data: temperature, k.
        :param p1_data: start pressure MPa.
        :param p2_data: end pressure MPa.
        :param m_data: concentration, molality.
        :param m1_data: start concentration.
        :param m2_data: end concentration.
        """
        self.t_data = t_data
        self.m_data = m_data
        self.p1_data = p1_data
        self.p2_data = p2_data
        self.salt = salt

    def ion_and_charge(self):
        solid_data = solids[self.salt]
        groups = {}
        for key, value in solid_data.items():
            if value['type'] == 'cation':
                groups['cation'] = (key, value['value'])
            elif value['type'] == 'anion':
                groups['anion'] = (key, value['value'])

        return groups

    def get_molalities(self, m):
        ic = self.ion_and_charge()
        molalities = {
            ic['cation'][0]: m * ic['cation'][1],
            ic['anion'][0]: m * ic['anion'][1],
        }
        return molalities

    def get_ionic_strength(self, m):
        """
        For calculating ionic strength, can be molality based or mole fraction based.
        :param m: moalilty of the electrolyte.
        :return: ionic strength.
        """
        i = calculate_ionic_strength(self.get_molalities(m))
        return i

    @staticmethod
    def get_a_phi(t, p):
        return pm.a_phi_iapws(t=t, p=p)

    @staticmethod
    def get_a_l(t, p):
        return pm.a_l_iapws(t=t, p=p)

    @staticmethod
    def get_a_j(t, p):
        return pm.a_j_iapws(t=t, p=p)

    def h(self, i):
        b = 1.2
        return np.log(1 + b * np.sqrt(i)) / (2 * b)

    def apparent_molar_enghtalpy_difference(self):
        ic = self.ion_and_charge()
        cation = ic['cation'][0]
        anion = ic['anion'][0]
        z_c = get_charge_number(cation)
        z_a = get_charge_number(anion)

        nu_c = ic['cation'][1]
        nu_a = ic['anion'][1]

        nu = nu_c + nu_a

        t = self.t_data
        m = self.m_data
        a_h_p1 = self.get_a_l(
            t=self.t_data,
            p=self.p1_data
        )
        a_h_p2 = self.get_a_l(
            t=self.t_data,
            p=self.p2_data
        )

        int_p_bv_t = pm.int_p_bv_t(
            P1=self.p1_data,
            P2=self.p2_data,
            T=self.t_data,
        )
        i = self.get_ionic_strength(m=m)

        return nu * abs(z_c * z_a) * (a_h_p1 - a_h_p2) * self.h(i=i) - 2 * nu_c * nu_a * tp.R * t ** 2 * (
                    m * int_p_bv_t)
