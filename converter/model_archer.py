import sys
import os
import iapws
import numpy as np

# test
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import converter.methods as pm

import thermopy as tp
from thermopy.database import DbStoichio
from thermopy.utilities import get_charge_number, calculate_ionic_strength
from thermopy.water import dielectric, a_phi, a_h, a_j, a_v


class PressureConverter:
    def __init__(self, T, P1, P2, m, salt,
                 int_pbeta0p=None,
                 int_pbeta1p=None,
                 int_pcphip=None,
                 int_pbvt=None,
                 int_pcvt=None,
                 int_pcvt2=None,
                 int_pv0t2=None
                 ):
        """
        :param T: temperature, k.
        :param P1: start pressure MPa.
        :param P2: end pressure MPa.
        :param m: concentration, molality.
        :param m1: start concentration.
        :param m2: end concentration.
        :param int_pbetap0: # ∫{[∂β(0)/∂P]T}dp, P1 -> P2
        :param int_pbetap1: # ∫{[∂β(1)/∂P]T}dp, P1 -> P2
        :param int_pcphip : # ∫{[∂C^ϕ/∂P]T}dp, P1 -> P2
        :param int_pbvt   : # ∫{[∂Bv/∂T]P}dp, P1 -> P2
        :param int_pcvt   : # ∫{[∂Cv/∂T]P}dp, P1 -> P2
        :param int_pbvt2  : # ∫{[∂²Bv/∂T²]P}dp, P1 -> P2
        :param int_pcvt2  : # ∫{[∂²Bv/∂T²]P}dp, P1 -> P2
        :param int_pv0t2  : # ∫{[∂²V0(2)/∂T²]P}dp, P1 -> P2, apparent molar volume of the solute.
        """
        self.T = T
        self.m = m
        self.P1 = P1
        self.P2 = P2
        self.salt = salt
        self.salt_data = self.get_salt_data()
        self.int_pbeta0p = int_pbeta0p
        self.int_pbeta1p = int_pbeta1p
        self.int_pcphip = int_pcphip
        self.int_pbvt = int_pbvt
        self.int_pcvt = int_pcvt
        self.int_pbvt2 = int_pcvt2
        self.int_pcvt2 = int_pcvt2
        self.int_pv0t2 = int_pv0t2
        self.alpha = 1

    def get_salt_data(self):
        data = DbStoichio(compound=self.salt)
        return data

    def get_molalities(self, m):
        cations = self.salt_data.cations
        anions = self.salt_data.anions
        neutral = self.salt_data.neutrals

        molalities = {}

        for cation in cations:
            molalities[cation] = m * self.salt_data.coefficients[cation]
        for anion in anions:
            molalities[anion] = m * self.salt_data.coefficients[anion]

        return molalities

    def get_ionic_strength(self, m):
        """
        Calculating ionic strength, can be molality based or mole fraction based.
        :param m: moalilty of the electrolyte.
        :return: ionic strength.
        """
        i = calculate_ionic_strength(self.get_molalities(m))
        return i

    @staticmethod
    def get_a_phi(T, P):
        return a_phi(T=T, P=P, method="Fernandez")

    @staticmethod
    def get_a_l(T, P):
        return a_h(T=T, P=P, method="Fernandez")

    @staticmethod
    def get_a_j(T, P):
        return a_j(T=T, P=P, method="Fernandez")

    def h(self, i):
        b = 1.2
        return np.log(1 + b * np.sqrt(i)) / (2 * b)

    def osmotic_coefficient_difference(self):
        """
        Returns:the difference between φ(P2) and φ(P1), that is, φ(P2) - φ(P1).
        """
        alpha = self.alpha
        cation = self.salt_data.cations[0]
        anion = self.salt_data.cations[0]
        z_c = get_charge_number(cation)
        z_a = get_charge_number(anion)

        nu_c = self.salt_data.coefficients[cation]
        nu_a = self.salt_data.coefficients[anion]

        nu = nu_c + nu_a
        i = self.get_ionic_strength(self.m)

        a_phi_p1 = self.get_a_phi(self.T, self.P1)
        a_phi_p2 = self.get_a_phi(self.T, self.P2)
        return - abs(z_c * z_a) * (a_phi_p2 - a_phi_p1) * np.sqrt(i) / (1 + 1.2 * np.sqrt(i)) + 2 * nu_c * nu_a / nu * (
                self.m * self.int_pbeta0p + self.m * self.int_pbeta1p * np.exp(-alpha * np.sqrt(i)) + self.m ** 2 * (
                nu_c * nu_a) ** (
                        1 / 2) * self.int_pcphip
        )

    def activity_coefficient_difference(self):
        """
        Returns:the difference between γ±(P2) and γ±(P1), that is, γ±(P2) - γ±(P1).
        """
        alpha = self.alpha
        cation = self.salt_data.cations[0]
        anion = self.salt_data.cations[0]
        z_c = get_charge_number(cation)
        z_a = get_charge_number(anion)

        nu_c = self.salt_data.coefficients[cation]
        nu_a = self.salt_data.coefficients[anion]

        nu = nu_c + nu_a
        i = self.get_ionic_strength(self.m)
        a_phi_p1 = self.get_a_phi(self.T, self.P1)
        a_phi_p2 = self.get_a_phi(self.T, self.P2)

        return - abs(z_c * z_a) * (a_phi_p2 - a_phi_p1) * (
                np.sqrt(i) / (1 + np.sqrt(i)) + 2 / 1.2 * np.log(1 + 1.2 * np.sqrt(i))) \
               + 2 * nu_c * nu_a / nu * (
                       2 * self.m * self.int_pbeta0p + 2 * self.m / (alpha ** 2 * i) * self.int_pbeta1p * (
                       1 - (1 + alpha * np.sqrt(i) - alpha ** 2 * i / 2) * np.exp(-alpha * np.sqrt(i))
               )
                       + 3 / 2 * self.m ** 2 * (nu_c * nu_a) ** (1 / 2) * self.int_pcphip
               )

    def apparent_molar_enthalpy_difference(self):
        """
        Returns: the difference between ϕL(P2) and ϕL(P1), that is, ϕL(P2) - ϕL(P1), J·mol⁻¹.
        """
        cation = self.salt_data.cations[0]
        anion = self.salt_data.cations[0]
        z_c = get_charge_number(cation)
        z_a = get_charge_number(anion)

        nu_c = self.salt_data.coefficients[cation]
        nu_a = self.salt_data.coefficients[anion]

        nu = nu_c + nu_a

        T = self.T
        m = self.m
        a_h_p1 = self.get_a_l(
            T=self.T,
            P=self.P1
        )
        a_h_p2 = self.get_a_l(
            T=self.T,
            P=self.P2
        )

        i = self.get_ionic_strength(m=m)

        return nu * abs(z_c * z_a) * (a_h_p2 - a_h_p1) * self.h(i=i) - 2 * nu_c * nu_a * tp.R * T ** 2 * (
                m * self.int_pbvt + m ** 2 * (nu_c * z_c) * self.int_pcvt
        )

    def apparent_molar_heat_capacity_difference(self):
        """
        Returns:the difference between ϕCp(P2) and ϕCp(P1), that is, ϕCp(P2) - ϕCp(P1).
        """
        cation = self.salt_data.cations[0]
        anion = self.salt_data.cations[0]
        z_c = get_charge_number(cation)
        z_a = get_charge_number(anion)

        nu_c = self.salt_data.coefficients[cation]
        nu_a = self.salt_data.coefficients[anion]

        nu = nu_c + nu_a
        i = self.get_ionic_strength(self.m)
        a_j_p1 = self.get_a_j(self.T, self.P1)
        a_j_p2 = self.get_a_j(self.T, self.P2)

        return nu * abs(z_c * z_a) * (a_j_p2 - a_j_p1) * self.h(
            i=i) + self.T * self.int_pv0t2 + 2 * nu_c * nu_a * tp.R * self.T ** 2 * (
                       self.m * (self.int_pbvt2 + 2 / self.T * self.int_pbvt) +
                       self.m ** 2 * nu_c * z_c * (self.int_pcvt2 + 2 / self.T * self.int_pcvt)
               )
