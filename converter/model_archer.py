import sys
import os
import iapws
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import converter.methods as pm

import thermopy as tp
from thermopy.database import DbStoichio
from thermopy.utilities import get_charge_number, calculate_ionic_strength
from thermopy.water import dielectric, a_phi, a_h, a_j, a_v


class PressureConverter:
    def __init__(self, T, P1, P2, m, salt,
                 int_pb0p=None,
                 int_pb1p=None,
                 int_pc0p=None,
                 int_pc1p=None,
                 int_pb0pt=None,
                 int_pb1pt=None,
                 int_pc0pt=None,
                 int_pc1pt=None,
                 int_pb0ptt=None,
                 int_pb1ptt=None,
                 int_pc0ptt=None,
                 int_pc1ptt=None,
                 int_pv0t2=None,
                 ):
        """
        :param T: temperature, k.
        :param P1: start pressure MPa.
        :param P2: end pressure MPa.
        :param m: concentration, molality.
        :param m1: start concentration.
        :param m2: end concentration.
        :param int_pbp0: # ∫{[∂β(0)/∂P]T}dp, P1 -> P2
        :param int_pbp1: # ∫{[∂β(1)/∂P]T}dp, P1 -> P2
        :param int_pc0p : # ∫{[∂c0/∂P]T}dp, P1 -> P2
        :param int_pc1p : # ∫{[∂c1/∂P]T}dp, P1 -> P2
        :param int_pb0pt   : # ∫{[∂β(0)/∂P/∂T]}dp, P1 -> P2
        :param int_pb1pt   : # ∫{[∂β(1)/∂P/∂T]}dp, P1 -> P2
        :param int_pc0pt   : # ∫{[∂c0/∂P/∂T]}dp, P1 -> P2
        :param int_pc1pt   : # ∫{[∂c1/∂P/∂T]}dp, P1 -> P2
        :param int_pb0ptt   : # ∫{[∂β(0)/∂P/∂T²]}dp, P1 -> P2
        :param int_pb1ptt   : # ∫{[∂β(1)/∂P/∂T²]}dp, P1 -> P2
        :param int_pc0ptt   : # ∫{[∂c0/∂P/∂T²]}dp, P1 -> P2
        :param int_pc1ptt   : # ∫{[∂c1/∂P/∂T²]}dp, P1 -> P2

        :param int_pv0t2  : # ∫{[∂²V0(2)/∂T²]P}dp, P1 -> P2, apparent molar volume of the aqueous solute.
        """
        self.T = T
        self.m = m
        self.P1 = P1
        self.P2 = P2
        self.salt = salt
        self.salt_data = self.get_salt_data()
        self.int_pb0p = int_pb0p
        self.int_pb1p = int_pb1p
        self.int_pc0p = int_pc0p
        self.int_pc1p = int_pc1p
        self.int_pb0pt = int_pb0pt
        self.int_pb1pt = int_pb1pt
        self.int_pc0pt = int_pc0pt
        self.int_pc1pt = int_pc1pt
        self.int_pb0ptt = int_pb0ptt
        self.int_pb1ptt = int_pb1ptt
        self.int_pc0ptt = int_pc0ptt
        self.int_pc1ptt = int_pc1ptt
        self.int_pv0t2 = int_pv0t2
        self.alpha = 2
        self.alpha2 = 2.5

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

        # check input
        if not self.int_pb0p:
            print("Needs input of 'int_pb0p' (= ∫{[∂β(0)/∂P]T}dp, P1 -> P2)")
        if not self.int_pb1p:
            print("Needs input of 'int_pb1p' (= ∫{[∂β(1)/∂P]T}dp, P1 -> P2)")
        if not self.int_pc0p:
            print("Needs input of 'int_pc01p' (= ∫{[∂c(0)/∂P]T}dp, P1 -> P2)")
        if not self.int_pc1p:
            print("Needs input of 'int_pc01p' (= ∫{[∂c(1)/∂P]T}dp, P1 -> P2)")

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

        return - abs(z_c * z_a) * (a_phi_p2 - a_phi_p1) * np.sqrt(i) / (
                1 + 1.2 * np.sqrt(i)) \
               + self.m * 2 * nu_c * nu_a / nu * (
                       self.int_pb0p + self.int_pb1p * np.exp(-self.alpha * np.sqrt(i))
               ) \
               + self.m ** 2 * 4 * nu_c ** 2 * nu_a * z_c / nu * (
                       self.int_pc0p + self.int_pc1p * np.exp(-self.alpha2 * np.sqrt(i))
               )

    def activity_coefficient_difference(self):
        """
        Returns:the difference between γ±(P2) and γ±(P1), that is, γ±(P2) - γ±(P1).
        """

        # check input
        if not self.int_pb0p:
            print("Needs input of 'int_pb0p' (= ∫{[∂β(0)/∂P]T}dp, P1 -> P2)")
        if not self.int_pb1p:
            print("Needs input of 'int_pb1p' (= ∫{[∂β(1)/∂P]T}dp, P1 -> P2)")
        if not self.int_pc0p:
            print("Needs input of 'int_pc01p' (= ∫{[∂c(0)/∂P]T}dp, P1 -> P2)")
        if not self.int_pc1p:
            print("Needs input of 'int_pc01p' (= ∫{[∂c(1)/∂P]T}dp, P1 -> P2)")

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
                np.sqrt(i) / (1 + np.sqrt(i)) + 2 / 1.2 * np.log(1 + 1.2 * np.sqrt(i))
        ) \
               + self.m * 2 * nu_c * nu_a / nu * (
                       2 * self.int_pb0p + 2 / (alpha ** 2 * i) * self.int_pb1p * (
                       1 - (1 + alpha * np.sqrt(i) - alpha ** 2 * i / 2) * np.exp(-alpha * np.sqrt(i)))
               ) \
               + self.m ** 2 * (2 * nu_c ** 2 * nu_a * z_c) / nu * (
                       3 * self.int_pc0p + 4 * self.int_pc1p * (
                       6 - (6 + 6 * self.alpha2 * np.sqrt(i) + 3 * self.alpha2 ** 2 * i + self.alpha2 ** 3 * i ** (
                       3 / 2) - self.alpha2 ** 4 * i ** 2 / 2) * np.exp(-self.alpha2 * np.sqrt(i))
               ) / (
                               self.alpha2 ** 4 * i ** 2
                       )
               )

    def apparent_molar_enthalpy_difference(self):
        """
        Returns: the difference between ϕL(P2) and ϕL(P1), that is, ϕL(P2) - ϕL(P1), J·mol⁻¹.
        """

        # check input
        if not self.int_pb0pt:
            print("Needs input of 'int_pb0pt' (= ∫{[∂β(0)/∂P/∂T]T}dp, P1 -> P2)")
        if not self.int_pb1pt:
            print("Needs input of 'int_pb1pt' (= ∫{[∂β(1)/∂P/∂T]T}dp, P1 -> P2)")
        if not self.int_pc0pt:
            print("Needs input of 'int_pc0pt' (= ∫{[∂c(0)/∂P/∂T]T}dp, P1 -> P2)")
        if not self.int_pc1pt:
            print("Needs input of 'int_pc1pt' (= ∫{[∂c(1)/∂P/∂T]T}dp, P1 -> P2)")


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

        int_pbvt = self.int_pb0pt + 2 * self.int_pb1pt * (
                1 - (1 + self.alpha * np.sqrt(i)) * np.exp(-self.alpha * np.sqrt(i))
        ) / (self.alpha ** 2 * i)

        int_pcvt = self.int_pc0pt + 4 * self.int_pc1pt * (
                6 - (
                6 + 6 * self.alpha2 * np.sqrt(i) + 3 * self.alpha2 ** 2 * i + self.alpha2 ** 3 * i ** (3 / 2)
        ) * np.exp(-self.alpha2 * np.sqrt(i))
        ) / (self.alpha2 ** 4 * i ** 2)

        return nu * abs(z_c * z_a) * (a_h_p2 - a_h_p1) * self.h(i=i) \
               - 2 * nu_c * nu_a * tp.R * T ** 2 * (
                       m * int_pbvt
                       + m ** 2 * (nu_c * z_c) * int_pcvt
               )

    def apparent_molar_heat_capacity_difference(self):
        """
        Returns:the difference between ϕCp(P2) and ϕCp(P1), that is, ϕCp(P2) - ϕCp(P1).
        """
        # check input
        if not self.int_pb0pt:
            print("Needs input of 'int_pb0pt' (= ∫{[∂β(0)/∂P/∂T]T}dp, P1 -> P2)")
        if not self.int_pb1pt:
            print("Needs input of 'int_pb1pt' (= ∫{[∂β(1)/∂P/∂T]T}dp, P1 -> P2)")
        if not self.int_pc0pt:
            print("Needs input of 'int_pc0pt' (= ∫{[∂c(0)/∂P/∂T]T}dp, P1 -> P2)")
        if not self.int_pc1pt:
            print("Needs input of 'int_pc1pt' (= ∫{[∂c(1)/∂P/∂T]T}dp, P1 -> P2)")
        if not self.int_pb0ptt:
            print("Needs input of 'int_pb0ptt' (= ∫{[∂β(0)/∂P/∂T²]}dp, P1 -> P2")
        if not self.int_pb1ptt:
            print("Needs input of 'int_pb1ptt' (= ∫{[∂β(1)/∂P/∂T²]}dp, P1 -> P2")
        if not self.int_pc0ptt:
            print("Needs input of 'int_pb1ptt' (= ∫{[∂c(0)/∂P/∂T²]}dp, P1 -> P2")
        if not self.int_pc1ptt:
            print("Needs input of 'int_pb1ptt' (= ∫{[∂c(1)/∂P/∂T²]}dp, P1 -> P2")


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

        int_pbvt = self.int_pb0pt + 2 * self.int_pb1pt * (
                1 - (1 + self.alpha * np.sqrt(i)) * np.exp(-self.alpha * np.sqrt(i))
        ) / (self.alpha ** 2 * i)

        int_pbvt2 = self.int_pb0ptt + 2 * self.int_pb1ptt * (
                1 - (1 + self.alpha * np.sqrt(i)) * np.exp(-self.alpha * np.sqrt(i))
        ) / (self.alpha ** 2 * i)

        int_pcvt = self.int_pc0pt + 4 * self.int_pc1pt * (
                6 - (
                6 + 6 * self.alpha2 * np.sqrt(i) + 3 * self.alpha2 ** 2 * i + self.alpha2 ** 3 * i ** (3 / 2)
        ) * np.exp(-self.alpha2 * np.sqrt(i))
        ) / (self.alpha2 ** 4 * i ** 2)

        int_pcvt2 = self.int_pc0ptt + 4 * self.int_pc1ptt * (
                6 - (
                6 + 6 * self.alpha2 * np.sqrt(i) + 3 * self.alpha2 ** 2 * i + self.alpha2 ** 3 * i ** (3 / 2)
        ) * np.exp(-self.alpha2 * np.sqrt(i))
        ) / (self.alpha2 ** 4 * i ** 2)

        return nu * abs(z_c * z_a) * (a_j_p2 - a_j_p1) * self.h(i=i) \
               + self.T * self.int_pv0t2 \
               + 2 * nu_c * nu_a * tp.R * self.T ** 2 * (
                       self.m * (int_pbvt2 + 2 / self.T * int_pbvt) +
                       self.m ** 2 * nu_c * z_c * (int_pcvt2 + 2 / self.T * int_pcvt)
               )
