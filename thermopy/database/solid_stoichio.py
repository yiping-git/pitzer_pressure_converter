# -*- coding: utf-8 -*-
# Author: Yiping Liu
# Description:
# Version: 1.0
# Last Modified: Mar 14, 2024

import pandas as pd

# Create an empty dataframe
df = pd.DataFrame(columns=['formula', 'components', 'coefficients'])


# Define a function to add data for a new solid
def add_solid(df, formula, components, coefficients):
    df.loc[len(df)] = [formula, components, coefficients]


# Add data for existing solids
add_solid(df, 'H2O(s)', {'cations': [], 'anions': [], 'neutrals': ['H2O']}, {'H2O': 1})
add_solid(df, 'NaCl', {'cations': ['Na+'], 'anions': ['Cl-'], 'neutrals': []}, {'Na+': 1, 'Cl-': 1})
add_solid(df, 'NaCl-2H2O', {'cations': ['Na+'], 'anions': ['Cl-'], 'neutrals': ['H2O']}, {'Na+': 1, 'Cl-': 1, 'H2O': 2})
add_solid(df, 'KCl', {'cations': ['K+'], 'anions': ['Cl-'], 'neutrals': []}, {'K+': 1, 'Cl-': 1})
add_solid(df, 'CaCl2-1/3H2O', {'cations': ['Ca+2'], 'anions': ['Cl-'], 'neutrals': ['H2O']},
          {'Ca+2': 1, 'Cl-': 2, 'H2O': 1 / 3})
add_solid(df, 'CaCl2-H2O', {'cations': ['Ca+2'], 'anions': ['Cl-'], 'neutrals': ['H2O']},
          {'Ca+2': 1, 'Cl-': 2, 'H2O': 1})
add_solid(df, 'CaCl2-2H2O', {'cations': ['Ca+2'], 'anions': ['Cl-'], 'neutrals': ['H2O']},
          {'Ca+2': 1, 'Cl-': 2, 'H2O': 2})
add_solid(df, 'CaCl2-4H2O(alpha)', {'cations': ['Ca+2'], 'anions': ['Cl-'], 'neutrals': ['H2O']},
          {'Ca+2': 1, 'Cl-': 2, 'H2O': 4})
add_solid(df, 'CaCl2-4H2O(beta)', {'cations': ['Ca+2'], 'anions': ['Cl-'], 'neutrals': ['H2O']},
          {'Ca+2': 1, 'Cl-': 2, 'H2O': 4})
add_solid(df, 'CaCl2-4H2O(gamma)', {'cations': ['Ca+2'], 'anions': ['Cl-'], 'neutrals': ['H2O']},
          {'Ca+2': 1, 'Cl-': 2, 'H2O': 4})
add_solid(df, 'CaCl2-6H2O(gamma)', {'cations': ['Ca+2'], 'anions': ['Cl-'], 'neutrals': ['H2O']},
          {'Ca+2': 1, 'Cl-': 2, 'H2O': 6})

add_solid(df, 'LiCl0', {'cations': [], 'anions': [], 'neutrals': ['LiCl0']}, {'LiCl0': 1})
add_solid(df, 'LiCl', {'cations': ['Li+'], 'anions': ['Cl-'], 'neutrals': []}, {'Li+': 1, 'Cl-': 1})
add_solid(df, 'LiCl-H2O', {'cations': ['Li+'], 'anions': ['Cl-'], 'neutrals': ['H2O']}, {'Li+': 1, 'Cl-': 1, 'H2O': 1})
add_solid(df, 'LiCl-2H2O', {'cations': ['Li+'], 'anions': ['Cl-'], 'neutrals': ['H2O']}, {'Li+': 1, 'Cl-': 1, 'H2O': 2})
add_solid(df, 'LiCl-3H2O', {'cations': ['Li+'], 'anions': ['Cl-'], 'neutrals': ['H2O']}, {'Li+': 1, 'Cl-': 1, 'H2O': 3})
add_solid(df, 'LiCl-5H2O', {'cations': ['Li+'], 'anions': ['Cl-'], 'neutrals': ['H2O']}, {'Li+': 1, 'Cl-': 1, 'H2O': 5})

add_solid(df, 'MgCl2', {'cations': ['Mg+2'], 'anions': ['Cl-'], 'neutrals': []}, {'Mg+2': 1, 'Cl-': 2})
add_solid(df, 'MgCl2-2H2O', {'cations': ['Mg+2'], 'anions': ['Cl-'], 'neutrals': ['H2O']},
          {'Mg+2': 1, 'Cl-': 2, 'H2O': 2})
add_solid(df, 'MgCl2-4H2O', {'cations': ['Mg+2'], 'anions': ['Cl-'], 'neutrals': ['H2O']},
          {'Mg+2': 1, 'Cl-': 2, 'H2O': 4})
add_solid(df, 'MgCl2-6H2O', {'cations': ['Mg+2'], 'anions': ['Cl-'], 'neutrals': ['H2O']},
          {'Mg+2': 1, 'Cl-': 2, 'H2O': 6})
add_solid(df, 'MgCl2-8H2O', {'cations': ['Mg+2'], 'anions': ['Cl-'], 'neutrals': ['H2O']},
          {'Mg+2': 1, 'Cl-': 2, 'H2O': 8})
add_solid(df, 'MgCl2-12H2O', {'cations': ['Mg+2'], 'anions': ['Cl-'], 'neutrals': ['H2O']},
          {'Mg+2': 1, 'Cl-': 2, 'H2O': 12})
add_solid(df, 'KCl-MgCl2-6H2O', {'cations': ['K+', 'Mg+2'], 'anions': ['Cl-'], 'neutrals': ['H2O']},
          {'K+': 1, 'Mg+2': 1, 'Cl-': 3, 'H2O': 6})
add_solid(df, 'CaCl2-Mg2Cl4-12H2O', {'cations': ['Ca+2', 'Mg+2'], 'anions': ['Cl-'], 'neutrals': ['H2O']},
          {'Ca+2': 1, 'Mg+2': 2, 'Cl-': 6, 'H2O': 12})
add_solid(df, 'CaCl2-Mg2Cl4-12H2O', {'cations': ['Ca+2', 'Mg+2'], 'anions': ['Cl-'], 'neutrals': ['H2O']},
          {'Ca+2': 1, 'Mg+2': 2, 'Cl-': 6, 'H2O': 12})

add_solid(df, 'FeCl2-2H2O', {'cations': ['Fe+2'], 'anions': ['Cl-'], 'neutrals': ['H2O']},
          {'Mg+2': 1, 'Cl-': 2, 'H2O': 2})
add_solid(df, 'FeCl2-4H2O', {'cations': ['Fe+2'], 'anions': ['Cl-'], 'neutrals': ['H2O']},
          {'Mg+2': 1, 'Cl-': 2, 'H2O': 4})
add_solid(df, 'FeCl2-6H2O', {'cations': ['Fe+2'], 'anions': ['Cl-'], 'neutrals': ['H2O']},
          {'Mg+2': 1, 'Cl-': 2, 'H2O': 6})
add_solid(df, '2KCl-FeCl2-2H2O', {'cations': ['K+', 'Fe+2'], 'anions': ['Cl-'], 'neutrals': ['H2O']},
          {'K+': 2, 'Cl-': 4, 'H2O': 2})

add_solid(df, 'Na2SO4', {'cations': ['Na+'], 'anions': ['SO4-2'], 'neutrals': []}, {'Na+': 2, 'SO4-2': 1})
add_solid(df, 'Na2SO4-10H2O', {'cations': ['Na+'], 'anions': ['SO4-2'], 'neutrals': ['H2O']},
          {'Na+': 2, 'SO4-2': 1, 'H2O': 10})
add_solid(df, 'MgSO4-6H2O', {'cations': ['Mg+2'], 'anions': ['SO4-2'], 'neutrals': ['H2O']},
          {'Mg+2': 1, 'SO4-2': 1, 'H2O': 6})
add_solid(df, 'MgSO4-7H2O', {'cations': ['Mg+2'], 'anions': ['SO4-2'], 'neutrals': ['H2O']},
          {'Mg+2': 1, 'SO4-2': 1, 'H2O': 7})
add_solid(df, 'K2SO4', {'cations': ['K+'], 'anions': ['SO4-2'], 'neutrals': []}, {'K+': 2, 'SO4-2': 1})
add_solid(df, 'K2SO4-MgSO4-6H2O', {'cations': ['K+', 'Mg+2'], 'anions': ['SO4-2'], 'neutrals': ['H2O']},
          {'K+': 2, 'Mg+2': 1, 'SO4-2': 2, 'H2O': 6})


class DbStoichio:
    def __init__(self, compound):
        """
        A class that help retrieve ions and their stoichiometric coefficients of a compund.
        :param compound: chemical formula of a compound. e.g., 'NaCl'. Hydrates should be written like 'NaCl-2H2O'.
        """
        self.compound = compound
        self.data = self.get_data()
        print(type(self.data))

    def get_data(self):
        """
        Get data from the database.
        :return: a instance of panda Series contains data describing the compound.
         """
        compound_data = df[df['formula'] == self.compound]
        if not compound_data.empty:
            return compound_data.iloc[0]
        else:
            print("Compound '{}' not found in the dataframe.".format(self.compound))
            return None

    @property
    def cations(self):
        """
        :return: a list of cations in the compound.
        """
        return self.data['components']['cations']

    @property
    def anions(self):
        """
        :return: a list of anions in the compound.
        """
        return self.data['components']['anions']

    @property
    def neutrals(self):
        """
        :return: a list of neutrals in the compound.
        """
        return self.data['components']['neutrals']

    @property
    def coefficients(self):
        """
        :return:  A dictionary contains all the ions as keys and their stoichiometric coefficients as values. e.g.,
        {'Na+': 1, 'Cl-': 1}
        """
        return self.data['coefficients']
