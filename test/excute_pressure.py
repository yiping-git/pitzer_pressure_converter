import pandas as pd
import numpy as np
from converter.model_archer import PressureConverter
input_file_path = r"E:\work\data\fit\KCl-H2O-complete-v2-enthalpy.csv"
output_file_path = r"E:\work\data\fit\KCl-H2O-complete-v2-enthalpy.csv"

df_ori = pd.read_csv(input_file_path)
df = df_ori[df_ori['rm'] != 1]

def apparent_molar_enthalpy_difference(row):
    t_data =  row['T']
    p1_data = row['p']
    p2_data = row['p1_mf']
    m_data = row['mf']
    salt =    row['salt']

    solution_i = PressureConverter(
        t_data=t_data,
        p1_data=p1_data,
        p2_data=p2_data,
        m_data=m_data,
        salt = salt
    )

    dif_h = solution_i.apparent_molar_enghtalpy_difference()

    return dif_h

df['L_dif_f'] = df.apply(apparent_molar_enthalpy_difference, axis=1)

df.to_csv(output_file_path, index=False)