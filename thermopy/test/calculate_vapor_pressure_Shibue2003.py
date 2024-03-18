# -*- coding: utf-8 -*-
# Author: Yiping Liu
# Description:
# Version: 1.0
# Last Modified: Mar 14, 2024

from vapor_pressure.Shibue2003 import pressure_molality
import numpy as np

data = np.array([
(298.16,4),
])

for i in data:
    T, m = i
    print(pressure_molality(
        salt='NaCl',
        m=m,
        T=T,
    ), 'MPa')
