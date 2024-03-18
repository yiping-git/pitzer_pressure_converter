# -*- coding: utf-8 -*-
# Author: Yiping Liu
# Description:
# Version: 1.0
# Last Modified: Mar 14, 2024

from vapor_pressure.Shibue2004 import pressure_molality
import numpy as np

data = np.array([
(373.16,0.281305186),
(373.16,0.573202787),
(373.16,0.867162904),
(373.16,1.18994368),
(373.16,1.5088691),
(393.16,0.281305186),
(393.16,0.573202787),
(393.16,0.867162904),
(393.16,1.18994368),
(393.16,1.5088691),
])

for i in data:
    T, m = i
    print(pressure_molality(
        salt='KCl',
        m=m,
        T=T,
    ), 'MPa')
