# -*- coding: utf-8 -*-
# Author: Yiping Liu
# Description:
# Version: 1.0
# Last Modified: Mar 14, 2024

from solution.vapor_pressure.Shibue2003 import pressure_molality
import numpy as np

data = np.array([
(298.16	,0.99472),
(298.16	,1.26102),
(298.16	,1.50179),
(298.16	,1.78243),
(298.16	,1.965),
(298.16	,2.00338),
(298.16	,2.25174),
(298.16	,2.47333),
(298.16	,2.76245),
(298.16	,2.95143),
(298.16	,2.98355),
(298.16	,3.4871),
(298.16	,3.9447),
(298.16	,4.4473),
(298.16	,4.9303),
(298.16	,5.3859),
(298.16	,5.9518),
(318.16	,0.78747),
(318.16	,0.99472),
(318.16	,1.261),
(318.16	,1.5018),
(318.16	,1.7824),
(318.16	,2.0034),
(318.16	,2.2517),
(318.16	,2.4733),
(318.16	,2.7625),
(318.16	,2.9836),
(318.16	,3.4871),
(318.16	,3.9447),
(318.16	,4.4473),
(318.16	,4.9303),
(318.16	,5.3859),
(318.16	,5.9518),


])

for i in data:
    T,m = i
    print(pressure_molality(
        salt='NaCl',
        m=m,
        T=T,
    ), 'MPa')