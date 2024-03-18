
import numpy as np
from vapor_pressure.property_pressure_convert import osmotic_coefficient, water_activity

data = [
(4.897	,	348.16	,0.031321466),
(6.006	,	348.16	,0.02940163),
(6.098	,	348.16	,0.029229644),
(6.46	,	348.16	,0.028679024),
]
for i in data:
    m, T, P = i
    print(osmotic_coefficient(
        T=T,
        m=m,
        P=P,
        nu=2
    ))
# for i in data:
#     m, T, P = i
#     print(np.exp(water_activity(
#         T=T,
#         P=P,
#     )))
