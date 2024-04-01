from water import a_phi
from utilities import calculate_ionic_strength
import iapws
import numpy as np


def osmotic(T, m, beta0, beta1, Cphi):
    z_c = 2
    z_a = -1
    nu_c = 1
    nu_a = 2
    nu = nu_c + nu_a

    alpha = 2
    b = 1.2

    i = calculate_ionic_strength({
        'Mg+2': nu_c * m,
        'Cl-': nu_a * m
    })

    Bphi = beta0 + beta1 * np.exp(-alpha * i ** (1 / 2))

    P = iapws.iapws97._PSat_T(T)
    aphi = a_phi(T=T, P=P, method="Fernandez")

    return 1 - abs(z_c * z_a) * aphi * (i ** (1 / 2) / (1 + b * i ** (1 / 2))) + m * (2 * nu_c * nu_a / nu) * Bphi \
           + m ** 2 * (2 * (nu_c * nu_a) ** (3 / 2) / nu) * Cphi


def activity(T, m, beta0, beta1, Cphi):
    z_c = 2
    z_a = -1
    nu_c = 1
    nu_a = 2
    nu = nu_c + nu_a

    alpha = 2
    b = 1.2

    i = calculate_ionic_strength({
        'Mg+2': nu_c * m,
        'Cl-': nu_a * m
    })

    Bgamma = 2*beta0 + (2*beta1/alpha**2/i) * (1 - (1 + alpha * i **(1/2)- 1/2 * alpha**2 * i) * np.exp(-i**(1/2)))
    Cgamma = Cphi*3/2
    P = iapws.iapws97._PSat_T(T)
    aphi = a_phi(T=T, P=P, method="Fernandez")

    return abs(z_c * z_a) * aphi * (i ** (1 / 2) / (1 + b * i ** (1 / 2)) + 2/b * np.log(1 + b * i **(1/2)) ) + m * (2 * nu_c * nu_a / nu) * Bgamma \
           + m ** 2 * (2 * (nu_c * nu_a) ** (3 / 2) / nu) * Cgamma

#
# data = [
#     (323.16, 6.2, 0.3151, 2.2601, 0.0071),
#     (353.16, 6.8, 0.2879, 2.6610, 0.0067),
#     (373.16, 7.6, 0.3120, 1.9774, 0.0013),
#     (423.16, 10.1, 0.3122, 2.3640, -0.0023),
#     (473.16, 12.1, 0.3167, 2.5344, -0.0065),
#     (473.16, 12.1, 0.3218, 2.3870, -0.0069),
#     (473.16, 12.1, 0.3308, -1.3905, -0.0075),
# ]
#
# for i in data:
#     T,m,beta0,beta1,Cphi = i
#     print(T,m,beta0,beta1,Cphi, osmotic(T,m,beta0,beta1,Cphi))
data=[
(473.16 ,6.939,0.3167 ,2.5344, -0.0065 ),
(473.16 ,7.929,0.3167 ,2.5344, -0.0065 ),
(473.16 ,9.243,0.3167 ,2.5344, -0.0065 ),
(473.16 ,10.100,0.3167 ,2.5344, -0.0065 ),
(473.16 ,11.11,0.3167 ,2.5344, -0.0065 ),
(473.16 ,12.069,0.3167 ,2.5344, -0.0065 ),
]
for i in data:
    T, m, beta0, beta1, Cphi = i
    print(T,m,beta0,beta1,Cphi, activity(T,m,beta0,beta1,Cphi))