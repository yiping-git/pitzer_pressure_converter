# pitzer_pressure_converter

## Introduction

Package for converting thermodynamic properties like osmotic coefficient, activity coefficient, apparent molal enthalpy
and apparent molal heat capacity from one pressure (*P*1) to another pressure (*P*2) at constant temperature and composition.


## Models

### converter/model_general.py

Implemented the general pressure converter model of Rogers and Pitzer(1982).

Verifications of properties:

- [ ] osmotic coefficient
- [ ] avtivity coefficient
- [ ] apparent molal enthalpy
- [ ] apparent molar heat capacity


### converter/model_archer.py

Implemented the general pressure converter model of Archer(1982).

Verifications of properties:

- [ ] osmotic coefficient
- [ ] avtivity coefficient
- [ ] apparent molal enthalpy
- [ ] apparent molar heat capacity

## References
>Archer, D. G. (1992). Thermodynamic Properties of the NaCl+H2O System. II. Thermodynamic Properties of NaCl(aq), NaCl⋅2H2(cr), and Phase Equilibria. Journal of Physical and Chemical Reference Data, 21, 793-829. 
> 
> Rogers, P. S. Z., & Pitzer, K. S. (1982). Volumetric properties of aqueous sodium chloride solutions. Journal of Physical and Chemical Reference Data, 11(1), 15-81.