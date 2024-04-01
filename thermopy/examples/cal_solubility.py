import solubility

ts = [25, 50, 75, 80, 100, 125, 150, 175, 200, 225, 250, 250, 275, 300]
for t in ts:
    print(t,solubility.MgCl2_solubility(t=t))

"""
solubility/
    __init__.py
    MgCl2.py
"""
