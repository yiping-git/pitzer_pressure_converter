# -*- coding: utf-8 -*-
# Author: Yiping Liu
# Description:
# Version: 1.0
# Last Modified: Mar 15, 2024

import sympy as sp

def lagrange(Lx, Ly):
    """
    Lagrange interpolation method.
    :param Lx: x value list, e.g., [1,2,3,...].
    :param Ly: y value list, e.g., [2,4,9,...].
    :return: the expression of y = f(x).
    """
    X = sp.symbols('X')
    if len(Lx) != len(Ly):
        print("ERROR")
        return 1
    y = 0
    for k in range(len(Lx)):
        t = 1
        for j in range(len(Lx)):
            if j != k:
                t = t * ((X - Lx[j]) / (Lx[k] - Lx[j]))
        y += t * Ly[k]
    return y


def partial_lagrange(x, Lx, Ly):
    """
    First derivative of the Lagrange interpolation method [dy/dx].
    :param x: x value at where we want to calculate [dy/dx].
    :param Lx: x value list, e.g., [1,2,3,...].
    :param Ly: y value list, e.g., [2,4,9,...].
    :return: the expression of y = f(x).
    """
    X = sp.symbols('X')
    y = lagrange(Lx, Ly)
    dy = sp.diff(y, X).subs(X, x)
    return dy


def partial_lagrange_2ed(x, Lx, Ly):
    """
    First derivative of the Lagrange interpolation method [dy/dx].
    :param x: x value at where we want to calculate [dy/dx].
    :param Lx: x value list, e.g., [1,2,3,...].
    :param Ly: y value list, e.g., [2,4,9,...].
    :return: the expression of y = f(x).
    """
    X = sp.symbols('X')
    y = lagrange(Lx, Ly)
    dy = sp.diff(y, X)
    dyy = sp.diff(dy, X).subs(X, x)
    return dyy
