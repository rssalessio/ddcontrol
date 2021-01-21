#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Alessio Russo [alessior@kth.se]. All rights reserved.
# This file is part of ddcontrol.
# ddcontrol is free software: you can redistribute it and/or modify
# it under the terms of the MIT License. You should have received a copy of
# the MIT License along with ddcontrol.
# If not, see <https://opensource.org/licenses/MIT>.
#
# Code author: [Alessio Russo - alessior@kth.se]
# Last update: 21th January 2021, by alessior@kth.se
#

import cvxpy as cp
import numpy as np
from scipy.linalg import sqrtm

eps = 1e-15

def optimal_control(X: np.ndarray, U: np.ndarray, Qx: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Compute virtual reference signal by performing signal deconvolution
    Parameters
    ----------
    X : np.ndarray
        Matrix of dimensions (T, n), where n is the state dimensionality and T
        is the time horizon, containing state measurements
    U : np.ndarray
        Matrix of dimensions (T, m), where m is the input dimensionality and T
        is the time horizon, containing input measurements
    Qx : np.ndarray
        Square matrix of dimensions (n, n) containing the state weights
    R : np.ndarray
        Square matrix of dimensions (m, m) containing the input weights

    Returns
    -------
    K : np.ndarray
        The optimal control gain
    V : float
        The value of the problem
    """
    if U.shape[0] != X.shape[0]:
        raise AttributeError('X and U do not have the same batch dimension')
    T, n = X.shape
    _, m = U.shape

    if Qx.shape[0] != Qx.shape[1] or Qx.shape[0] != n:
        raise AttributeError('Q is not a square matrix, or the dimensionality is not n x n')

    if R.shape[0] != R.shape[1] or R.shape[0] != m:
        raise AttributeError('R is not a square matrix, or the dimensionality is not m x m')

    if not np.all(np.linalg.eigvals(Qx) > 0):
        raise ValueError('Matrix Qx is not positive definite')

    if not np.all(np.linalg.eigvals(R) >= 0):
        raise ValueError('Matrix R is not semi-positive definite')

    X_0_N, X_1_N, _U = X[:-1, :].T, X[1:, :].T, U[:-1, :].T
    _R = sqrtm(R)

    Q = cp.Variable((T - 1, n), 'full')
    P = cp.Variable((n, n), 'symmetric')
    W = cp.Variable((m, m), 'symmetric')

    cons1 = cp.bmat(
        [[P- np.identity(n), X_1_N @ Q],
        [(X_1_N @ Q).T, P]]) >> eps * np.identity(2 * n)
    cons2 = P == X_0_N @ Q

    cons3 = cp.bmat(
        [[W, _R @ _U @ Q],
        [(_R @ _U @ Q).T, P]]) >> eps * np.identity(m + n)

    optprob = cp.Problem(cp.Minimize(cp.trace(Qx @ P) + cp.trace(W)), constraints=[cons1, cons2, cons3])
    optprob.solve(verbose=False)
    K = _U @ Q.value @ np.linalg.inv(P.value)
    return K, optprob.value
