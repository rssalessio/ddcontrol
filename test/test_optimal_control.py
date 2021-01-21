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

from unittest import TestCase
import numpy as np
import scipy.signal as scipysig
from scipy.linalg import solve_discrete_are as dare
from ddcontrol.optimal_control import optimal_control


class TestOptimalControl(TestCase):
    def test_unstable_system(self):
        T = 15
        Ts = 1e-1
        A = [[ 1.178, 0.001,  0.511, -0.403],
             [-0.051, 0.661, -0.011,  0.061],
             [ 0.076, 0.335,  0.560,  0.382],
             [ 0.,    0.335,  0.089,  0.849]]

        B = [[0.004, -0.087],
             [0.467,  0.001],
             [0.213, -0.235],
             [0.213, -0.016]]

        A = np.array(A)
        B = np.array(B)
        n, m = B.shape

        sys = scipysig.StateSpace(A, B, np.identity(n), np.zeros((n, m)), dt=Ts)
        u = np.random.randn(T+1, m)
        _, _, x = scipysig.dlsim(sys, u)
        Qs = [np.identity(n), 0 * np.identity(n)]
        Rs = [np.identity(m), 0 * np.identity(m)]

        for Q in Qs:
            for R in Rs:
                if np.all(Q == Qs[1]):
                    with self.assertRaises(ValueError):
                        K, V = optimal_control(x, u, Q, R)
                else:
                    # Solve optimal control problem with Riccati equation
                    P = dare(A, B, Q, R)
                    trueK = -np.linalg.inv(B.T @ P @ B + R) @ B.T @ P @ A

                    # Solve using Willems' lemma
                    K, V = optimal_control(x, u, Q, R)

                    # Compare results
                    self.assertTrue(np.linalg.norm(K - trueK) < 1e-4)
