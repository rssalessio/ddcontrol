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

import numpy as np
import scipy.signal as scipysig
from ddcontrol.optimal_control import optimal_control

# Example 1
# ------------
# In this example we apply optimal control using Willem's lemma
# to an unstable system
#

# System definition
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

# Generate open-loop data using random normal noise
u = np.random.randn(T+1, m)
_, _, x = scipysig.dlsim(sys, u)

# Q,R matrices for optimal control
Q = np.identity(n)
R = np.identity(m)

# Compute optimal control law
K, V = optimal_control(x, u, Q, R)

# Compute closed-loop eigenvalues
eigs = np.linalg.eig(A + B @ K)[0]
print('Closed loop eigenvalues: {}'.format(eigs))