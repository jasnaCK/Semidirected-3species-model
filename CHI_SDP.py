import numpy as np
import csv
import time
from numba import njit, prange
from multiprocessing import Pool, cpu_count

# CODE FOR OBTAINING DYNAMICAL SUSCEPTIBILITY VS P 

@njit
def initialize_column(L):

    # Fully active initial configuration

    return 2 * np.ones(L, dtype=np.int64)

@njit
def evolve_column(column, L, p, epsilon):

    # Updating 0,1,2 states in 3 species model

    new_column = np.zeros(L, dtype=np.int64)
    i = 0
    while i < L:
        if column[i] > 0:
            start = i
            while i < L and column[i] > 0:
                i += 1
            cluster = column[start:i]
            contains_2 = np.any(cluster == 2)
            if contains_2:
                rand_vals = np.random.rand(i - start)
                new_column[start:i] = (rand_vals <= p) * 2
            else:
                if np.random.rand() <= epsilon:
                    new_column[start:i] = 2
                else:
                    rand_vals = np.random.rand(i - start)
                    new_column[start:i] = (rand_vals <= p) * 1
        else:
            i += 1

    # Handle wrap-around cluster

    if column[0] > 0 and column[-1] > 0:
        s = 0
        while s < L and column[s] > 0:
            s += 1
        e = L - 1
        while e >= 0 and column[e] > 0:
            e -= 1
        c1 = column[0:s]
        c2 = column[e + 1:L]
        contains_2 = np.any(c1 == 2) or np.any(c2 == 2)
        if contains_2:
            new_column[0:s] = (np.random.rand(s) <= p) * 2
            new_column[e + 1:L] = (np.random.rand(L - e - 1) <= p) * 2
        else:
            if np.random.rand() <= epsilon:
                new_column[0:s] = 2
                new_column[e + 1:L] = 2
            else:
                new_column[0:s] = (np.random.rand(s) <= p) * 1
                new_column[e + 1:L] = (np.random.rand(L - e - 1) <= p) * 1

    # Randomly initialize zeros
    zeros = column == 0
    rand_vals = np.random.rand(L)
    new_column[zeros] = (rand_vals[zeros] <= p) * 1

    return new_column

@njit
def relabel_column_12(column, L):

    # Spreading 2s into neighboring 1s

    out = column.copy().astype(np.int64)
    i = 0
    while i < L:
        if column[i] > 0:
            start = i
            while i < L and column[i] > 0:
                i += 1
            out[start:i] = 2 if np.any(column[start:i] == 2) else 1
        else:
            i += 1

    # Handle wrap-around
    if column[0] > 0 and column[-1] > 0:
        s = 0
        while s < L and column[s] > 0:
            s += 1
        e = L - 1
        while e >= 0 and column[e] > 0:
            e -= 1
        wrap = np.concatenate((column[0:s], column[e+1:L]))
        val = 2 if np.any(wrap == 2) else 1
        out[0:s] = val
        out[e+1:L] = val

    out[column == 0] = 1
    return out

@njit
def simulate(L, steps, prob, epsilon):
    frac = np.zeros(len(prob))
    fracsq = np.zeros(len(prob))

    for j in range(len(prob)):
        f = 0.0
        fsq = 0.0
        p = prob[j]
        column = initialize_column(L)
        column1 = relabel_column_12(column, L)

        for step in range(steps):
            column = evolve_column(column, L, p, epsilon)
            column1 = relabel_column_12(column, L)
            if step >= 1000:  #initial 1000 steps discarded
                count = np.sum(column1 == 2)
                f += count / L
                fsq += (count * count) / (L * L)

        avg_steps = steps - 1000
        frac[j] = f / avg_steps  #storing avg rho(t)
        fracsq[j] = fsq / avg_steps #storing avg rho(t) squared

    return frac, fracsq


L =4096
steps = 10000
epsilon = 0.01
prob = np.arange(0.05, 0.9, 0.001)
h, hsq = simulate(L, steps, prob, epsilon)
chi = L * (hsq - h * h)  # obtain dynamic susceptibility chi for various p
        

