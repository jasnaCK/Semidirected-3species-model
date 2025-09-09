import numpy as np
import csv
import time
import multiprocessing
from numba import njit
from pylab import*
from scipy.ndimage import label,sum

# CODE FOR OBTAINING RHO(T) VS T PLOT

@njit
def initialize_column(L):
    x = 2*np.ones(L)

    #Fully active initial configuration
    
    return x

@njit
def evolve_column(column, L, p, epsilon):

    # Process of updating 0,1,2 states

    relabeled_column = np.zeros(L)
   
    i = 0
    while i < L:
        if column[i] > 0:
            start = i
            while i < L and column[i] > 0:
                i += 1
            cluster = column[start:i]
            
            # Check if cluster contains a 2

            if 2 in cluster:
                
                rand_vals = np.random.uniform(0, 1, len(cluster))
                relabeled_column[start:i] = np.where(rand_vals <= p, 2, 0)
            elif 2 not in cluster:
                rand_vals = np.random.uniform(0, 1)
                if rand_vals<=epsilon:
                     relabeled_column[start:i]=2
                else:
                     rand_vals1 = np.random.uniform(0, 1, len(cluster))
                     relabeled_column[start:i] =np.where(rand_vals1 <= p, 1, 0)
                                                    
        else:
            i += 1

    # Handle periodic boundary clusters

    if column[0] > 0 and column[L - 1] > 0:
        start = 0
        while start < L and column[start] > 0:
            start += 1
        cluster1 = column[0:start]

        end = L - 1
        while end >= 0 and column[end] > 0:
            end -= 1
        cluster2 = column[end + 1:L]

        if 2 in cluster1 or 2 in cluster2:
            rand_vals1 = np.random.uniform(0, 1, len(cluster1))
           
            rand_vals2 = np.random.uniform(0, 1, len(cluster2))
            relabeled_column[0:start] = np.where(rand_vals1 <= p, 2, 0)
            relabeled_column[end + 1:L] = np.where(rand_vals2 <= p, 2, 0)
        elif (2 not in cluster1) and (2 not in cluster2):
            rand_vals=np.random.uniform(0, 1)
            if rand_vals<=epsilon:
                relabeled_column[0:start]=2
                relabeled_column[end+1:L]=2
            else:
                rand_vals1 = np.random.uniform(0, 1, len(cluster1))
                rand_vals2 = np.random.uniform(0, 1, len(cluster2))
                relabeled_column[0:start] = np.where(rand_vals1 <= p, 1, 0)
                relabeled_column[end + 1:L] = np.where(rand_vals2 <= p, 1, 0)
                                                  

    # Randomly reinitialize zeros to 1 or 0

    zero_indices = column == 0
    
    rand_vals = np.random.uniform(0, 1, L)
    relabeled_column[zero_indices] = np.where(rand_vals[zero_indices] <= p, 1, 0)

    return relabeled_column

@njit
def relabel_column_12(column, L):

    # Spreading of 2s into neighboring 1s, a kind of relabelling

    relabeled_column = np.copy(column)
    i = 0

    while i < L:
        if column[i] > 0:
            start = i
            while i < L and column[i] > 0:
                i += 1
            cluster = column[start:i]
            relabeled_column[start:i] = 2 if 2 in cluster else 1
        else:
            i += 1

    # Handle periodic boundary clusters

    if column[0] > 0 and column[L - 1] > 0:
        start = 0
        while start < L and column[start] > 0:
            start += 1
        cluster1 = column[0:start]

        end = L - 1
        while end >= 0 and column[end] > 0:
            end -= 1
        cluster2 = column[end + 1:L]

        if 2 in cluster1 or 2 in cluster2:
            relabeled_column[0:start] = 2
            relabeled_column[end + 1:L] = 2
        elif (2 not in cluster1) and (2 not in cluster2):
            relabeled_column[0:start] = 1
            relabeled_column[end + 1:L] = 1
    relabeled_column[column==0]=1

    return relabeled_column


@njit
def simulate(L, steps, p, epsilon):
    frac=np.zeros(steps)
    column = initialize_column(L)
    column1 = relabel_column_12(column, L)
    frac[0]=L
    
    # evolution over time T = steps

    for step in range(1, steps):
        column = evolve_column(column, L, p, epsilon)
        column1 = relabel_column_12(column, L)
        x=len(np.argwhere(column1==2))
        frac[step]+=x  #count number of active sites in each timesteps and store in frac
              
    
    return frac

# Parameters

L =4096
steps = 100000
p=0.632
epsilon =0


h= simulate(L,steps,p,epsilon)

#h gives density of active sites as a function of time t in (0,100000)        