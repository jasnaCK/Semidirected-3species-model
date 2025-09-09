
import matplotlib.pyplot as plt
from numba import njit
import numpy as np
import csv
import time
import multiprocessing
from pylab import*

# CODE FOR OBTAINING AUTOCORRELATION MEASURE VS DELTA T

@njit
def initialize_column(L):

    # Fuly occupied initial condition
    x = 2*np.ones(L)
    
    return x

@njit
def evolve_column(column, L, p, epsilon):
    relabeled_column = np.zeros(L)

    # Update rules of 3-species model

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

    # Spreading 2s into neighboring 1s

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
def find_root(site, ptr):

    # root finding in union find

    root = site
    while ptr[root] >= 0:
        root = ptr[root]
    while site != root:
        parent = ptr[site]
        ptr[site] = root
        site = parent
    return root


@njit
def union_sites(a, b, ptr):

    #union in union find

    ra = find_root(a, ptr)
    rb = find_root(b, ptr)
    if ra != rb:
        if ptr[ra] < ptr[rb]:
            ptr[ra] += ptr[rb]
            ptr[rb] = ra
        else:
            ptr[rb] += ptr[ra]
            ptr[ra] = rb
    return ptr


@njit
def label_clusters(arr, ptr, step, L):

    # cluster labelling for active sites

    for i in range(L):
        if arr[step, i] == 1:  # 1 = active cluster marker
            site = (step * L )+ i

            # connect to time neighbor
            if step > 0 and arr[step - 1, i] == 1:
                ptr = union_sites(site, ((step - 1) * L) + i, ptr)

            # connect to spatial neighbors (periodic boundary)
            if arr[step, (i - 1) % L] == 1:
                ptr = union_sites(site, ((step * L) + (i - 1) % L),ptr)
            if arr[step, (i + 1) % L] == 1:
                ptr = union_sites(site, ((step * L) + (i + 1) % L),ptr)
    return ptr




def simulate(L, steps, p, epsilon, t0, tdelta, r):
    arr = np.zeros((steps, L))
    column = initialize_column(L)
    column1 = relabel_column_12(column, L)
    column1 = (column1 == 2).astype(np.int64)  
    arr[0] = column1

    ptr = np.full(L * steps, -1, dtype=np.int64)
    
    corr = np.zeros(len(tdelta))
    for step in range(1, steps):
        column = evolve_column(column, L, p, epsilon)
        column1 = relabel_column_12(column, L)
        column1 = (column1 == 2).astype(np.int64)  
        arr[step] = column1

        ptr = label_clusters(arr, ptr, step, L)

        if step > t0:
            d = step - t0
            if d <= len(tdelta):
                f=0
                
                for ri in r:
                    lw1=t0*L+ri
                    lw2=step*L+ri
                    label1=find_root(lw1,ptr)
                    label2=find_root(lw2,ptr)
                    if label1==label2 and label1!=-1:
                        f+=1

                corr[d-1]=f  # auto correlation values  stored for each delta t
                
    

    return corr

#example parameters

L=4096
steps=10000
p=0.579
epsilon=1
tdelta=np.arange(1,1000,1)
t0=1000
r=np.arange(50,4000,50)


h= simulate(L,steps,p,epsilon,t0,tdelta,r)
       
# h gives autocorrelation as a function of tdelta 