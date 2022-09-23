#!/bin/env python
import timeit
import numpy as np
import math
import matplotlib.pyplot as plt
from itertools import combinations
from multiprocessing import Pool
from functools import wraps

def run_timeit (func):
    @wraps(func)
    def wrapper_timeit (s):
        return timeit.timeit(lambda: func(s), number = 100)
    return wrapper_timeit

@run_timeit
def factors_all (n):
    '''
    This is a brute force algorithm that returns a list of all factors of n
    including 1 and n. We cycle through all numbers from 1 to n and see
    if they divide n evenly.
    '''
    a = []
    for i in range(1,n+1):
        if n%i == 0: a.append(i)
    return a

@run_timeit
def factors_trial_math (n):
    '''
    This trial division factorization algorithm is taken from Wikipedia:
    https://en.wikipedia.org/wiki/Trial_division
    Additional lines use the factorization to get all factors of n including
    1 and n.
    '''
    a = []
    while n % 2 == 0:
        a.append(2)
        n //= 2
    f = 3
    while f * f <= n:
        if n % f == 0:
            a.append(f)
            n //= f
        else:
            f += 2
    if n != 1: a.append(n)

    b = []
    for i in range(1, len(a)+1):
        b += [math.prod(x) for x in combinations(a,i)]
    b = [*set(b), ]
    b.append(1)
    b.sort()
    return b

@run_timeit
def factors_trial_np (n):
    '''
    This trial division factorization algorithm is taken from Wikipedia:
    https://en.wikipedia.org/wiki/Trial_division
    Additional lines use the factorization to get all factors of n including
    1 and n.
    '''
    a = []
    while n % 2 == 0:
        a.append(2)
        n //= 2
    f = 3
    while f * f <= n:
        if n % f == 0:
            a.append(f)
            n //= f
        else:
            f += 2
    if n != 1: a.append(n)

    b = []
    for i in range(1, len(a)+1):
        b += [np.prod(x) for x in combinations(a,i)]
    b = [*set(b), ]
    b.append(1)
    b.sort()
    return b

if __name__ == '__main__':
    # Set up sample space
    n = 15
    x = np.logspace(1, n, num=50, base=2, dtype=np.uint32)

    # Run both algorithms in a process pool and time
    with Pool() as p:
        y_all = p.map (factors_all, x)
    with Pool() as p:
        y_trial_np = p.map (factors_trial_np, x)
    with Pool() as p:
        y_trial_math = p.map (factors_trial_math, x)

    # Plot results
    fig, ax = plt.subplots (layout='tight')
    ax.loglog (x, y_all, marker = '.', label='brute force division')
    ax.loglog (x, y_trial_np, marker = '.', label='trial division (numpy)')
    ax.loglog (x, y_trial_math, marker = '.', label='trial division (math)')
    ax.set_xlabel ('number to factor')
    ax.set_ylabel ('average loop time [s]')
    ax.set_title ('Benchmarking factorization algorithms')
    ax.legend ()
    ax.grid(True)
    plt.show ()
