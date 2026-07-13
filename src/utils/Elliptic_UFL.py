'''
Wrapper for scipy elliptic that can be called with UFL.

Definition is done through power series.
Call to functions is not fast but is precise with N = 100 to 10^-4 only close to 1
'''
import numpy as np

# Define the mesh and function spaces
def factorial(n):
    if n < 0:
        return 0
    elif n == 0 or n == 1:
        return 1
    else:
        fact = 1
        while(n > 1):
            fact *= n
            n -= 1
        return fact

def ellipk(m, N = 10):
    list = [(factorial(2*i)/(2**(2*i)*factorial(i)**2))**2 * m**(i) for i in range(0, N)]
    return 0.5*np.pi*np.sum(list)

def ellipe(m, N = 10):
    list = [(factorial(2*i)/(2**(2*i)*factorial(i)**2))**2 * m**(i)/(1-2*i) for i in range(0, N)]
    return 0.5*np.pi*np.sum(list)

def ellippi(n, m, N = 10):
    s = 0.0
    for k in range(N):
        for r in range(k+1):
            binom = factorial(k) / (factorial(r) * factorial(k - r))
            coef = (factorial(2*k) / (16**k * factorial(k)**2)) * binom**2
            s += coef * (n**r) * (m**(k - r)) / (1 - 2*k)
    return 0.5 * np.pi * s


