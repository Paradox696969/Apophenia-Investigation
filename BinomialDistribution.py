from numpy import random
from math import comb

n = 25*6
x = int(0.33*n)
p = 1/6
q = 5/6

combination = comb(n, x)
prob = combination * p**x * q**(n-x)

print(prob)