import random
import numpy
import matplotlib.pyplot as plt
data = []
lx = []
for i in numpy.arange(0, 10, 0.065):

    y = 2**i
    x = random.randint(1,9)
    lx.append(i)
    z = y+x
    data.append(z)
plt.plot(lx, data, marker="o")
plt.xlabel("input")
plt.ylabel("f(x)")
plt.show("")
print(data)