import random
import numpy
import matplotlib.pyplot as plt
data = []
lx = []

for i in numpy.arange(0, 10, 1.5):

    y = numpy.tan(i)
    x = random.uniform(0, 50)
    lx.append(i)
    z = y+x
    data.append(round(z, 1))

f = open("C:/Users/joyje/VSCODE/BTYSTE/Numerical/Research-Tests/Sequences.txt", mode="a")
f.write(f"{str(data)[1:-1]}, Tangent\n")
f.close()


plt.plot(lx, data, marker="o")
plt.xlabel("input")
plt.ylabel("f(x)")
plt.show()
