import random
import numpy
import matplotlib.pyplot as plt
data = []
lx = []
<<<<<<< HEAD
for i in numpy.arange(0, 10, 0.065):

    y = 2**i
    x = random.randint(1,9)
    lx.append(i)
    z = y+x
    data.append(z)
=======

for i in numpy.arange(0, 10, 1.5):

    y = numpy.tan(i)
    x = random.uniform(0, 50)
    lx.append(i)
    z = y+x
    data.append(round(z, 1))

f = open("C:/Users/joyje/VSCODE/BTYSTE/Numerical/Research-Tests/Sequences.txt", mode="a")
f.write(f"{str(data)[1:-1]}, Tangent\n")
f.close()


>>>>>>> fb09d5db5 (AudioGen for research done, number seq done)
plt.plot(lx, data, marker="o")
plt.xlabel("input")
plt.ylabel("f(x)")
plt.show()
<<<<<<< HEAD
print(data)
=======
>>>>>>> fb09d5db5 (AudioGen for research done, number seq done)
