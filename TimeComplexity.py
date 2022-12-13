import time
import matplotlib.pyplot as plt

def f(x):
    for i in range(x):
        print(i)


def g(x):
    for i in range(x):
        print(i)
    for j in range(x):
        print(j)

times = []
for n in range(100):
    start_time = time.time()
    g(n)
    times.append(time.time() - start_time)

xlist = range(100)
ylist = times.copy()

plt.plot(xlist, ylist, marker="o")
plt.show()