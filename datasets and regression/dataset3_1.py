import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

f = open("dataset3.txt")
lines = f.readlines()

x3 = []
y3 = []

for l in lines:
    s = l.split()

    x3.append(float(s[0]))
    y3.append(float(s[1]))

x3 = np.array(x3)
y3 = np.array(y3)

kb = 1.38e-23
c = 3.0e8
h = 6.6e-34

def plancklaw(f, t):
    f1 = (2*h*f*f*f)/(c*c)
    f2 = 1.0/(np.exp((h*f)/(kb*t)) - 1)
    return(f1*f2)

(t), _ = curve_fit(plancklaw, x3, y3, p0 = [5000]) 

print(f"Estimated T = {t} ")

yest3 = plancklaw(x3, t)  
plt.plot(x3, y3, x3, yest3)
plt.savefig("image3_1.png")