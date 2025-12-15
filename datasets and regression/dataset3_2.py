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

def plancklaw(f, h, t, kb, c):
    f1 = (2*h*f*f*f)/(c*c)
    f2 = 1.0/(np.exp((h*f)/(kb*t)) - 1)
    return(f1*f2)

(h, t, kb, c), _ = curve_fit(plancklaw, x3, y3, p0 = [1e-33,  2000, 1e-23, 1e8])    

yest3 = plancklaw(x3, h, t, kb, c)   
plt.plot(x3, y3, x3, yest3)
plt.savefig("image3_2.png") 

print(f"Estimated T = {t}, Planck's constant = {h}, Boltzmann constant = {kb}, Speed of light = {c}")
