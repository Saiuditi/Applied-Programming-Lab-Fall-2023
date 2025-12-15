import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

f = open("dataset2.txt", 'r')
lines = f.readlines()
f.close()
x2 = []
y2 = []

#Creating a list of the x and y values, and plotting them

for l in lines:
    s = l.split()
    x2.append(float(s[0]))
    y2.append(float(s[1]))
    
x2 = np.array(x2)
y2 = np.array(y2)

def func(x2,f1, p1, p2, p3):
    return p1 * np.sin(f1*x2) + p2 * np.sin(3*f1*x2) + p3 * np.sin(5*f1*x2)

(f1, p1, p2, p3), pcov = curve_fit(func, x2, y2, p0 = [2, 1, 1, 1])  
print (f"Estimated function: {p1} * np.sin({f1}*x2) + {p2} * np.sin(3*{f1}*x2) + {p3} * np.sin(5*{f1}*x2)")

y_estimate = func(x2,f1, p1, p2, p3)
plt.plot(x2, y2, x2, y_estimate)
plt.savefig("image2_1.png")

def f1(x2, p):
    return p[0] * np.sin(2.5*x2) + p[1] * np.sin(7.5*x2) + p[2] * np.sin(12.5*x2)


# Building a model to obtain the coefficient values
M = np.column_stack([np.sin(2.5*x2), np.sin(7.5*x2), np.sin(12.5*x2)])
# Use the lstsq function to solve for the array p
p, _, _, _ = np.linalg.lstsq(M, y2, rcond=None)
print(f"The estimated parameters are: {p[0]} sin(1/3*x2) + {p[1]} sin(1*x2) + {p[2]} sin(5/3*x2)")

yest2 = f1(x2, p) 
plt.plot(x2, y2, x2,yest2)
plt.savefig("image2_2.png")
