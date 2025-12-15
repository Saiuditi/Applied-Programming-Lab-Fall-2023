import numpy as np
import matplotlib.pyplot as plt

f = open("dataset1.txt")
lines = f.readlines()

x = []
y = []

#Creating a list of the x and y values, and plotting them

for l in lines:
    s = l.split()
    #print(s)
    
    x.append(float(s[0]))
    y.append(float(s[1]))


x = np.array(x)
y = np.array(y)

#print(x)
#print(y)

plt.plot(x, y)
plt.savefig

def stline(x, m, c):
    return m * x + c

#Performing Least Squares Fit Method
M = np.column_stack([x, np.ones(len(x))])

(m, c), _, _, _ = np.linalg.lstsq(M, y, rcond=None)
print(f"The estimated equation is {m} x + {c}")

plt.plot(x, y,label="noisy values")
#Defining fit_y as the best fit straight line for the given data
fit_y = stline(x, m, c) 

plt.plot(x, fit_y,label="straight line")
plt.savefig
#plotting errorbars
y_error = np.std(fit_y - y)
plt.plot(x, fit_y)
plt.legend(["straight line"])
plt.errorbar(x[::25], y[::25], y_error, fmt='ro',label="errorbar")
plt.legend()
plt.savefig("image1.png")
