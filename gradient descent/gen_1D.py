import numpy as np
from numpy import cos, sin, pi, exp 
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt

# Initializing starting point and learning rate
start=-1
lr=0.01

def func(x):
    return x ** 2 + 3 * x + 8
def deriv(x):
    return 2*x + 3
fig, ax = plt.subplots()
xbase = np.linspace(-2, 2, 100)
ybase = func(xbase)
ax.plot(xbase, ybase)
xall, yall = [], []
lnall,  = ax.plot([], [], 'ro')
# Implementing 1D gradient descent
def gradient_descent_1d(func, deriv, start, lr):
    x = start
    grad = deriv(x)
    path_x = [x]
    path_y = [func(x)]
    while abs(grad) > 1e-5:
        x -= lr * grad
        grad = deriv(x)
        path_x.append(x)
        path_y.append(func(x))
    # Creating a figure and plot the function
    plt.plot(path_x, path_y, '-o')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Gradient Descent Path')
    plt.show()
    plt.savefig("figure1")
    return x, func(x)
# Calling gradient_descent_1d to find the minima of the given function
x,y=gradient_descent_1d(func, deriv, start, lr)
print("The minima of the given function is",y,"at",x)


import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from numpy import cos, sin, pi, exp 


def func(x):
    return np.cos(x)**4 - np.sin(x)**3 - 4*np.sin(x)**2 + np.cos(x) + 1

def deriv(x):
    return -4 * (np.cos(x)**3) * np.sin(x) - 3 * (np.sin(x)**2) * np.cos(x) - 8*np.sin(x)*cos(x) - np.sin(x)

# Initializing starting point and learning rate
bestcost = 100000
start=1
lr=0.01
fig, ax = plt.subplots()
xbase = np.linspace(-5, 5, 100)
ybase = func(xbase)
ax.plot(xbase, ybase)
xall, yall = [], []
lnall,  = ax.plot([], [], 'ro')
# Implementing 1D gradient descent
def gradient_descent_1d(func, deriv, start, lr):
    x = start
    grad = deriv(x)
    path_x = [x]
    path_y = [func(x)]
    while abs(grad) > 1e-5:
        x -= lr * grad
        grad = deriv(x)
        path_x.append(x)
        path_y.append(func(x))
    # Creating a figure and plot the function
    plt.plot(path_x, path_y, '-o')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Gradient Descent Path')
    plt.show()
    plt.savefig("figure2")
    return x, func(x)
# Calling gradient_descent_1d to find the minima of the given function
x,y=gradient_descent_1d(func, deriv, start, lr)
print("The minima of the given function is",y,"at",x)


