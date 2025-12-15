import numpy as np
import matplotlib.pyplot as plt


# Defining the first two-variable function and partial derivative
def twovarfunc(x, y):
    return x**4 - 16*x**3 + 96*x**2 - 256*x + y**2 - 4*y + 262

def dev_x(x, y):
    return 4*x**3 - 48*x**2 + 192*x - 256

def dev_y(x, y):
    return 2*y - 4

#  Setting initial values and learning rate
bestcost = 100000
startx, starty = 0, 0
lr = 0.01  

# Defining a function to perform gradient descent 
def twovariable(f, df_dx, df_dy, startx, starty, lr):
    x, y = startx, starty
    grad_x = df_dx(x, y)
    grad_y = df_dy(x, y)
    path_x = [x]
    path_y = [y]
    # Gradient descent loop
    while np.sqrt(grad_x**2 + grad_y**2) > 0.00000001:
        x -= lr * grad_x
        y -= lr * grad_y
        grad_x = df_dx(x, y)
        grad_y = df_dy(x, y)
        path_x.append(x)
        path_y.append(y)
    # Creating a 3D plot to visualize the optimization path
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x_range = np.linspace(-5, 5)
    y_range = np.linspace(-5, 5)
    X, Y = np.meshgrid(x_range, y_range)
    Z = f(X, Y)
    ax.plot_surface(X, Y, Z, cmap='coolwarm_r', alpha=0.8)
    ax.plot(path_x, path_y, f(np.array(path_x), np.array(path_y)), '-o', color='red')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x, y)')
    plt.title('Gradient Descent Path')
    plt.show()
    plt.savefig("figure3")
    plt.close()
    return (x, y), f(x, y)

# Calling the gradient descent function for the function
point, minima = twovariable(twovarfunc, dev_x, dev_y, startx, starty, lr)
print("The minima of the given function is", minima, "and it occurs at", point)



import numpy as np
import matplotlib.pyplot as plt

# Defining the first two-variable function and partial derivative
def twovarfunc(x,y):
    return np.exp(-(x - y)**2) * np.sin(y)

def dev_x(x,y):
    return -2 * np.exp(-(x - y)**2) * np.sin(y) * (x - y)

def dev_y(x,y):
    return np.exp(-(x - y)**2) * np.cos(y) + 2 * np.exp(-(x - y)**2) * np.sin(y) * (x - y)

#  Setting initial values and learning rate
bestcost = 100000
startx, starty = 0, 0
lr=0.1

# Defining a function to perform gradient descent 
def twovariable(f, df_dx, df_dy, startx,starty, lr):
    x, y = startx,starty
    grad_x = df_dx(x, y)
    grad_y = df_dy(x, y)
    path_x = [x]
    path_y = [y]
    # Gradient descent loop
    while np.sqrt(grad_x**2 + grad_y**2) > 0.000000000001:
        x -= lr * grad_x
        y -= lr * grad_y
        grad_x = df_dx(x, y)
        grad_y = df_dy(x, y)
        path_x.append(x)
        path_y.append(y)
    # Creating a 3D plot to visualize the optimization path
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x_range = np.linspace(-5,5)
    y_range = np.linspace(-5,5)
    X, Y = np.meshgrid(x_range, y_range)
    Z = f(X, Y)
    ax.plot_surface(X, Y, Z, cmap='coolwarm_r', alpha=0.8)
    ax.plot(path_x, path_y, f(np.array(path_x), np.array(path_y)), '-o', color='red')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x, y)')
    plt.title('Gradient Descent Path')
    plt.show()
    plt.savefig("figure4")
    plt.close()
    return (x, y), f(x, y)

# Calling the gradient descent function for the function
point,minima=twovariable(twovarfunc, dev_x, dev_y, startx,starty, lr)
print("The minima of the given function is",minima,"and it occurs at",point )
