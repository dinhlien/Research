from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import scipy.optimize as opt
import numpy as np


s = 2
# np.random.seed(111)
x = np.random.uniform(-1,1,4)
y = np.random.uniform(-1,1,4)


def f(param):
   u, v = param
   z = ((u - x[0])**2 + (v - y[0])**2)**(-s*0.5) \
       + ((u - x[1])**2+(v - y[1])**2)**(-s*0.5) \
       + ((u - x[2])**2+(v - y[2])**2)**(-s*0.5) \
       + ((u - x[3])**2+(v - y[3])**2)**(-s*0.5)
   return z


bounds = opt.Bounds((-1.0, 1.0), (-1.0, 1.0))
x_0 = [0.9, 0.8]
res = opt.minimize(f, x_0, method='SLSQP', options={'disp': True}, bounds=bounds, tol=1e-6)
print(res.x)


def f(u ,v):
    z = ((u - x[0]) ** 2 + (v - y[0]) ** 2) ** (-s * 0.5) \
        + ((u - x[1]) ** 2 + (v - y[1]) ** 2) ** (-s * 0.5) \
        + ((u - x[2]) ** 2 + (v - y[2]) ** 2) ** (-s * 0.5) \
        + ((u - x[3]) ** 2 + (v - y[3]) ** 2) ** (-s * 0.5)
    return z


U = np.linspace(-1, 1)
V = np.linspace(-1, 1)
X, Y = np.meshgrid(U, V)
Z = f(X, Y)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
ax.set_zlim(0, 5*f(1,1))
ax.set_title('Surface plot')
ax.set_xlabel('u')
ax.set_ylabel('v')
plt.show()

