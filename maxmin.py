import math
from scipy.optimize import minimize


def f(param):
    u, v = param
    z = 0
    for i in range(0, 3):
        z += 1 / ((u - x[i]) ** 2 + (v - y[i]) ** 2)
    return z


N = 4
z1 = (1, 0)
z2 = (1, 1)
maximin = 0
curmin = 0
bounds = ((-1, 1), (-1, 1))
x = [None] * 4
y = [None] * 4

for i in range(0, N - 1):
    x[0] = -1 + (2 * i + 1.0) / N
    for j in range(0, N - 1):
        y[0] = -1 + (2 * j + 1.0) / N
        for k in range(0, N - 1):
            x[1] = -1 + (2 * k + 1.0) / N
            for l in range(0, N - 1):
                y[1] = -1 + (2 * j + 1.0) / N
                for m in range(0, N - 1):
                    x[2] = -1 + (2 * m + 1.0) / N
                    for n in range(0, N - 1):
                        y[2] = -1 + (2 * n + 1.0) / N
                        for p in range(0, N - 1):
                            x[3] = -1 + (2 * p + 1.0) / N
                            for q in range(0, N - 1):
                                y[3] = -1 + (2 * q + 1.0) / N
                                res1 = minimize(f, z1, method='SLSQP',
                                                options={'disp': True}, bounds=bounds, tol=1e-6)
                                res2 = minimize(f, z2, method='SLSQP',
                                                options={'disp': True}, bounds=bounds, tol=1e-6)
                                curmin = min(res1.fun, res2.fun)
                                if curmin == res1.fun:
                                    z=res1.x
                                else:
                                    z=res2.x
                                if curmin > maximin:
                                    c=z
                                    a=x[0],x[1],x[2],x[3]
                                    b=y[0],y[1],y[2],y[3]
                                    maximin = curmin

print(maximin, a, b, c)
