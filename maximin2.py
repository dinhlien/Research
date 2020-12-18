import math
from scipy.optimize import minimize


def f(param):
   u, v, i, j, k, l, m, n, p, q = param
   z = 1/((u - i)**2 + (v - j)**2)
   + 1/((u - k)**2 + (v - l)**2)
   + 1 / ((u - m) ** 2 + (v - n) ** 2)
   + 1 / ((u - p) ** 2 + (v - q) ** 2)
   return z


N=6
# z0 = (0, 0)
z1 = (1, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
z2 = (1, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
maximin=0
curmin=0
bounds = ((-1, 1), (-1, 1), (-1+1.0/N, -1+(2*(N-1)+ 1.0)/N), (-1+1.0/N, -1+(2*(N-1)+ 1.0)/N), (-1+1.0/N, -1+(2*(N-1)+ 1.0)/N), (-1+1.0/N, -1+(2*(N-1)+ 1.0)/N), (-1+1.0/N, -1+(2*(N-1)+ 1.0)/N), (-1+1.0/N, -1+(2*(N-1)+ 1.0)/N), (-1+1.0/N, -1+(2*(N-1)+ 1.0)/N))
x=[None]*4
y=[None]*4


def con(param):
    i, j, k, l, m, n, p, q = param[2:]
    for w in range(0, N - 1):
        if (i == (-1 + (2 * w + 1.0) / N)):
            c1 = True
        if (j == (-1 + (2 * w + 1.0) / N)):
            c2 = True
        if (k == (-1 + (2 * w + 1.0) / N)):
            c3 = True
        if (l == (-1 + (2 * w + 1.0) / N)):
            c4 = True
        if (m == (-1 + (2 * w + 1.0) / N)):
            c5 = True
        if (n == (-1 + (2 * w + 1.0) / N)):
            c6 = True
        if (p == (-1 + (2 * w + 1.0) / N)):
            c7 = True
        if (q == (-1 + (2 * w + 1.0) / N)):
            c8 = True
    if c1 & c2 & c3 & c4 & c5 & c6 & c7 & c8:
        return 0
    return 1

cons = [{'type':'eq', 'fun': con}]


# for i in range(0, N - 1):
#     x[0] = -1 + (2 * i + 1.0) / N
#     for j in range(0, N-1):
#         y[0] = -1 + (2 * j + 1.0) / N
#         for k in range(0, N - 1):
#             x[1] = -1 + (2 * k + 1.0) / N
#             for l in range(0, N - 1):
#                 y[1] = -1 + (2 * j + 1.0) / N
#                 for m in range(0, N - 1):
#                     x[2] = -1 + (2 * m + 1.0) / N
#                     for n in range(0, N - 1):
#                         y[2] = -1 + (2 * n + 1.0) / N
#                         for p in range(0, N - 1):
#                             x[3] = -1 + (2 * p + 1.0) / N
                            # for q in range(0, N - 1):
                            #     y[3] = -1 + (2 * q + 1.0) / N
                                # res0 = minimize(f, z0, method='SLSQP',
                                #            options={'disp': True}, bounds=bounds, tol=1e-6)
                            res1 = minimize(f, z1, method='SLSQP',
                                       options={'disp': True}, bounds=bounds, tol=1e-6, constraints=cons)
                            res2 = minimize(f, z2, method='SLSQP',
                                       options={'disp': True}, bounds=bounds, tol=1e-6, constraints=cons)
                            maximin=min(res1.fun,res2.fun)
                            if curmin > maximin:
                                maximin = curmin


print(maximin)
