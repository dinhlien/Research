import math
from scipy.optimize import minimize

# s = 2
# r = .8
# u = math.pi * 0.5


def f(x):
    z = ((x[0] - r * math.cos(u)) ** 2 + (x[1] - r * math.sin(u)) ** 2) ** (-s * .5) \
        + ((x[0]+ r * math.sin(u)) ** 2 + (x[1] - r * math.cos(u)) ** 2) ** (-s * .5) \
        + ((x[0] + r * math.cos(u)) ** 2 + (x[1] + r * math.sin(u)) ** 2) ** (-s * .5) \
        + ((x[0] - r * math.sin(u)) ** 2 + (x[1] + r * math.cos(u)) ** 2) ** (-s * .5)
    #z=x[1]**2+x[0]**2
    return z

bounds = ((-1, 1), (-1, 1))
x = [0.9, 0.9]
#[0.76873851 0.7909091 ]


res = minimize(f, x, method='SLSQP',
               options={'disp': True}, bounds=bounds, tol=1e-6)
print(res.x)

