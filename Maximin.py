import math
from scipy.optimize import minimize


def f(x, s, r, u):
    z = ((x[0] - r * math.cos(u)) ** 2 + (x[1] - r * math.sin(u)) ** 2) ** (-s * .5) \
        + ((x[0]+ r * math.sin(u)) ** 2 + (x[1] - r * math.cos(u)) ** 2) ** (-s * .5) \
        + ((x[0] + r * math.cos(u)) ** 2 + (x[1] + r * math.sin(u)) ** 2) ** (-s * .5) \
        + ((x[0] - r * math.sin(u)) ** 2 + (x[1] + r * math.cos(u)) ** 2) ** (-s * .5)
    return z


def maximin(s, u):
    r = 0.7
    x1 = (1, 1)
    x2 = (1, 0)

    maximin = min(f(x1, s, r, u), f(x2, s, r, u))
    while r<1:
        r += 0.1
        if maximin < min(f(x1, s, r, u), f(x2, s, r, u)):
            maximin = min(f(x1, s, r, u), f(x2, s, r, u))
        else:
            r -= 0.2
            break

    maximin = min(f(x1, s, r, u), f(x2, s, r, u))
    while r<1:
        r += 0.01
        if maximin < min(f(x1, s, r, u), f(x2, s, r, u)):
            maximin = min(f(x1, s, r, u), f(x2, s, r, u))
        else:
            r -= 0.02
            break

    maximin = min(f(x1, s, r, u), f(x2, s, r, u))
    while r < 1:
        r += 0.001
        if maximin < min(f(x1, s, r, u), f(x2, s, r, u)):
            maximin = min(f(x1, s, r, u), f(x2, s, r, u))
        else:
            r -= 0.002
            break

    maximin = min(f(x1, s, r, u), f(x2, s, r, u))
    while r < 1:
        r += 0.0001
        if maximin < min(f(x1, s, r, u), f(x2, s, r, u)):
            maximin = min(f(x1, s, r, u), f(x2, s, r, u))
        else:
            break

    r-=0.0001
    return maximin


maximin(s=2, u=math.pi*0.25)