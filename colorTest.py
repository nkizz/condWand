import math
import matplotlib.pyplot as plt

def floatRgb(mag, cmin, cmax):
    """ Return a tuple of floats between 0 and 1 for R, G, and B. """
    # Normalize to 0-1
    try: x = float(mag-cmin)/(cmax-cmin)
    except ZeroDivisionError: x = 0.5 # cmax == cmin
    blue  = 1-min((max((4*(0.75-x), 0.)), 1.))
    red   = 1-min((max((4*(x-0.25), 0.)), 1.))
    green = 1-min((max((4*math.fabs(x-0.5)-1., 0.)), 1.))
    return red, green, blue
for i in range(10):
    plt.plot(i, 'o', color=floatRgb(i, 2.5, 7.5))
plt.show()
