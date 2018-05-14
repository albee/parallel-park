import numpy as np
import scipy.interpolate as spint

RGI = spint.RegularGridInterpolator

x = np.linspace(0, 1, 3) 

# populate the 3D array of values (re-using x because lazy)
X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
vals = np.sin(X) + np.cos(Y) + np.tan(Z)
print(vals)
print(x)
# make the interpolator, (list of 1D axes, values at all points)
rgi = RGI(points=[x, x, x], values=vals)  # can also be [x]*3 or (x,)*3

tst = (0.47, 0.49, 0.53)

print rgi(tst)
print np.sin(tst[0]) + np.cos(tst[1]) + np.tan(tst[2])