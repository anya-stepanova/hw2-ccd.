import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from astropy.io import fits
import lsp
import json


with fits.open('ccd.fits') as fits_file:
    data = fits_file[0].data.astype(float)


dif = np.empty((100, 493, 659))
for i in range(100):
    dif[i] = abs(data[i][0] - data[i][1])


disp = np.empty(100)
for i in range(100):
    dif_m = np.mean(dif[i])
    dif[i] = dif[i] - dif_m*np.ones((493, 659))
dif = np.square(dif)
for i in range(100):
    disp[i] = np.mean(dif[i])


x = np.empty(100)
x[0] = np.mean(data[0])
for i in range(1, 100):
    x[i] = np.mean(data[i]) - x[0]


plt.plot(disp, x)
plt.grid()
plt.legend()
plt.xlabel("x")
plt.ylabel("дисперсия")
plt.savefig('ccd.png')

y = np.empty((100, 2))
for i in range(100):
    y[i][0] = x[i]
    y[i][1] = 1.0

ab = np.empty(2)

ab, cost, var = lsp.lstsq_ne(y, disp)
disp_m = np.dot(y, ab)
plt.plot(disp_m, x)


r = (2*ab[1]/ab[0]**2)**0.5
g = 2/ab[0]
g_err = (2*var[0,0])/(ab[0]**2)
r_err = (-(((2*ab[1])**0.5)/ab[0]**2)*var[0,0]**2+var[1,1]**2/(ab[0]*(2*ab[1])))**0.5


text = {
    "ron": float('{:.3f}'.format(r)),
    "ron_err": float('{:.3f}'.format(r_err)),
    "gain": float('{:.3f}'.format(g)),
    "gain_err": g_err
}
with open('ccd.json', 'w') as f:
    json.dump(text, f)


