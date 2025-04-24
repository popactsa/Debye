from scipy.interpolate import make_smoothing_spline
import matplotlib.pyplot as plt
import numpy as np

K_to_erg    = 1.381e-16
kappa_values = np.array([116.6, 113.5, 111.2, 110.1, 109.0, 108.3, 107.2, 106.8, 107.4,
                108.8, 107.5, 107.2, 103.6, 101.1, 99.0]) * 1.0e5 / K_to_erg

kappa_T_net = np.array([1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600, 2800, 3000,
               3200, 3400, 3600, 3695])

kappa_spline = make_smoothing_spline(kappa_T_net, kappa_values, None)
print(kappa_spline(1000))
T_net = np.linspace(1000, 3695, 200)
plt.plot(T_net, kappa_spline(T_net))
plt.plot(kappa_T_net, kappa_values)
plt.show()



