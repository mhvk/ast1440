"""Plot scattering cross section for any photon energy.

I.e., the Klein-Nishina formula.
"""

import matplotlib.pyplot as plt
import numpy as np

x = np.geomspace(0.001, 460, 1001)

# RL, eq.7.5
def sigma_kn(x):
    return (3/4)*(
        (1+x)/x**3 * (
            2*x*(1+x)/(1+2*x) - np.log(1+2*x)
        )
        + 1/(2*x) * np.log(1+2*x)
        - (1+3*x) / (1+2*x)**2
    )

sigma = sigma_kn(x)

midpoint = np.interp(0.5, sigma[::-1], x[::-1])
print(f"σ={sigma_kn(1):.3f}σ(T)) at hν/mₑc²=1")
print(f"σ=½σ(T) at hν/mₑc²={midpoint:.3f}")

sigma_nr = 1 - 2*x + (26/5)*x**2
sigma_er = (3/8)/x*(np.log(2*x) + 1/2)

plt.plot(x[x<0.2], sigma_nr[x<0.2], ":")
plt.plot(x[x>1], sigma_er[x>1], ":")
plt.plot(x, sigma)
plt.xscale('log')
# plt.yscale('log')
plt.xlabel(r"$h\nu/m_{e}c^{2}=\lambda_{C}/\lambda$")
plt.ylabel(r"$\sigma/\sigma_{T}$")
plt.show()
