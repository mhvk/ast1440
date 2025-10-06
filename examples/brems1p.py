"""Bremsstralung for 1 particle.

Calculate explicitly given hyperbolic trajectory, following Pad p.296.
Cannot yet reproduce the integration with the Hankel functions yet.
"""
import astropy.units as u
import astropy.constants as const
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sps
from astropy.visualization import quantity_support

plt.ion()
quantity_support()

u.set_enabled_equivalencies(u.dimensionless_angles())


def power(z):
    return z.real**2 + z.imag**2


def interp(tg, t, x):
    """Interpolate over first dimension of 2-dim array."""
    t, x = np.broadcast_arrays(t, x, subok=True)
    tg = np.broadcast_to(tg, t.shape[:-1] + tg.shape[-1:], subok=True)
    return np.stack([np.interp(_tg, _t, _x) for (_tg, _t, _x) in zip(tg, t, x)])


# b = [[5]] * u.um
v_e = np.sqrt(const.k_B*1e4*u.K/const.m_e).to("km/s")
bmin = (4*const.e.esu**2/(np.pi*const.m_e*v_e**2)).to("um")
# Upper limit seems reasonable for n_e = 1/cm3; linear approx good.
# b = np.geomspace(5 * u.nm, 5 * u.mm, 7)[:, np.newaxis]
b = np.geomspace(bmin, 1000 * bmin, 7)[:, np.newaxis]
q_e = 1 * const.e.si
q_ion = 1 * const.e.si
alpha = q2by4pieps0 = q_e * q_ion / (4 * np.pi * const.eps0)

epsilon = np.sqrt(1 + b**2 * ((const.m_e * v_e**2) / q2by4pieps0)**2).to(1)

# semi-major axis of hyperbolic trajectory.
a = (q2by4pieps0 / (const.m_e * v_e**2)).to("um")

xi = np.linspace(-2*np.pi, 2*np.pi, 2001)

x = a * (epsilon-np.cosh(xi))
y = a * np.sqrt(epsilon**2 - 1) * np.sinh(xi)

fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 8))
ax0.scatter(0*u.mm, 0*u.mm)
ax0.plot(y.T, x.T)
ax0.axis("equal")
ax0.set_xlim(-2*b.max(), 2*b.max())

t_xi = (np.sqrt(const.m_e * a**3 / alpha) * (epsilon * np.sinh(xi) - xi)).to("ns")

t = np.linspace(t_xi[..., 1], t_xi[..., -1], 1001, axis=-1)
dt = t[..., 1:2]-t[..., 0:1]
xt = interp(t, t_xi, x)
yt = interp(t, t_xi, y)

om = np.linspace(0, 2 * (v_e / b[:, 0]).to("rad/s"), 1001, axis=-1)
om[:, 0] = om[:, 1].min() / 10.
dom = om[..., 2:3]-om[..., 1:2]

# Slow FT, of acceleration directly (which is well-behaved,
# going to zero at large t)
axt = np.gradient(np.gradient(xt, axis=-1), axis=-1) / dt**2
axom = (axt[..., np.newaxis, :] * np.exp(1j*om[..., np.newaxis]*t[..., np.newaxis, :]) * dt[..., np.newaxis, :]).sum(-1)
ayt = np.gradient(np.gradient(yt, axis=-1), axis=-1) / dt**2
ayom = (ayt[..., np.newaxis, :] * np.exp(1j*om[..., np.newaxis]*t[..., np.newaxis, :]) * dt[..., np.newaxis, :]).sum(-1)

# Comparison with 6.208, NOT OK YET!!
def h1(nu, z):
    # Pad has imaginary nu, use https://dlmf.nist.gov/10.24#E2
    assert np.all(nu.real == 0)
    return sps.hankel1(nu.imag, z).real / np.cosh(np.pi / 2 * nu.imag)

def h1prime(nu, z):
    assert np.all(nu.real == 0)
    return h1(nu, z)

# mu = (a / v_e * om).to(1)
# xom_pad = np.pi * a / om * h1prime(1j*mu, 1j*mu*epsilon)
# yom_pad = np.pi*a*np.sqrt(epsilon**2-1)/om/epsilon*h1(1j*mu, 1j*mu*epsilon)

factor_check = 2/(3*np.pi) * const.e.esu**2 / const.c**3 * (q_e*q_ion/const.e.si**2)
factor = 2/(3*np.pi) * alpha / const.c**3
assert np.isclose(factor, factor_check)
dedomx = (factor * power(axom)).to("aJ/Hz")
dedomy = (factor * power(ayom)).to("aJ/Hz")
dedom = dedomx + dedomy
dedoms = dedom * b * dom[:1] / dom
ax1.semilogx(om.T, dedoms.T)

# Show x, y of middle one
i = om.shape[0] // 2
ax1.semilogx(om[i], dedomx[i] * b[i] * dom[0] / dom [i], 'k:')
ax1.semilogx(om[i], dedomy[i] * b[i] * dom[0] / dom [i], 'k:')

_om = np.geomspace(om.min(), om.max(), 1001)
dedomi = interp(_om, om, dedoms)
ax1.semilogx(_om, dedomi.mean(0))
om_lim = v_e/bmin
ax1.vlines(om_lim, dedoms.min(), dedoms.max())
ax1.set_xlim(om.min(), np.maximum(om_lim, om.max())*1.1)
