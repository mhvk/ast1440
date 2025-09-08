"""Simulate a electron undergoing a short acceleration pulse."""
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.constants import c, e, eps0, mu0
from astropy.coordinates import CartesianRepresentation
from astropy.units import Unit as U
from astropy.visualization import quantity_support
from matplotlib.animation import FuncAnimation
from scipy.special import erf

# Input trajectory with a gaussian acceleration pulse.
# Time span modelled.
t = np.linspace(-5, 5, 201) << U("ns")
# Initial velocity.
v0 = 0.1 * c * CartesianRepresentation(1, 0, 0)
# Velocity change due to acceleration.
vlim = 0.5 * c * CartesianRepresentation(1, 0., 0.)
# Timescale of acceleration timescale (around t=0).
tacc = 0.5 * U("ns")

# Time at which field should be measured.  If more than one, animate it.
tm = np.linspace(t[0], t[-1], 51)
# tm = t[-1]

# Set up grid
g = np.linspace(-2., 2., 50) << U("m")
rg = CartesianRepresentation(g, g[:, np.newaxis], 0*U("m"))


def interp(tg, t, rep):
    """Interpolate a representation given for times t on new times tg.

    Like np.interp, but for a representation.
    """
    return CartesianRepresentation(*[np.interp(tg, t, c) for c in rep.xyz])


def get_pret(rg, tm, t, rp, vp, ap):
    """Get retarded positions for given grid positions and times.

    Returns tret, rpret, vpret, apret.
    """
    # What was travel time? Initial guess zero.
    dt = 0 * U("s")
    it = 0
    tret = 1e10*U("s")  # obviously bad to enter while
    while not np.allclose(tm - tret, dt, atol=5*U("ps"), rtol=0):
        # Calculate where particle was a light travel time ago.
        tret = tm - dt
        rpret = interp(tret, t, rp)
        # Calculate new travel time.
        dt = (rg - rpret).norm() / c
        it += 1
        if it > 30:
            raise RuntimeError("not converging on retarded position.")

    return tret, rpret, interp(tret, t, vp), interp(tret, t, ap)


def calcEandB(q, rg, rret, vret, aret):
    """E and B on given grid for given retarded properties."""
    # RL, Eq. 3.9. Note that wikipedia/libretexts write, perhaps more
    # logically, the 1-β² term in the nominator as γ² in the denominator.
    beta = vret / c
    beta_dot = aret / c
    Rg = rg - rret
    R = Rg.norm()
    n = Rg / R
    kappa = 1 - n.dot(beta)
    E = q / (4*np.pi*eps0) / kappa**3 * (
        ((n-beta) * (1-beta.norm()**2) / R**2)
        + ((n / R).cross((n-beta).cross(beta_dot))) / c)
    return E, n.cross(E) / c


def get_ExEyBzn(rg, tm, t, rp, vp, ap):
    """Get normalized Ex, Ey, Bz for given grid and particular trajectory.

    Just a combination of get_pret, calcEandB, and normalization by E,
    used for plotting.
    """
    tret, rpret, vpret, apret = get_pret(rg, tm, t, rp, vp, ap)
    E, B = calcEandB(-e.si, rg, rpret, vpret, apret)
    En = E.norm()
    return (E.x / En).to_value(1), (E.y / En).to_value(1), (B.z * c / En).to_value(1)


# Define particle trajectory (done classically, i.e., not quite correct).
tr = 0.5**0.5 * t / tacc
ap = np.exp(-tr**2) / (np.sqrt(np.pi) * tacc.to("s")) * vlim
vp = v0 + vlim * (erf(tr) + 1) / 2.
rp = (v0 * t.to("s")
      + vlim * tacc * (tr*erf(tr) + np.exp(-tr**2)/np.sqrt(np.pi) + tr) / 2.)

# Calculate interpolated positions and fields.
rp_tm = interp(tm, t, rp)
_tm = tm.reshape((-1,) + (1,)*rg.ndim) if tm.ndim == 1 else tm
Exn, Eyn, Bzn = get_ExEyBzn(rg, _tm, t, rp, vp, ap)


# Make plot or animation.
quantity_support()
fig, ax = plt.subplots(figsize=(10, 8))
plot_unit = rg.x.unit
ax.set_xlabel(rf"$x~[{plot_unit.to_string('latex_inline')[1:-1]}]$")
ax.set_ylabel(rf"$y~[{plot_unit.to_string('latex_inline')[1:-1]}]$")
ax.set_xlim(rg.x.min(), rg.x.max())
ax.set_ylim(rg.y.min(), rg.y.max())
ax.axis("square")
ax.scatter(rp.x, rp.y, c="blue", alpha=10/len(t))

# Fake initial set-up, just to have elements to update.
# Need to cover -1 to 1 to get color bar to work right.
_uni = np.linspace(-1, 1, rg.size).reshape(rg.shape)
quiver = ax.quiver(rg.x, rg.y, _uni, _uni, _uni,
                   headlength=0, headwidth=0, headaxislength=0, pivot="middle")
location = ax.scatter([], [], c="magenta")


def update(frame):
    """Draw field vectors as well as current time and position."""
    tm, Exn, Eyn, Bzn, rp_tm = frame
    ax.set_title(f"$t={tm.to_string(format='latex', formatter='+.3f')[1:-1]}$")
    quiver.set_UVC(Exn, Eyn, Bzn)
    location.set_offsets([rp_tm.x.to_value(plot_unit),
                          rp_tm.y.to_value(plot_unit)])
    return quiver, location

if tm.shape:
    anim = FuncAnimation(fig, update, frames=zip(tm, Exn, Eyn, Bzn, rp_tm),
                         interval=3, repeat_delay=1000, save_count=len(tm))
else:
    update((tm, Exn, Eyn, Bzn, rp_tm))

fig.tight_layout()
plt.show()
