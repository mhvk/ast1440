"""Particle class that allows updating via the Boris method.

Based on https://github.com/gbogopolsky/boris-method
by L. Fuster & G. Bogopolsky

Adjusted to use SI, astropy units, and allow multiple particles via
broadcasting by M. H. van Kerkwijk

"""
# Would be nice to use CartesianRepresentation for vectors, but
# unfortunately, it is quite a lot slower.
from dataclasses import dataclass

import numpy as np
import astropy.constants as cst
import astropy.units as u

c2 = cst.c ** 2


def norm2(v, keepdims=False):
    """Norm squared of a vector (array)."""
    return (v**2).sum(-1, keepdims=keepdims)


def norm(v, keepdims=False):
    """Norm of a vector (array)."""
    return norm2(v, keepdims=keepdims) ** 0.5


def gamma(v, keepdims=False):
    """Gamma for given spatial velocity vector u."""
    return (1 - norm2(v, keepdims=keepdims) / c2) ** -0.5


def gammau(u, keepdims=False):
    """Gamma for given spatial part of 4-velocity vector u = gamma v."""
    return (1 + norm2(u, keepdims=keepdims) / c2) ** 0.5


# We define the particle class
@dataclass
class Particle:
    """Particle properties.

    Inputs can give multiple particles as arrays, with their properties
    broadcast.  If so, all vectors should have X, Y, Z in their last
    dimension (i.e., their shape ends in 3), while array-like scalar
    quantities need to have 1 as their last dimension.

    Parameters
    ----------
    mass, charge : ~astropy.units.Quantity
        Mass and charge of the particle.
    r, v : ~astropy.units.Quantity
        Initial position and velocity of the particles.
    """
    mass : u.Quantity
    charge : u.Quantity
    r : u.Quantity = [0, 0, 0] << u.m
    v : u.Quantity = [0, 0, 0] << u.m / u.s

    def push(self, dt, *, E=None, B=None):
        """Push the particles using Boris' method.

        Updates position and speed of the particle in-place, applying forces
        due to E and B assuming special relativity, i.e.,

        dp     du      d γv
        -- = m -- =  m ---- = q (E + v x B)
        dt     dt       dt

        where p is the spatial part of the 4-momentum, u the spatial part of
        the 4-velocity, v the 3-velocity, γ = √(1+(u/c)²) =
        1 / √(1-(v/c)²) the usual lorentz factor, q the charge, and
        E and B the electric and magnetic fields.

        Implementation follows description in Ripperda et al.,
        2018ApJS..235...21R, §2.1.1.

        Parameters
        ----------
        dt : ~astropy.units.Quantity
            Time step to take.
        E, B : ~astropy.units.Quantity or callable
            Electric and magnetic field.  If callable, they will be called
            with a particle instance, with the position propagated by half
            a time step.

        Returns
        -------
        self : Particle
            Updated version (note that update is done in-place).

        """
        if callable(E) or callable(B):
            rh = self.r + 0.5 * dt * self.v
            tp = self.__class__(self.mass, self.charge, rh, self.v)
            if callable(E):
                E = E(tp)
            if callable(B):
                B = B(tp)

        # Constant part in equations.
        factor = 0.5 * dt * self.charge / self.mass

        # Initial spatial part of the 4-velocity
        u_old = self.v * gamma(self.v, keepdims=True)
        # Apply half the electric field part (if present)
        u_m = u_old + E * factor if E is not None else u_old
        gamma_m = gammau(u_m, keepdims=True)

        if B is not None:
            # Rotate speed according to magnetic field
            t = B * factor / gamma_m
            s = 2 * t / (1 + norm2(t, keepdims=True))
            u_p = u_m + np.cross(u_m + np.cross(u_m, t), s)
        else:
            u_p = u_m

        # Add second half of the electric field part
        u_new = u_p + E * factor if E is not None else u_p

        # Calculate average 3-velocity for position update.
        vav = (u_old + u_new) / (2 * gamma_m)

        # Update position and velocity inplace.
        self.r += dt * vav
        self.v = u_new / gamma_m
        return self
