"""Particle class that allows updating via the Boris method.

Based on https://github.com/gbogopolsky/boris-method
by L. Fuster & G. Bogopolsky

Adjusted to use SI, astropy units, and allow multiple particles via
broadcasting by M. H. van Kerkwijk

"""
# Would be nice to use CartesianRepresentation for vectors, but
# unfortunately, it is quite a lot slower.
from dataclasses import dataclass, replace

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


# Define the particle class
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

        Implementation follows description in §2.1.1 of Ripperda et al.,
        2018ApJS..235...21R (R+18 below).

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
        # Compute constant q*dt/(2m) found in many equations.
        factor = 0.5 * self.charge * dt / self.mass

        # First half time step on position per R+18 Eq. 6.
        self.r += 0.5 * dt * self.v

        # If E or B are functions, evaluate them at the half time step position.
        if callable(E):
            E = E(self)
        if callable(B):
            B = B(self)

        # Get previous timestep's u.
        u_old = self.v * gamma(self.v, keepdims=True)

        # Apply first half the electric field part (if present).
        u_m = u_old + E * factor if E is not None else u_old  # R+18, Eq. 10

        # Gamma cannot change in applying B, so gamma_m=gamma_p=gamma_mid.
        gamma_mid = gammau(u_m, keepdims=True)  # paragraph after R+18, Eq. 12

        # Apply magnetic field contribution.
        if B is not None:
            # Rotate velocity according to magnetic field.
            t = B * factor / gamma_mid  # paragraph after R+18, Eq. 12
            s = 2 * t / (1 + norm2(t, keepdims=True))
            u_p = u_m + np.cross(u_m + np.cross(u_m, t), s)  # R+18, Eq. 11

        else:
            u_p = u_m

        # Add second half of the electric field part.
        u_new = u_p + E * factor if E is not None else u_p  # R+18, Eq. 12
        # Use this to calculate new velocity v.
        self.v = u_new / gammau(u_new, keepdims=True)
        # Second half time step on position per R+18, Eq. 8
        self.r += 0.5 * dt * self.v
        return self
