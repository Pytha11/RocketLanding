"""Implements a simple time invariant, stateless wind model."""

import numpy as np

from PyFlyt.core import Aviary


# define the wind field
def simple_wind(time: float, position: np.ndarray):
    """Defines a simple wind updraft model.

    Args:
        time (float): time
        position (np.ndarray): position as an (n, 3) array

    """
    # the xy velocities are 0...
    wind = np.zeros_like(position)

    # and the vertical velocity is dependent on the logarithmic of height
    wind[:, 1] = -10

    return wind
