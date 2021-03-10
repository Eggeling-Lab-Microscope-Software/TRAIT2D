#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

# Models used for ADC and SD analysis
class ModelBase:
    def __init__(self):
        self.R = 0.0
        self.dt = 0.0

class ModelBrownian(ModelBase):
    r"""Model for free, unrestricted diffusion.

    .. math:: D_\mathrm{app} = D + \frac{\delta^2}{2 t (1 - 2 R (dt / t))}
    """
    lower = [0.0, 0.0]
    upper = np.inf
    initial = [0.5e-12, 2.0e-9]

    def __call__(self, t, D, delta):
        return D + delta**2 / (2*t*(1-2*self.R*self.dt/t))

class ModelConfined(ModelBase):
    r"""Model for confined diffusion.

    .. math:: D_\mathrm{app} = D_\mu \cdot \frac{\tau}{t} \left( 1 - \exp \left( - \frac{\tau}{t} \right) \right) + \frac{\delta}{2 t (1 - 2 R (dt / t))}
    """
    lower = [0.0, 0.0, 0.0]
    upper = np.inf
    initial = [0.5e-12, 2.0e-9, 1.0e-3]

    def __call__(self, t, D_micro, delta, tau):
        return D_micro * (tau/t) * (1 - np.exp(-t/tau)) + \
            delta ** 2 / (2 * t * (1 - 2 * self.R * self.dt / t))

class ModelHop(ModelBase):
    r"""Model for hop diffusion.

    .. math:: D_\mathrm{app} = D_M + D_\mu \cdot \frac{\tau}{t} \left( 1 - \exp \left( - \frac{\tau}{t} \right) \right) + \frac{\delta}{2 t (1 - 2 R (dt / t))}
    """
    lower = [0.0, 0.0, 0.0, 0.0]
    upper = np.inf
    initial = [0.5e-12, 0.5e-12, 2.0e-9, 1.0e-3]

    def __call__(self, t, D_macro, D_micro, delta, tau):
        return D_macro + \
            D_micro * (tau/t) * (1 - np.exp(-t/tau)) + \
            delta ** 2 / (2 * t * (1 - 2 * self.R * self.dt / t))

class ModelImmobile(ModelBase):
    r"""Model for immobile diffusion.

    .. math:: D_\mathrm{app} =  \frac{\delta}{2 t (1 - 2 R (dt / t))}
    """
    upper = [np.inf]
    lower = [0.0]
    initial = [0.5e-12]

    def __call__(self, t, delta):
        return delta**2 / (2*t*(1-2*self.R*self.dt/t))

class ModelHopModified(ModelBase):
    r"""Modified model for hop diffusion.

    .. math::  D_\mathrm{app} = \alpha \cdot D_\mathrm{M} + (1 - \alpha) \cdot D_\mu \left(1 - \exp \left(- \frac{t}{\tau}\right)\right)
    """
    lower = [0.0, 0.0, 0.0, 0.0]
    upper = np.inf
    initial = [0.5e-12, 0.5e-12, 0.0, 1.0e-3]
    def __call__(self, t, D_macro, D_micro, alpha, tau):
        return alpha * D_macro + \
            (1.0 - alpha) * D_micro * (1 - np.exp(-t/tau))


# Models used for MSD analysis
class ModelLinear(ModelBase):
    """Linear model for MSD analysis."""
    def __call__(self, t, D, delta2):
        # 4 * D * t + 2 * delta2 - 8 * D * R * dt
        return 4.0 * D * (t - 2.0 * self.R * self.dt) + 2.0 * delta2 

class ModelPower(ModelBase):
    """Generic power law model for MSD analysis."""
    def __call__(self, t, D, delta2, alpha):
        # 4 * D * t**alpha + 2 * delta2 - 8 * D * R * dt
        return 4.0 * D * (t**alpha - 2.0 * self.R * self.dt) + 2.0 * delta2 