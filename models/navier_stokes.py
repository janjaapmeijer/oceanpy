import numpy as np
import matplotlib.pyplot as plt
from OceanPy.animations import play1D, play2D, play3D

# L = 2
# nx = 41
# dx = L / (nx - 1)
# x = np.linspace(0, L, nx)
# nt = 25
# dt = 0.025
# t = np.linspace(0, 0.6, nt)
# c = 1
#
# u = np.ones((nt, nx))
# # un = np.ones(nx)
#
# u[0, int(.5 / dx) : int(1 / dx + 1)] = 2
#
# for n in range(nt-1):
#     # un = u[n, :].copy()
#     for i in range(1, nx):
#         u[n+1, i] = u[n, i] - c * dt / dx * (u[n, i] - u[n, i-1])
#
# play1D(t, x, u, interval=10)

################################################

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

# constants
nu = .01

# initialisation
nx = 41
ny = 41
nt = 120
# c = 1

L = 2
W = 2
dx = L / (nx - 1)
dy = W / (ny - 1)
sigma = .0009
dt = sigma * dx * dy / nu

x = np.linspace(0, L, nx)
y = np.linspace(0, W, ny)
X, Y = np.meshgrid(x, y)

u = np.ones((nt, ny, nx))
v = np.ones((nt, ny, nx))

# boundary conditions
u[0, int(0.5 / dy) : int(1 / dy + 1), int(.5 / dx) : int(1 / dx + 1)] = 2
v[0, int(0.5 / dy) : int(1 / dy + 1), int(.5 / dx) : int(1 / dx + 1)] = 2

### 1-DIMENSIONAL ######################################################################################################

## Linear convection
def linear_convection():
    for n in range(nt - 1):
        u[n+1, 1:, 1:] = (u[n, 1:, 1:] - (c * dt / dx * (u[n, 1:, 1:] - u[n, 1:, :-1])) -
                          (c * dt / dy * (u[n, 1:, 1:] - u[n, :-1, 1:])))
        u[n, 0, :] = 1
        u[n, -1, :] = 1
        u[n, :, 0] = 1
        u[n, :, -1] = 1

    return u

## Diffusion
def diffusion():
    for n in range(nt - 1):
        u[n+1, 1:-1, 1:-1] = (u[n, 1:-1, 1:-1] +
                              nu * dt / dx ** 2 * (u[n, 1:-1, 2:] - 2 * u[n, 1:-1, 1:-1] + u[n, 1:-1, 0:-2]) +
                              nu * dt / dy ** 2 * (u[n, 2:, 1:-1] - 2 * u[n, 1:-1, 1:-1] + u[n, 0:-2, 1:-1]))
        u[n, 0, :] = 1
        u[n, -1, :] = 1
        u[n, :, 0] = 1
        u[n, :, -1] = 1

    return u

### 2-DIMENSIONAL ######################################################################################################

## Non-linear convection
def nonlinear_convection():
    for n in range(nt - 1):
        u[n+1, 1:, 1:] = (u[n, 1:, 1:] -
                          u[n, 1:, 1:] * dt / dx * (u[n, 1:, 1:] - u[n, 1:, :-1]) -
                          v[n, 1:, 1:] * dt / dy * (u[n, 1:, 1:] - u[n, :-1, 1:]))
        v[n+1, 1:, 1:] = (v[n, 1:, 1:] -
                          u[n, 1:, 1:] * dt / dx * (v[n, 1:, 1:] - v[n, 1:, :-1]) -
                          v[n, 1:, 1:] * dt / dy * (v[n, 1:, 1:] - v[n, :-1, 1:]))
        u[n, 0, :] = 1
        u[n, -1, :] = 1
        u[n, :, 0] = 1
        u[n, :, -1] = 1

    return u, v




## Burgers' Equation
def burgers_equation():
    for n in range(nt - 1):
        u[n+1, 1:-1, 1:-1] = (u[n, 1:-1, 1:-1] -
                              dt / dx * u[n, 1:-1, 1:-1] * (u[n, 1:-1, 1:-1] - u[n, 1:-1, 0:-2]) -
                              dt / dy * v[n, 1:-1, 1:-1] * (u[n, 1:-1, 1:-1] - u[n, 0:-2, 1:-1]) +
                              nu * dt / dx ** 2 * (u[n, 1:-1, 2:] - 2 * u[n, 1:-1, 1:-1] + u[n, 1:-1, 0:-2]) +
                              nu * dt / dy ** 2 * (u[n, 2:, 1:-1] - 2 * u[n, 1:-1, 1:-1] + u[n, 0:-2, 1:-1]))

        v[n+1, 1:-1, 1:-1] = (v[n, 1:-1, 1:-1] -
                              dt / dx * u[n, 1:-1, 1:-1] * (v[n, 1:-1, 1:-1] - v[n, 1:-1, 0:-2]) -
                              dt / dy * v[n, 1:-1, 1:-1] * (v[n, 1:-1, 1:-1] - v[n, 0:-2, 1:-1]) +
                              nu * dt / dx ** 2 * (v[n, 1:-1, 2:] - 2 * v[n, 1:-1, 1:-1] + v[n, 1:-1, 0:-2]) +
                              nu * dt / dy ** 2 * (v[n, 2:, 1:-1] - 2 * v[n, 1:-1, 1:-1] + v[n, 0:-2, 1:-1]))
        u[n, 0, :] = 1
        u[n, -1, :] = 1
        u[n, :, 0] = 1
        u[n, :, -1] = 1

        v[n, 0, :] = 1
        v[n, -1, :] = 1
        v[n, :, 0] = 1
        v[n, :, -1] = 1

    return u, v

def momentum_equations():
    for n in range(nt - 1):
        u[n+1, 1:-1, 1:-1] = (u[n, 1:-1, 1:-1] -
                              dt / dx * u[n, 1:-1, 1:-1] * (u[n, 1:-1, 1:-1] - u[n, 1:-1, 0:-2]) -
                              dt / dy * v[n, 1:-1, 1:-1] * (u[n, 1:-1, 1:-1] - u[n, 0:-2, 1:-1]) -
                              dt / 2 * rho * dx * ()
                              nu * dt / dx ** 2 * (u[n, 1:-1, 2:] - 2 * u[n, 1:-1, 1:-1] + u[n, 1:-1, 0:-2]) +
                              nu * dt / dy ** 2 * (u[n, 2:, 1:-1] - 2 * u[n, 1:-1, 1:-1] + u[n, 0:-2, 1:-1]))

        v[n+1, 1:-1, 1:-1] = (v[n, 1:-1, 1:-1] -
                              dt / dx * u[n, 1:-1, 1:-1] * (v[n, 1:-1, 1:-1] - v[n, 1:-1, 0:-2]) -
                              dt / dy * v[n, 1:-1, 1:-1] * (v[n, 1:-1, 1:-1] - v[n, 0:-2, 1:-1]) +
                              nu * dt / dx ** 2 * (v[n, 1:-1, 2:] - 2 * v[n, 1:-1, 1:-1] + v[n, 1:-1, 0:-2]) +
                              nu * dt / dy ** 2 * (v[n, 2:, 1:-1] - 2 * v[n, 1:-1, 1:-1] + v[n, 0:-2, 1:-1]))
        u[n, 0, :] = 1
        u[n, -1, :] = 1
        u[n, :, 0] = 1
        u[n, :, -1] = 1

        v[n, 0, :] = 1
        v[n, -1, :] = 1
        v[n, :, 0] = 1
        v[n, :, -1] = 1

    return u, v


u, v = nonlinear_convection()

play3D(X,Y,u, cmap=cm.viridis)
# play2D(X,Y,u)
