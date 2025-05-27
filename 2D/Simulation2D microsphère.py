import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 1) Constantes physiques
c0   = 3e8
mu0  = 4*np.pi*1e-7
eps0 = 1/(mu0*c0**2)

# 2) Longueur dâ€™onde et frÃ©quence (source gaussienne Ã  1030 nm)
lambda0 = 1030e-9
f0      = c0 / lambda0
omega0  = 2*np.pi * f0

# 3) Grille spatiale et pas de temps
dx = dy = lambda0 / 20
dt = 0.99 * dx / (c0 * np.sqrt(2))
Nx, Ny        = 200, 350
n_steps       = 800
pml_thickness = 30

# 4) Champs et permittivitÃ© initiale
Ez = np.zeros((Nx, Ny))
Hx = np.zeros((Nx, Ny))
Hy = np.zeros((Nx, Ny))
epsilon_r = np.ones((Nx, Ny))  # homogÃ¨ne air (n=1)

# 5) DÃ©finition des microsphÃ¨res (masques)
rayon = 30
Y, X = np.meshgrid(np.arange(Ny), np.arange(Nx))

cx0, cy0 = 100, 75
cx1, cy1 =  40, 75
cx2, cy2 = 160, 75

mask_cercle   = (X - cx0)**2 + (Y - cy0)**2 <= rayon**2
mask_cercle_1 = (X - cx1)**2 + (Y - cy1)**2 <= rayon**2
mask_cercle_2 = (X - cx2)**2 + (Y - cy2)**2 <= rayon**2

# Changement des valeurs de permittivitÃ© AVANT le calcul des coefficients
epsilon_r[mask_cercle]   = 2.1
epsilon_r[mask_cercle_1] = 2.1
epsilon_r[mask_cercle_2] = 2.1

# 6) ConductivitÃ©s pour PML (profil doux)
pml_thickness = 30  # augmente lÃ©gÃ¨rement l'Ã©paisseur
sigma_max = 2 * eps0 * (omega0)  # valeur adoucie mais frÃ©quence-dÃ©pendante

sigma_x = np.zeros(Nx)
sigma_y = np.zeros(Ny)
for i in range(pml_thickness):
    x_ratio = (pml_thickness - i) / pml_thickness
    profile = sigma_max * (x_ratio**4)  # quartique, lisse
    sigma_x[i] = sigma_x[Nx - 1 - i] = profile
    sigma_y[i] = sigma_y[Ny - 1 - i] = profile

Ïƒx2 = np.tile(sigma_x[:, None], (1, Ny))
Ïƒy2 = np.tile(sigma_y[None, :], (Nx, 1))

# 7) Coefficients FDTD (aprÃ¨s mise Ã  jour de epsilon_r)
Da = (1 - dt * Ïƒy2 / (2 * mu0)) / (1 + dt * Ïƒy2 / (2 * mu0))
Db = (dt / (mu0 * dy)) / (1 + dt * Ïƒy2 / (2 * mu0))
Ea = (1 - dt * Ïƒx2 / (2 * mu0)) / (1 + dt * Ïƒx2 / (2 * mu0))
Eb = (dt / (mu0 * dx)) / (1 + dt * Ïƒx2 / (2 * mu0))

Ce = Ïƒx2 + Ïƒy2
Ca = (1 - dt * Ce / (2 * eps0 * epsilon_r)) / (1 + dt * Ce / (2 * eps0 * epsilon_r))
Cb = (dt / (eps0 * epsilon_r * dx)) / (1 + dt * Ce / (2 * eps0 * epsilon_r))

# 8) Source gaussienne spatio-temporelle
source_y = Ny - pml_thickness - 15
source_x = Nx // 2
t0_phys  = 100 * dt
tau_phys = 30 * dt
width_x  = 10

# 9) PrÃ©paration de l'affichage
fig, ax = plt.subplots(figsize=(5, 6))
im = ax.imshow(Ez.T, origin='lower', cmap='RdBu', vmin=-0.1, vmax=0.1)

# TracÃ© des disques
for cx, cy in [(cx0, cy0), (cx1, cy1), (cx2, cy2)]:
    circle = plt.Circle((cx, cy), rayon, fill=False, color='r', lw=1.5)
    ax.add_patch(circle)

# Ligne repÃ¨re et source
x_vals = np.arange(Nx)
ax.plot(x_vals, np.full_like(x_vals, 40), 'r--', lw=1.5, label='Plaque y=40')
ax.plot(x_vals, np.full_like(x_vals, source_y), 'b-', lw=1.2, label=f'Source y={source_y}')

plt.colorbar(im, ax=ax, label='$E_z$')
ax.set_xlim(0, Nx)
ax.set_ylim(0, Ny)
ax.legend(loc='upper right')

# 10) Boucle dâ€™animation
def update(n):
    global Ez, Hx, Hy
    t = n * dt

    gauss_spatial = np.exp(-((np.arange(Nx) - source_x) ** 2) / (2 * width_x ** 2))
    gauss_env = np.exp(-((t - t0_phys) ** 2) / (2 * tau_phys ** 2))
    Ez[:, source_y] += gauss_spatial * gauss_env * np.sin(omega0 * t) * 0.2

    Hx[:, :-1] = Da[:, :-1] * Hx[:, :-1] - Db[:, :-1] * (Ez[:, 1:] - Ez[:, :-1])
    Hy[:-1, :] = Ea[:-1, :] * Hy[:-1, :] + Eb[:-1, :] * (Ez[1:, :] - Ez[:-1, :])

    Ez[1:, 1:] = Ca[1:, 1:] * Ez[1:, 1:] + Cb[1:, 1:] * (
        (Hy[1:, 1:] - Hy[:-1, 1:]) - (Hx[1:, 1:] - Hx[1:, :-1])
    )

    im.set_array(Ez.T)
    ax.set_title(f"Propagation Ã  t = {n}")
    return [im]

ani = FuncAnimation(fig, update, frames=n_steps, interval=30, blit=False)
plt.tight_layout()
plt.show()