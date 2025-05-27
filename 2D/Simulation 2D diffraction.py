import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Paramètres de la grille
Nx, Ny = 275, 275
dx = dy = 1
c = 1.0  # Vitesse normalisée
dt = 0.5 * dx / c
n_steps = 400
pml_thickness = 35

# Paramètres de l'impulsion 
lambda0 = 20   
f0 = c / lambda0       
t0 = 40                 
tau = 10                

# Initialisation des champs
Ez = np.zeros((Nx, Ny))
Hx = np.zeros((Nx, Ny))
Hy = np.zeros((Nx, Ny))

# PML 
sigma_max = 5.0 / (dt * pml_thickness)
sigma_x = np.zeros(Nx)
sigma_y = np.zeros(Ny)

for i in range(pml_thickness):
    value = sigma_max * ((pml_thickness - i) / pml_thickness) ** 3
    sigma_x[i] = sigma_x[Nx - 1 - i] = value
    sigma_y[i] = sigma_y[Ny - 1 - i] = value

sigma_x_2D = np.tile(sigma_x[:, np.newaxis], (1, Ny))
sigma_y_2D = np.tile(sigma_y[np.newaxis, :], (Nx, 1))

# Interface sinusoïdale : variation selon y
amplitude = 20
wavelength = 50
interface_curve = (Nx // 2) + amplitude * np.sin(2 * np.pi * np.arange(Ny) / wavelength)
interface_curve = np.clip(interface_curve.astype(int), 0, Nx - 1)  # sécurité

# Matériau : Air (ε=1) / Mur opaque (ε=2.1) avec fentes
epsilon_r = np.ones((Nx, Ny))
mur_x = Nx // 2
fente_height = 5
fente_y_1 = Ny // 2 + 7  # position de la première fente
fente_y_2 = Ny // 2 - 7  # position de la deuxième fente

mur_start = Nx // 2
mur_width = 5

for i in range(mur_start, mur_start + mur_width):
    for j in range(Ny):
        in_fente1 = (fente_y_1 - fente_height // 2 <= j <= fente_y_1 + fente_height // 2)
        in_fente2 = (fente_y_2 - fente_height // 2 <= j <= fente_y_2 + fente_height // 2)
        if not (in_fente1 or in_fente2):
            epsilon_r[i, j] = 1000  # mur très réfléchissant

# Coefficients de mise à jour
Ca = (1 - dt * (sigma_x_2D + sigma_y_2D) / 2) / (1 + dt * (sigma_x_2D + sigma_y_2D) / 2)
Cb = dt / (dx * epsilon_r * (1 + dt * (sigma_x_2D + sigma_y_2D) / 2))
Da = (1 - dt * sigma_y_2D / 2) / (1 + dt * sigma_y_2D / 2)
Db = dt / dy / (1 + dt * sigma_y_2D / 2)
Ea = (1 - dt * sigma_x_2D / 2) / (1 + dt * sigma_x_2D / 2)
Eb = dt / dx / (1 + dt * sigma_x_2D / 2)

# Initialisation de la figure
fig, ax = plt.subplots()
im = ax.imshow(Ez.T, cmap='RdBu', origin='lower', vmin=-0.05, vmax=0.05, animated=True)
# ligne optionnelle pour tracer la forme de l’interface sinusoïdale
# line, = ax.plot(interface_curve, np.arange(Ny), 'r--', linewidth=2, label="Interface Air/Verre")
plt.colorbar(im, ax=ax)
ax.set_title("Propagation d'une onde EM en 2D (FDTD)")

# Position de la source
source_x, source_y = Nx // 5, Ny // 2

# Animation
def update(frame):
    global Ez, Hx, Hy

    # Source impulsion
    t = frame * dt
    pulse = 9 * np.exp(-((t - t0) / tau) ** 2) * np.sin(2 * np.pi * f0 * dt * (frame - t0))
    Ez[source_x, source_y] += pulse

    # Mise à jour des champs magnétiques
    Hx[:, :-1] = Da[:, :-1] * Hx[:, :-1] - Db[:, :-1] * (Ez[:, 1:] - Ez[:, :-1])
    Hy[:-1, :] = Ea[:-1, :] * Hy[:-1, :] + Eb[:-1, :] * (Ez[1:, :] - Ez[:-1, :])

    # Mise à jour du champ électrique
    Ez[1:, 1:] = Ca[1:, 1:] * Ez[1:, 1:] + Cb[1:, 1:] * (
        (Hy[1:, 1:] - Hy[:-1, 1:]) - (Hx[1:, 1:] - Hx[1:, :-1]))

    im.set_array(Ez.T)
    ax.set_title(f"Propagation à t = {frame}")
    return [im]

ani = FuncAnimation(fig, update, frames=n_steps, interval=30, blit=True)
plt.show()
