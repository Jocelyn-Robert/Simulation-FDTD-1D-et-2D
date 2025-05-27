import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ParamÃ¨tres de la grille
Nx, Ny = 275, 275
dx = dy = 1
c = 1.0  # Vitesse de la lumiÃ¨re normalisÃ©e
dt = 0.5 * dx / c
n_steps = 400
pml_thickness = 35
silicax_start = Nx //(2)
epsilon_r_2D = np.ones((Nx, Ny))
epsilon_r_2D[silicax_start:, :] = 2.1

lambda0 = 20      # Longueur dâ€™onde centrale (m)
f0 = c / lambda0        # FrÃ©quence (Hz)
t0 = 40                 # DÃ©calage temporel en nombre dâ€™itÃ©rations
tau = 25                # Largeur gaussienne en nombre dâ€™itÃ©rations

# Champs
Ez = np.zeros((Nx, Ny))
Hx = np.zeros((Nx, Ny))
Hy = np.zeros((Nx, Ny))

# PML (conductivitÃ© croissante sur les bords)
sigma_max = 5.0 / (dt * pml_thickness)
sigma_x = np.zeros(Nx)
sigma_y = np.zeros(Ny)

for i in range(pml_thickness):
    value = sigma_max * ((pml_thickness - i) / pml_thickness) ** 3
    sigma_x[i] = sigma_x[Nx - 1 - i] = value
    sigma_y[i] = sigma_y[Ny - 1 - i] = value

sigma_x_2D = np.tile(sigma_x[:, np.newaxis], (1, Ny))
sigma_y_2D = np.tile(sigma_y[np.newaxis, :], (Nx, 1))

# Coefficients de mise Ã  jour
Ca = (1 - dt * (sigma_x_2D + sigma_y_2D) / 2) / (1 + dt * (sigma_x_2D + sigma_y_2D) / 2)
Cb = dt / (dx * epsilon_r_2D * (1 + dt * (sigma_x_2D + sigma_y_2D) / 2))
Da = (1 - dt * sigma_y_2D / 2) / (1 + dt * sigma_y_2D / 2)
Db = dt / dy / (1 + dt * sigma_y_2D / 2)
Ea = (1 - dt * sigma_x_2D / 2) / (1 + dt * sigma_x_2D / 2)
Eb = dt / dx / (1 + dt * sigma_x_2D / 2)

# Initialisation de la figure
fig, ax = plt.subplots()
im = ax.imshow(Ez.T, cmap='RdBu', origin='lower', vmin=-0.3, vmax=0.3, animated=True)
plt.colorbar(im, ax=ax)
ax.set_title("Propagation d'une onde EM en 2D (FDTD)")

# Position de la source
source_x, source_y = Nx // 5, Ny // 2

# Ajouter la ligne rouge en pointillÃ©s pour la sÃ©paration
separation_plan = Nx // (2) # Position de la sÃ©paration
plt.axvline(x=separation_plan, color='red', linestyle='--', linewidth=2, label="Interface Air/Verre")

# Animation
def update(frame):
    global Ez, Hx, Hy

    #Source impulsion
    t = frame * dt
    pulse = 9 * np.exp(-((t - t0) / tau) ** 2) * np.sin(2 * np.pi * f0 * dt * (frame - t0))
    Ez[source_x, source_y] += pulse

    # Mise Ã  jour des champs magnÃ©tiques
    Hx[:, :-1] =  (Da[:, :-1] * Hx[:, :-1] - Db[:, :-1] * (Ez[:, 1:] - Ez[:, :-1]))
    Hy[:-1, :] =  (Ea[:-1, :] * Hy[:-1, :] + Eb[:-1, :] * (Ez[1:, :] - Ez[:-1, :]))

    # Mise Ã  jour du champ Ã©lectrique
    Ez[1:, 1:] = (Ca[1:, 1:] * Ez[1:, 1:] + Cb[1:, 1:] * (
        (Hy[1:, 1:] - Hy[:-1, 1:]) - (Hx[1:, 1:] - Hx[1:, :-1]))
    )

    # Mise Ã  jour visuelle
    im.set_array(Ez.T)
    ax.set_title(f"Propagation Ã  t = {frame}")
    return [im]

ani = FuncAnimation(fig, update, frames=n_steps, interval=30, blit=False)
plt.show()