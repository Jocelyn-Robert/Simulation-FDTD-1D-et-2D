import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button

# 1) Constantes physiques
c0 = 3e8
mu0 = 4 * np.pi * 1e-7
eps0 = 1 / (mu0 * c0**2)

# 2) Longueur d’onde et fréquence
lambda0 = 1030e-9
f0 = c0 / lambda0
omega0 = 2 * np.pi * f0

# 3) Grille
dx = dy = lambda0 / 30
dt = 0.99 * dx / (c0 * np.sqrt(2))
Nx, Ny = 800, 200
n_steps = 1000
pml_thickness = 40

# 4) Champs et permittivité
Ez = np.zeros((Nx, Ny))
Hx = np.zeros((Nx, Ny))
Hy = np.zeros((Nx, Ny))
epsilon_r = np.ones((Nx, Ny))

# Lentille plan-convexe : couche plane + sphère convexe
lens_eps = 2.1
lens_thickness = 30  # Épaisseur en x
dome_radius = 120
lens_start_x = Nx // 4
lens_end_x = lens_start_x + lens_thickness
lens_center_y = Ny // 2

# Couche plane en verre
epsilon_r[lens_start_x:lens_end_x, :] = lens_eps

# Partie convexe : sphère en verre qui déborde sur la droite
for i in range(Nx):
    for j in range(Ny):
        x = i
        y = j
        if x >= lens_end_x:
            dx_ = x - lens_end_x
            dy_ = y - lens_center_y
            if dx_**2 + dy_**2 <= dome_radius**2:
                epsilon_r[x, y] = lens_eps

# 5) PML doux quartique
sigma_max = 2 * eps0 * omega0
sigma_x = np.zeros(Nx)
sigma_y = np.zeros(Ny)

for i in range(pml_thickness):
    x_ratio = (pml_thickness - i) / pml_thickness
    profile = sigma_max * (x_ratio**4)
    sigma_x[i] = sigma_x[Nx - 1 - i] = profile
    sigma_y[i] = sigma_y[Ny - 1 - i] = profile

σx2 = np.tile(sigma_x[:, None], (1, Ny))
σy2 = np.tile(sigma_y[None, :], (Nx, 1))

# 6) Coefficients FDTD
Da = (1 - dt * σy2 / (2 * mu0)) / (1 + dt * σy2 / (2 * mu0))
Db = (dt / (mu0 * dy)) / (1 + dt * σy2 / (2 * mu0))
Ea = (1 - dt * σx2 / (2 * mu0)) / (1 + dt * σx2 / (2 * mu0))
Eb = (dt / (mu0 * dx)) / (1 + dt * σx2 / (2 * mu0))

Ce = σx2 + σy2
Ca = (1 - dt * Ce / (2 * eps0 * epsilon_r)) / (1 + dt * Ce / (2 * eps0 * epsilon_r))
Cb = (dt / (eps0 * epsilon_r * dx)) / (1 + dt * Ce / (2 * eps0 * epsilon_r))

# 7) Source
source_y = Ny // 2
source_x = pml_thickness + 10
t0_phys = 100 * dt
tau_phys = 30 * dt
width_x = Ny // 2

# 8) Affichage
fig, ax = plt.subplots(figsize=(6, 6))
plt.subplots_adjust(bottom=0.25)
im = ax.imshow(Ez.T, origin='lower', cmap='RdBu', vmin=-0.3, vmax=0.3)

# Affichage lentille : contour plan-convexe corrigé
theta = np.linspace(-np.pi/2, np.pi/2, 200)
circle_x = lens_end_x + dome_radius * np.cos(theta)
circle_y = lens_center_y + dome_radius * np.sin(theta)

rect_x = [lens_start_x, lens_start_x, lens_end_x, lens_end_x]
rect_y = [0, Ny, Ny, 0]

ax.plot(circle_x, circle_y, 'k--', lw=1, label='Surface convexe')
ax.plot(rect_x, rect_y, 'k:', lw=1, label='Face plane')

plt.colorbar(im, ax=ax, label='$E_z$')
ax.set_xlim(0, Nx)
ax.set_ylim(0, Ny)
ax.legend(loc='upper right')

# 9) Bouton Pause/Play
is_paused = [False]
def toggle_pause(event):
    is_paused[0] = not is_paused[0]
    pause_button.label.set_text("▶ Reprendre" if is_paused[0] else "⏸ Pause")

ax_button = plt.axes([0.4, 0.05, 0.2, 0.075])
pause_button = Button(ax_button, "⏸ Pause")
pause_button.on_clicked(toggle_pause)

# 10) Animation
intensity_accum = np.zeros(Nx)
accum_count = 0
measure_started = False
intensity_profile = np.zeros(Nx)

def update(n):
    if is_paused[0]:
        return [im]

    global Ez, Hx, Hy, measure_started, intensity_accum, accum_count

    t = n * dt
    gauss_spatial = np.exp(-((np.arange(Ny) - source_y) ** 2) / (2 * width_x ** 2))
    gauss_env = np.exp(-((t - t0_phys) ** 2) / (2 * tau_phys ** 2))
    Ez[source_x, :] += gauss_spatial * gauss_env * np.sin(omega0 * t) * 0.2

    Hx[:, :-1] = Da[:, :-1] * Hx[:, :-1] - Db[:, :-1] * (Ez[:, 1:] - Ez[:, :-1])
    Hy[:-1, :] = Ea[:-1, :] * Hy[:-1, :] + Eb[:-1, :] * (Ez[1:, :] - Ez[:-1, :])
    Ez[1:, 1:] = Ca[1:, 1:] * Ez[1:, 1:] + Cb[1:, 1:] * (
        (Hy[1:, 1:] - Hy[:-1, 1:]) - (Hx[1:, 1:] - Hx[1:, :-1])
    )

    if n > 100:
        if not measure_started:
            print("📷 Début de la mesure à n =", n)
            measure_started = True
        band = 3
        Ez_slice = Ez[:, lens_center_y-band:lens_center_y+band+1]
        intensity_accum += np.mean(Ez_slice**2, axis=1)
        accum_count += 1
    if n == n_steps - 1:
        intensity_profile[:] = intensity_accum / accum_count
        x = np.arange(Nx) * dx
        focal_index = np.argmax(intensity_profile)
        focal_distance = (focal_index - lens_start_x) * dx
        print(f"📏 Distance focale estimée : {focal_distance*1e6:.2f} µm")

        plt.figure()
        plt.plot(x * 1e6, intensity_profile, label='Intensité moyenne $E_z^2$')
        plt.axvline(x=focal_index * dx * 1e6, color='r', linestyle='--', label='Focale')
        plt.axvline(x=lens_start_x * dx * 1e6, color='k', linestyle=':', label='Entrée lentille')
        plt.xlabel("x (µm)")
        plt.ylabel("Intensité")
        plt.title("Profil d'intensité le long de l'axe optique")
        plt.legend()
        plt.grid()
        plt.pause(5)

    im.set_array(Ez.T)
    ax.set_title(f"Propagation à t = {t * 1e15:.1f} fs")
    return [im]

ani = FuncAnimation(fig, update, frames=n_steps, interval=30, blit=False)
plt.show()

# 12) Théorie
f_theorique = dome_radius / (np.sqrt(lens_eps) - 1)
print(f"📐 Distance focale théorique (plan-convexe) : {f_theorique*dx*1e6:.2f} µm")
