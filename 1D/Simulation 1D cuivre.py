import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button

# Constantes physiques
c = 3e8  # Vitesse de la lumière dans le vide (m/s)
dx = 50e-9  # Pas spatial (50 nm)
dt = dx / (2 * c)  # Pas temporel pour respecter la condition de Courant

# ParamÃ¨tres de simulation
Nx = 1000  # Nombre de points spatiaux
Nt = 1500  # Nombre de pas temporels (augmentÃ© pour donner plus de temps aux ondes de se sÃ©parer)

# Champs Électrique (Ez) et magnétique (By)
Ez = np.zeros(Nx)
By = np.zeros(Nx)

# Paramètres de l'onde EM
source_pos = 100
lambda0 = 1030e-9  # Longueur d'onde centrale (m)
omega0 = 2 * np.pi * c / lambda0  # FrÃ©quence angulaire

#Implusion 
T0 = 10e-15  # Durée de l'impulsion (s)
T_peak = 200  # Décalage temporel pour éviter le troncage
gaussian_pulse = lambda t: np.exp(-((t - T_peak) * dt / T0) ** 2) * np.sin(omega0 * (t - T_peak) * dt)

#Source continue 
#source_continue = lambda t: np.sin(omega0 * t * dt)

# Coefficients FDTD
C1 = np.ones(Nx) * (c * dt / dx)
C2 = np.ones(Nx) * (c * dt / dx)

# Listes pour l'animation
Ez_list = []
By_list = []

# Zone absorbante Ã gauche
absorption_region_length = int(5e-6 / dx)
absorption_factor = 0.5

# Position du cuivre
copper_start = Nx // 2
copper_stop = (Nx // 2) + 1e-6

# Paramètres du cuivre Ã  1030 nm (permittivité complexe)
epsilon_r_copper_real = -45.761
epsilon_r_copper_imag = 4.5744
mu_r_copper = 1.0

# Mise Ã  jour de C1 et C2 dans le cuivre (utilisation de np.arange pour un pas flottant)
for i in np.arange(copper_start, copper_stop, dx):  # Utilisation de np.arange avec dx comme pas
    # i est un indice flottant, on le convertit en entier pour l'accÃ¨s aux indices du tableau
    i_int = int(i)
    epsilon_complex = complex(epsilon_r_copper_real, epsilon_r_copper_imag)
    n_complex = np.sqrt(epsilon_complex * mu_r_copper)
    C1[i_int] = (c * dt / dx) / abs(n_complex)
    C2[i_int] = (c * dt / dx) / abs(n_complex)

# Facteur d'atténuation pour simuler les pertes dans le cuivre
attenuation_factor_copper = np.exp(-0.0584)

# Boucle FDTD principale
for t in range(Nt):
    # Mise Ã  jour du champ magnétique By
    for i in range(Nx - 1):
        By[i] += C1[i] * (Ez[i + 1] - Ez[i])
    
    # Mise Ã  jour du champ Électrique Ez
    for i in range(1, Nx):
        Ez[i] += C2[i] * (By[i] - By[i - 1])
    
    # Injection de l'impulsion gaussienne
    Ez[source_pos] += gaussian_pulse(t)
    
    # AttÃ©nuation dans la zone absorbante gauche
    for i in range(absorption_region_length):
        Ez[i] *= absorption_factor
    
    # AttÃ©nuation dans le cuivre (partie droite)
    for i in np.arange(copper_start, copper_stop, dx):  # Utilisation de np.arange avec dx comme pas
        i_int = int(i)  # Conversion de l'indice flottant en entier
        Ez[i_int] *= attenuation_factor_copper
    
    # Sauvegarde des champs pour animation
    Ez_list.append(Ez.copy())
    By_list.append(By.copy())

# CrÃ©ation de la figure pour l'animation
fig, ax = plt.subplots()
ax.set_xlim(0, Nx * dx * 1e6)
ax.set_ylim(-1, 1)
ax.set_xlabel("Position (Âµm)")
ax.set_xticks(np.linspace(0, Nx * dx * 1e6, 25))
ax.set_xticklabels([f"{x:.1f}" for x in np.linspace(0, Nx * dx * 1e6, 25)])
time_text = ax.text(0.02, 0.95, f'Temps : 0.000 fs', transform=ax.transAxes, fontsize=12, verticalalignment='top')

# Lignes de champ pour Ez et By
line_ez, = ax.plot([], [], lw=2, label='Ez (Champ Électrique)')
line_by, = ax.plot([], [], lw=2, label='By (Champ Magnétique)', color='red')

# Ligne de l'interface cuivre
copper_line = ax.axvline(x=copper_start * dx * 1e6, color='orange', linestyle='-', label="Plaque de cuivre (1 µm)")
# Grille
ax.grid(True)

# Fonction pause
paused = False
def toggle_pause(event):
    global paused
    paused = not paused
    if paused:
        button.label.set_text("Reprendre")
    else:
        button.label.set_text("Pause")

# Bouton pause
ax_button = plt.axes([0.85, 0.02, 0.1, 0.05])
button = Button(ax_button, 'Pause', color='lightgoldenrodyellow')
button.on_clicked(toggle_pause)

# Fonction de mise à jour de l'animation
def update(frame):
    if not paused:
        line_ez.set_data(np.linspace(0, Nx * dx * 1e6, Nx), Ez_list[frame])
        line_by.set_data(np.linspace(0, Nx * dx * 1e6, Nx), By_list[frame])
        time_fs = frame * dt * 1e15
        time_text.set_text(f'Temps : {time_fs:.3f} fs')
    return line_ez, line_by, time_text

# Analyse finale : extraction des amplitudes réflechie et transmise
Ez_final = Ez_list[-1]
Ez_reflected = Ez_final[:copper_start]
Ez_transmitted = Ez_final[copper_start:]
max_reflected = np.max(np.abs(Ez_reflected))
max_transmitted = np.max(np.abs(Ez_transmitted))

print(f"Amplitude maximale de l'onde réfléchie : {max_reflected:.4f}")
print(f"Amplitude maximale de l'onde transmise dans le cuivre : {max_transmitted:.4f}")

# Lancement de l'animation
ani = animation.FuncAnimation(fig, update, frames=Nt, interval=5, blit=True)
ax.legend(loc='upper right')
plt.show()