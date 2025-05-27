import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button

# Physical constants
c = 3e8  # Speed of light in vacuum (m/s)
dx = 50e-9  # Spatial step (50 nm)
dt = dx / (2 * c)  # Time step to satisfy the Courant condition

# Simulation parameters
Nx = 1000 # Number of grid points (increased to view the full pulse)
Nt = 1500 # Number of time steps (increased to track evolution)

# Definition of electric and magnetic fields
Ez = np.zeros(Nx)  # Electric field
Hy = np.zeros(Nx)  # Magnetic field

# Position of the pulse and central frequency
source_pos = 100
lambda0 = 1030e-9  # Central wavelength (m)
T0 = 10e-15  # Pulse duration (s)
omega0 = 2 * np.pi * c / lambda0  # Angular frequency

# Definition of the Gaussian pulse in time
T_peak = 200  # Temporal peak offset to avoid cut-off
gaussian_pulse = lambda t: np.exp(-((t - T_peak) * dt / T0) ** 2) * np.sin(omega0 * (t - T_peak) * dt)

# Coefficients for FDTD
C1 = np.ones(Nx) * (c * dt / dx)  # Homogeneous medium (vacuum)
C2 = np.ones(Nx) * (c * dt / dx)

# List to store electric and magnetic field values for animation
Ez_list = []
Hy_list = []

# Absorbing region parameter on the left
absorption_region_length = int(5e-6 / dx)  # 5 Âµm absorbing region on the left
absorption_factor = 0.5  # Absorption factor

# Fused silica parameters
epsilon_r_silica = 1.7755 # Relative permittivity of fused silica
mu_r_silica = 1.0  # Relative permeability of fused silica (approximated as 1)

# Update C1 and C2 for the fused silica region (right half of the grid)
silica_start = Nx // 2  # Position of the silica interface
for i in range(silica_start, Nx):
    C1[i] = (c * dt / dx) / np.sqrt(epsilon_r_silica * mu_r_silica)
    C2[i] = (c * dt / dx) / np.sqrt(epsilon_r_silica * mu_r_silica)

# Loop for FDTD time steps
for t in range(Nt):
    # Update magnetic field Hy
    for i in range(Nx - 1):
        Hy[i] += C1[i] * (Ez[i + 1] - Ez[i])

    # Update electric field Ez
    for i in range(1, Nx):
        Ez[i] += C2[i] * (Hy[i] - Hy[i - 1])

    # Add the Gaussian pulse to the electric field
    Ez[source_pos] += gaussian_pulse(t)

    # Apply absorption in the left region
    for i in range(absorption_region_length):
        Ez[i] *= absorption_factor  # Progressive attenuation in the absorbing region

    # Store fields for animation
    Ez_list.append(Ez.copy())
    Hy_list.append(Hy.copy())

# Create figure and axis for plotting
fig, ax = plt.subplots()
ax.set_xlim(0, Nx * dx * 1e6)  # Convert to micrometers
ax.set_ylim(-1, 1)
ax.set_xlabel("Position (Âµm)")
ax.set_xticks(np.linspace(0, Nx * dx * 1e6, 25))
ax.set_xticklabels([f"{x:.1f}" for x in np.linspace(0, Nx * dx * 1e6, 25)])

# Add text to display the time step in femtoseconds
time_text = ax.text(0.02, 0.95, f'Time: 0.000 fs', transform=ax.transAxes, fontsize=12, verticalalignment='top')

# Create lines for the plots
line_ez, = ax.plot([], [], lw=2, label='Ez (Electric Field)')
line_hy, = ax.plot([], [], lw=2, label='Hy (Magnetic Field)', color='red')

# Vertical line at the fused silica interface
silica_line = ax.axvline(x=silica_start * dx * 1e6, color='green', linestyle='--', label="Fused Silica Interface")

# Add grid to the graph
ax.grid(True)

# Pause functionality
paused = False

def toggle_pause(event):
    global paused
    paused = not paused
    if paused:
        button.label.set_text("Resume")
    else:
        button.label.set_text("Pause")

# Create a pause button
ax_button = plt.axes([0.85, 0.02, 0.1, 0.05])
button = Button(ax_button, 'Pause', color='lightgoldenrodyellow')
button.on_clicked(toggle_pause)

# Function to update the plot for animation
def update(frame):
    if not paused:
        # Update the data for the electric and magnetic fields
        line_ez.set_data(np.linspace(0, Nx * dx * 1e6, Nx), Ez_list[frame])  # Electric field
        line_hy.set_data(np.linspace(0, Nx * dx * 1e6, Nx), Hy_list[frame])  # Magnetic field
        
        # Convert the time step to femtoseconds for the time text
        time_fs = frame * dt * 1e15  # Convert time from seconds to femtoseconds
        time_text.set_text(f'Time: {time_fs:.3f} fs')  # Update the time text

    return line_ez, line_hy, time_text

# Analyse finale : trouver les pics dans la derniÃ¨re frame
Ez_final = Ez_list[-1]  # Dernier Ã©tat du champ Ã©lectrique

# SÃ©parer la zone avant et aprÃ¨s l'interface
Ez_reflected = Ez_final[:silica_start]
Ez_transmitted = Ez_final[silica_start:]

# Trouver les amplitudes maximales
max_reflected = np.max(np.abs(Ez_reflected))
max_transmitted = np.max(np.abs(Ez_transmitted))

# Afficher les rÃ©sultats
print(f"Amplitude maximale de l'onde rÃ©flÃ©chie : {max_reflected:.4f}")
print(f"Amplitude maximale de l'onde transmise dans la silice : {max_transmitted:.4f}")

# Animation
ani = animation.FuncAnimation(fig, update, frames=Nt, interval=5, blit=True)

# Legend
ax.legend(loc='upper right')
plt.show()