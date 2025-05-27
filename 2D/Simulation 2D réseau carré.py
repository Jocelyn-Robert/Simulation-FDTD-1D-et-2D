import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button

# 1) Constantes physiques
c0   = 3e8
mu0  = 4*np.pi*1e-7
eps0 = 1/(mu0*c0**2)

# 2) Longueur d’onde et fréquence
lambda0 = 1030e-9
f0      = c0/lambda0
omega0  = 2*np.pi*f0

# 3) Grille et PML
dx = dy = lambda0/20
dt = 0.99*dx/(c0*np.sqrt(2))
Nx, Ny        = 325, 225
n_steps       = 800
pml_thickness = 40

# 4) Champs et permittivité
Ez = np.zeros((Nx,Ny))
Hx = np.zeros((Nx,Ny))
Hy = np.zeros((Nx,Ny))
epsilon_r = np.ones((Nx,Ny))   # on ne l’utilise pas pour le PEC

# 5) Définition du réseau carré de blocs d’or (PEC)
# Hypothèse : on reste en PEC
a = 2e-6      # période 2 µm
s = 1e-6    # blocs de 0.5 µm
L_free = 5e-6 # zone libre avant le réseau

# calcul de la zone libre en mailles et de la coord. x0
buffer_cells = int(L_free/dx)
x0 = (pml_thickness + buffer_cells)*dx

# grille physique
x = np.arange(Nx)*dx
y = np.arange(Ny)*dy
X, Y = np.meshgrid(x,y,indexing='ij')

# centres du réseau carré
cx = np.arange(x0 + a/2, Nx*dx, a)
cy = np.arange(a/2,     Ny*dy, a)

# masque PEC : True là où se trouvent les blocs d’or
mask_PEC = np.zeros((Nx,Ny), dtype=bool)
for xc in cx:
    for yc in cy:
        mask_PEC |= (np.abs(X-xc)< s/2) & (np.abs(Y-yc)< s/2)
# --- restriction du réseau à ix ∈ [150,250) ---
ix_min, ix_max = 150, 185
# on construit un masque d’indices X
I = np.arange(Nx)[:, None]      # vecteur colonnes de 0…Nx-1
mask_band = (I >= ix_min) & (I < ix_max)
# on désactive tout ce qui est hors bande
mask_PEC &= mask_band

# 6) PML doux quartique
sigma_max = 2*eps0*omega0
sigma_x = np.zeros(Nx)
sigma_y = np.zeros(Ny)
for i in range(pml_thickness):
    r = (pml_thickness - i)/pml_thickness
    prof = sigma_max*(r**4)
    sigma_x[i] = sigma_x[Nx-1-i] = prof
    sigma_y[i] = sigma_y[Ny-1-i] = prof

σx2 = np.tile(sigma_x[:,None],(1,Ny))
σy2 = np.tile(sigma_y[None,:],(Nx,1))

# 7) Coefficients FDTD
Da = (1 - dt*σy2/(2*mu0)) / (1 + dt*σy2/(2*mu0))
Db = (dt/(mu0*dy))    / (1 + dt*σy2/(2*mu0))
Ea = (1 - dt*σx2/(2*mu0)) / (1 + dt*σx2/(2*mu0))
Eb = (dt/(mu0*dx))    / (1 + dt*σx2/(2*mu0))

Ce = σx2 + σy2
Ca = (1 - dt*Ce/(2*eps0*epsilon_r)) / (1 + dt*Ce/(2*eps0*epsilon_r))
Cb = (dt/(eps0*epsilon_r*dx)) / (1 + dt*Ce/(2*eps0*epsilon_r))

# 8) Source gaussienne spatio-temporelle
source_y = Ny//2
source_x = pml_thickness + 10
t0_phys  = 100*dt
tau_phys =  30*dt
width_x  =  10

# 9) Affichage
# 9) Affichage
fig, ax = plt.subplots(figsize=(6,6))
plt.subplots_adjust(bottom=0.2)

# 9.1 Champ Ez
im = ax.imshow(Ez.T, origin='lower', cmap='RdBu', vmin=-0.1, vmax=0.1)

# 9.2 Calque or semi-transparent
gold_layer = np.zeros_like(Ez)
gold_layer[mask_PEC] = 1.0
ax.imshow(
    gold_layer.T,
    origin='lower',
    cmap='Oranges',
    alpha=0.4,
    vmin=0, vmax=1
)
plt.colorbar(im, ax=ax, label='$E_z$')
ax.set_xlim(0,Nx)
ax.set_ylim(0,Ny)


# 10) Bouton Pause/Play
is_paused = [False]
def toggle_pause(event):
    is_paused[0] = not is_paused[0]
    pause_button.label.set_text("▶ Reprendre" if is_paused[0] else "⏸ Pause")

ax_btn = plt.axes([0.4,0.05,0.2,0.075])
pause_button = Button(ax_btn,"⏸ Pause")
pause_button.on_clicked(toggle_pause)

# 11) Animation FDTD
def update(n):
    global Ez, Hx, Hy
    if is_paused[0]:
        return [im]
    t = n*dt

    # injection de la source
    sp = np.exp(-((np.arange(Ny)-source_y)**2)/(2*width_x**2))
    env= np.exp(-((t-t0_phys)**2)/(2*tau_phys**2))
    Ez[source_x,:] += 0.2*sp*env*np.sin(omega0*t)

    # mise à jour H
    Hx[:, :-1] = Da[:, :-1]*Hx[:, :-1] - Db[:, :-1]*(Ez[:,1:]-Ez[:,:-1])
    Hy[:-1, :] = Ea[:-1, :]*Hy[:-1, :] + Eb[:-1, :]*(Ez[1:,:]-Ez[:-1,:])

    # mise à jour E
    Ez[1:,1:] = Ca[1:,1:]*Ez[1:,1:] + Cb[1:,1:]*(
        (Hy[1:,1:]-Hy[:-1,1:]) - (Hx[1:,1:]-Hx[1:,:-1])
    )

    # condition PEC (or) : Ez = 0 dans les blocs
    Ez[mask_PEC] = 0.0

    im.set_array(Ez.T)
    ax.set_title(f"Propagation à t = {t*1e15:.1f} fs")
    return [im]

ani = FuncAnimation(fig, update, frames=n_steps, interval=30, blit=False)
plt.show()
# ── Post-traitement : mesure des ordres ±1 au probe à ix_max + 75 ────────
# 1) Choix de la sonde
ix_probe = ix_max + 75
Ez_probe = Ez[ix_probe, :]      # profil Ez(y) à x = x_probe

# 2) Fenêtrage pour limiter les effets de bord
w = np.hanning(Ny)
Ew = Ez_probe * w

# 3) FFT spatiale en y
Efft = np.fft.fftshift(np.fft.fft(Ew))
Ifft = np.abs(Efft)**2

# 4) Conversion ky → θ
ky    = np.fft.fftshift(np.fft.fftfreq(Ny, d=dy)) * 2*np.pi
theta = np.degrees(np.arcsin(np.clip(ky/(2*np.pi/lambda0), -1, 1)))

# 5) Calcul de l’angle théorique pour m=±1
theta_th = np.degrees(np.arcsin(lambda0 / a))
print(f"Angle théorique m=±1 : ±{theta_th:.1f}°")

# 6) Recherche des indices les plus proches de ±θ_th
idx_m1 = np.argmin(np.abs(theta + theta_th))
idx_p1 = np.argmin(np.abs(theta - theta_th))

# 7) Affichage des résultats
print(f"Ordre -1 mesuré : θ = {theta[idx_m1]:.1f}°,  I = {Ifft[idx_m1]:.3e}")
print(f"Ordre +1 mesuré : θ = {theta[idx_p1]:.1f}°,  I = {Ifft[idx_p1]:.3e}")

# 8) Tracé pour vérification
plt.figure(figsize=(5,4))
plt.plot(theta, Ifft/np.max(Ifft), '-k')
plt.axvline( theta_th, color='r', linestyle='--', label=f'+{theta_th:.1f}° théorique')
plt.axvline(-theta_th, color='r', linestyle='--', label=f'-{theta_th:.1f}° théorique')
plt.scatter([theta[idx_m1], theta[idx_p1]],
            [Ifft[idx_m1]/np.max(Ifft), Ifft[idx_p1]/np.max(Ifft)],
            color='b', zorder=5)
plt.xlim(-60,60)
plt.xlabel(r'$\theta$ (°)')
plt.ylabel('I(θ) normalisée')
plt.legend()
plt.title("Spectre de diffraction — ordres ±1")
plt.grid(True)
plt.show()

