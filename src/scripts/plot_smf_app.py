from hcipy import *
import numpy as np
from matplotlib import pyplot as plt
import paths

#import matplotlib as mpl
#mpl.use('MacOSX')

# Set up the grids
grid = make_pupil_grid(256, 1.1)
aperture = evaluate_supersampled(make_circular_aperture(1), grid, 4)

focal_grid = make_focal_grid(q=15, num_airy=6)
prop = FraunhoferPropagator(grid, focal_grid)

# Setup the coronagraph
app_phase = read_field(str(paths.data/'smf_app_phase_example.fits'))
app = PhaseApodizer(app_phase)

separation = 2.5
mfd = 1.4
stellar_smf = SingleModeFiberInjection(focal_grid, make_gaussian_fiber_mode(mfd), position=np.array([separation, 0.0]))
photometric_aperture = evaluate_supersampled(make_circular_aperture(mfd, center=np.array([separation, 0.0])), focal_grid, 4)

num_waves = 151
bandwidth = 0.2
wavelengths = 1 + np.linspace(-bandwidth/2, bandwidth/2, num_waves)

# Make the wavefront
norm = 0
for wave in wavelengths:
	wf = Wavefront(aperture, wave)
	wf.total_power = 1
	norm += prop(wf).power.max()

# Simulate the optical systems
corim = 0
eta_star = 0
eta_wave = []
photo_star = []
for wave in wavelengths:
	wf = Wavefront(aperture, wave)
	wf.total_power = 1

	wf_app = app(wf)
	wf_foc = prop(wf_app)
	wf_smf = stellar_smf.backward(stellar_smf(wf_foc))
	
	mono_im = wf_foc.power / norm
	corim += mono_im
	
	photo_star.append(np.sum(mono_im * photometric_aperture))
	eta_wave.append(wf_smf.total_power)

# Plot the results

fig, axes = plt.subplots(1,4,figsize=(13, 3.6))
ax = np.ndarray.flatten(axes)

#imshow_field(app.phase, grid, vmin=-np.pi, vmax=np.pi, cmap='twilight',ax=ax[0])
imshow_field(app.phase, grid, vmin=-np.pi/4, vmax=np.pi/4, cmap='twilight',ax=ax[0])

imshow_field(wf_foc.phase, vmin=-np.pi, vmax=np.pi, cmap='twilight',ax=ax[1])
circ = plt.Circle((separation, 0.0), mfd/2, fill=False, lw=2, ls='--', color='white')
ax[1].add_patch(circ)
ax[1].set_xlim([separation - 2.5, separation + 2.5])
ax[1].set_ylim([-2.5, 2.5])

imshow_psf(corim, vmax=1, vmin=1e-6, colorbar=False, ax=ax[2])
circ = plt.Circle((separation, 0.0), mfd/2, fill=False, lw=2, ls='--', color='white')
ax[2].add_patch(circ)
ax[2].set_xlim([separation - 2.5, separation + 2.5])
ax[2].set_ylim([-2.5, 2.5])

ax[3].plot(wavelengths, eta_wave, 'C0', lw=2.5, label='SMF power')
ax[3].plot(wavelengths, photo_star, 'C3', lw=2.5, label='MMF power')
ax[3].set_ylim([1e-10, 1e-3])
ax[3].set_yscale('log')

plt.setp(ax[0],xticks=[],yticks=[])

ax[1].set_xlabel('$X [\lambda/D]$',fontsize=14)
ax[2].set_xlabel('$X [\lambda/D]$',fontsize=14)
ax[3].set_xlabel('$\lambda/\lambda_0$',fontsize=14)
ax[3].text(1.0,1e-4,"Multi Mode Fibre power",fontsize=12,color='red',fontweight='bold',ha='center',va='center')
ax[3].text(1.0,3e-7,"Single Mode Fibre power",fontsize=12,color='blue',fontweight='bold',ha='center',va='center')


ax[0].set_title('APP $[-\pi/4,\pi/4]$',fontsize=14)
ax[1].set_title('Focal Plane Phase',fontsize=14)
ax[2].set_title('Focal Plane Intensity',fontsize=14)
ax[3].set_title('Power',fontsize=14)

plt.subplots_adjust(left=0.05, right=0.95, bottom=0.2, top=0.9)
plt.savefig(paths.figures/'smf_app.pdf')
#plt.show()
