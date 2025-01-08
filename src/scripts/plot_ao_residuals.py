import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches

from hcipy import *
import psfsim as ex

import paths

def wave2erg(wavelength):
	return 1.986e-12 * 1e-6 / wavelength


def jansky2photons(flux, wavelength, bandwidth):
	#erg
	energy_per_photon = wave2erg(wavelength)
	
	# bandwidth conversion from delta_m to delta_Hz
	c = 299792.0e3 # m/s
	nu_0 = c / (wavelength - bandwidth/2)
	nu_1 = c / (wavelength + bandwidth/2)
	frequency_bandwidth = abs(nu_0 - nu_1)

	return flux * 1e-23 * frequency_bandwidth / energy_per_photon
	

# Aperture
telescope_diameter = 39.4
aperture_function = make_elt_aperture()

dgrid = 2. * telescope_diameter
num_otf_pixels = 1024

# Star parameters
wavelength = 765.e-9
magnitude = 8

# AO system parameters
nact = 128
nact_ncpc = 60
integration_time = 1 / 2000.
delay = 250e-6
nsubap = 4 * (1.2 * nact)**2 * np.pi/4
readnoise = 0.1
photon_sensitivity = 1
read_noise_sensitivity = 1
AO_throughput = 0.3

# Derived parameters
lamD = wavelength / telescope_diameter

# Science camera sampling
focal_plane_q = 5
field_of_view = 150

# The pupil grid and the apertures
grid = make_pupil_grid(num_otf_pixels, dgrid)
aperture = evaluate_supersampled(aperture_function, grid, 4)
aperture /= np.sqrt(np.sum(aperture**2) * grid.weights)

# zeropoint flux
telescope_area = np.sum(aperture / aperture.max() * grid.weights)
zeropoint_flux = jansky2photons(2432.84, 790.0e-9, 149e-9) * telescope_area * 1e4

# AO system
atmosphere = ex.make_single_atmospheric_layers(0.16, 18.7 * 2)
#atmosphere = ex.make_armazones_atmospheric_layers('Q2')

pitch = telescope_diameter / nact
dm = ex.DeformableMirror(nact, pitch)
wfs = ex.WavefrontSensor(integration_time, nsubap, readnoise, photon_sensitivity, read_noise_sensitivity)
controller = ex.IntegralController(integration_time, delay)
ao = ex.AdaptiveOptics(grid, delay, dm, wfs, controller)

# System
photon_flux = zeropoint_flux * integration_time * 10**(-magnitude / 2.5) * AO_throughput
ao.optimize(atmosphere, wavelength, photon_flux)

# The coronagraph
cor_pupil_grid = make_pupil_grid(256, 1.1 * telescope_diameter)
focal_grid = make_focal_grid(q=focal_plane_q, num_airy=field_of_view / 2, spatial_resolution=lamD)
sci_cor_prop = FraunhoferPropagator(cor_pupil_grid, focal_grid)

# Simulate the coronagraph residuals
cor_aperture = evaluate_supersampled(aperture_function, cor_pupil_grid, 4)
sa = SurfaceAberration(cor_pupil_grid, 0.05 * wavelength, telescope_diameter)
ff = FourierFilter(cor_pupil_grid, lambda t: make_circular_aperture(2 * np.pi * nact_ncpc / telescope_diameter)(t))

forward_fft = make_fourier_transform(cor_pupil_grid, ff.internal_grid)
backward_fft = make_fourier_transform(ff.internal_grid, grid)

low_order_ncpa = np.real(ff.forward(sa.surface_sag + 0j))
high_order_ncpa = sa.surface_sag - low_order_ncpa
high_order_ncpa *= 1/2 * 25e-9 / np.std(high_order_ncpa[cor_aperture>0])
low_order_ncpa *= 1/2 * 1e-9 / np.std(low_order_ncpa[cor_aperture>0])
sa.surface_sag = high_order_ncpa + low_order_ncpa

upscaled_surface_sag = (aperture / aperture.max()) * np.real( backward_fft.forward( forward_fft.forward( cor_aperture * sa.surface_sag + 0j) ) )
upscaled_surface_sag = upscaled_surface_sag.shaped[::-1, ::-1].ravel()
upscaled_surface_sag *= np.std(sa.surface_sag[cor_aperture>0]) / np.std(upscaled_surface_sag[aperture>0])

# Coronagraph
switch_radius = 2.0 * lamD
perfect_coronagraph = OpticalSystem([sa, PerfectCoronagraph(cor_aperture, order=4), sci_cor_prop])
imager = OpticalSystem([sa, sci_cor_prop])

####
telescope = ex.Telescope(aperture * np.exp(1j * 4 * np.pi / wavelength * upscaled_surface_sag), telescope_diameter)
cor_imager = ex.CoronagraphicImager(focal_grid, grid, cor_pupil_grid, switch_radius, cor_aperture, perfect_coronagraph, sci_cor_prop)
sci_imager = ex.CoronagraphicImager(focal_grid, grid, cor_pupil_grid, switch_radius, cor_aperture, imager, sci_cor_prop)

hci_cor = ex.HighContrastImager(telescope, atmosphere, ao, cor_imager)
hci_sci = ex.HighContrastImager(telescope, atmosphere, ao, sci_imager)

low_pass_res, high_pass_res, total_residual = hci_cor.psf(wavelength, jitter_rms=None)
low_pass_res, high_pass_res, psf = hci_sci.psf(wavelength, jitter_rms=None)
angle = np.array([1 * lamD, 0.0])
low_pass_res, high_pass_res, planet_psf = hci_cor.psf(wavelength, angle, jitter_rms=None)

fig, axes = plt.subplots(1,2,figsize=(12,4.7))
imshow_psf(psf, vmax=1, vmin=1e-6, spatial_resolution=lamD, ax=axes[0])
axes[0].set_xlabel(r'x ($\lambda / D$)')
axes[0].set_ylabel(r'y ($\lambda / D$)')

imshow_psf(total_residual, vmax=1e-3, vmin=1e-6, spatial_resolution=lamD, ax=axes[1])
axes[1].set_xlabel(r'x ($\lambda / D$)')
axes[1].set_ylabel(r'y ($\lambda / D$)')

axes[1].text(0,11,"Wind driven\nhalo",color='white',fontsize=12,
		ha='center',va='center')

axes[1].text(0,40,"Uncorrected NCPA",color='white',fontsize=12,
		ha='center',va='center')

axes[1].text(0,68,"Free Atmosphere",color='white',fontsize=12,
		ha='center',va='center')

axes[1].text(-40,-40,"NCPA correction\nradius",color='white',fontsize=12,
		ha='center',va='center')

axes[1].text(0,-66,"DM control radius",color='white',fontsize=12,
		ha='center',va='center')

# axes[1].plot((0,0),(-5,-35),linewidth=1,color='white')
# axes[1].plot((0,0),(-46,-60),linewidth=1,color='white')

axes[1].plot((0,0),(-5,-60),linewidth=1,color='white')

axes[1].plot((63,63),(0,-70),linewidth=1,color='white')

axes[1].plot((30,30),(0,-50),linewidth=1,color='white')

ar = patches.Arrow(18,-40,12,0,width=5.0,color='white')
axes[1].add_patch(ar)
ar = patches.Arrow(50,-66,12,0,width=5.0,color='white')
axes[1].add_patch(ar)


plt.savefig(paths.figures/'postao_psf_and_coronagraph_image.pdf', bbox_inches='tight')
