from hcipy import *
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec
import paths

# Make the grids
grid = make_pupil_grid(512, 1.1)
aperture = evaluate_supersampled(make_circular_aperture(1), grid, 4)

lyot_mask = evaluate_supersampled(make_circular_aperture(0.95), grid, 4)
lyot_stop = Apodizer(lyot_mask)

# focal plane parameters
focal_grid = make_focal_grid(q=5, num_airy=15)
prop = FraunhoferPropagator(grid, focal_grid)

# Make the wavefront
wf = Wavefront(aperture)
wf.total_power = 1
norm = prop(wf).power.max()

aplc_amp = read_field(str(paths.data/'aplc_amplitude_example.fits'))
pre_apodizer = Apodizer(aplc_amp)

# Make the focal plane mask
fpm_diameter = 6
cor_grid = make_focal_grid(q=25, num_airy=1.1 * fpm_diameter / 2)
focal_plane_mask = 1 - evaluate_supersampled(make_circular_aperture(fpm_diameter), cor_grid, 4)

lyot_coronagraph = LyotCoronagraph(grid, focal_plane_mask, lyot_stop)

#
tiptilt = TipTiltMirror(grid)

#
wf_apodized = pre_apodizer(wf)

# Make the throughput curve
tilts = np.linspace(0, 8, 41)
throughput = np.zeros_like(tilts)
for i, tilt in enumerate(tilts):
	tiptilt.actuators[0] = tilt / 2
	wf_foc = prop(lyot_coronagraph(tiptilt(wf_apodized)))

	# Make a photometric aperture of 2 lambda/D in diameter centered on the off-axis planet
	photometric_aperture = evaluate_supersampled(make_circular_aperture(2, center=[tilt, 0]), focal_grid, 4)
	throughput[i] = np.sum(photometric_aperture * wf_foc.power)

# find off-axis maximum throughput == no focal plane mask
wf_foc = prop(lyot_stop(wf_apodized))
photometric_aperture = evaluate_supersampled(make_circular_aperture(2), focal_grid, 4)
maximum_throughput = np.sum(photometric_aperture * wf_foc.power)

# find inner-working angle
iwa = np.interp(0.5 * maximum_throughput, throughput, tilts)
iwa_throughput = 0.5 * maximum_throughput

# Make representative PSFs
psf_tilts = np.array([0.0, 2.5, 3.5, 5.0])
throughput_psfs = np.zeros_like(psf_tilts)
psf_cube = focal_grid.zeros((4,))
for i, tilt in enumerate(psf_tilts):
	tiptilt.actuators[0] = tilt / 2
	wf_foc = prop(lyot_coronagraph(tiptilt(wf_apodized)))

	# Make a photometric aperture of 2 lambda/D in diameter centered on the off-axis planet
	photometric_aperture = evaluate_supersampled(make_circular_aperture(2, center=[tilt, 0]), focal_grid, 4)
	throughput_psfs[i] = np.sum(photometric_aperture * wf_foc.power)
	psf_cube[i] = wf_foc.power / norm


# Plot the results
#fig = plt.figure(constrained_layout=True, figsize=(7.5,7))
fig = plt.figure(figsize=(8,4.5))
gs = gridspec.GridSpec(ncols=4, nrows=2, figure=fig)	

for ti in range(psf_tilts.size):
	ax = fig.add_subplot(gs[0, ti])
	imshow_psf(psf_cube[ti] / psf_cube.max(), vmin=1e-7, vmax=1, cmap='inferno', ax=ax, colorbar=False)

	circle_ao = plt.Circle((psf_tilts[ti], 0), 1, color='white', ls=':', lw=1.5, fill=False)
	plt.gca().add_patch(circle_ao)

	plt.xlim([-7, 7])
	plt.ylim([-7, 7])
	plt.xlabel(r'x ($\lambda / D$)')
	if ti == 0:
		plt.ylabel(r'y ($\lambda / D$)')

ax = fig.add_subplot(gs[1, :])
plt.axhline(maximum_throughput, color='C0', lw=2, ls=':')
ax.plot(iwa, iwa_throughput, 'C0o', ms=8)
ax.plot([iwa, iwa], [0, iwa_throughput], lw=2, color='C0', ls=':')

ax.plot(tilts, throughput, lw=2)
ax.plot(psf_tilts, throughput_psfs, 'C1o')

plt.ylim([0.0, 0.62])
plt.ylabel('planet throughput', fontsize=14)
plt.xlabel(r'angular separation ($\lambda / D$)', fontsize=13)

plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.12)

plt.savefig(paths.figures / 'planet_throughput.pdf')