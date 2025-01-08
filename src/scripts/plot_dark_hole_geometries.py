from hcipy import *
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches
import paths

def add_box(ax, position, size):
	x0 = position[0] - size[0]/2
	y0 = position[1] - size[1]/2

	rect = patches.Rectangle((x0, y0), size[0], size[1], fill=False, lw=2, ls='--', color='white')
	ax.add_patch(rect)

def add_circular_darkhole(ax, iwa, owa):

	iwa_circ = patches.Circle((0, 0), iwa, fill=False, lw=2, ls='--', color='white')
	ax.add_patch(iwa_circ)
	
	owa_circ = patches.Circle((0, 0), owa, fill=False, lw=2, ls='--', color='white')
	ax.add_patch(owa_circ)

	# add definitons of IWA and OWA
#	line1 = patches.Line2D((0,2),)
	ax.text(iwa/2,7,"IWA",color='white',fontsize=12,
		ha='center',va='center')
	ax.text(owa/2,-5,"OWA",color='white',fontsize=12,
		ha='center',va='center')

	iwa_leng = 3
	iwa_offset = 5
	ar = patches.Arrow(iwa+iwa_leng,iwa_offset,-iwa_leng,0,width=1.0,color='white')
	ax.add_patch(ar)
	ar = patches.Arrow(-iwa_leng,iwa_offset,iwa_leng,0,width=1.0,color='white')
	ax.add_patch(ar)

	ax.plot((0,0),(0,6),linewidth=1,color='white')
	ax.plot((iwa,iwa),(0,6),linewidth=1,color='white')


	owa_leng = 3
	owa_offset = -5
	ar = patches.Arrow(owa+owa_leng,owa_offset,-owa_leng,0,width=1.0,color='white')
	ax.add_patch(ar)
	ar = patches.Arrow(-owa_leng,owa_offset,owa_leng,0,width=1.0,color='white')
	ax.add_patch(ar)


	ax.plot((0,0),(0,-8),linewidth=1,color='white')
	ax.plot((owa,owa),(0,-8),linewidth=1,color='white')


def add_d_shaped_darkhole(ax, iwa, owa):
	x = np.linspace(iwa, owa, 128)
	y = np.linspace(-owa, owa, 128)

	x_iwa = iwa * np.ones_like(x)

	theta = np.linspace(0, 2 * np.pi, 512)
	x_owa = owa * np.sin(theta)
	y_owa = owa * np.cos(theta)
	
	y_new = y_owa[x_owa > iwa]
	x_new = x_owa[x_owa > iwa]

	x_full = np.concatenate( (x_iwa, x_new) )
	y_full = np.concatenate( (y, y_new) )

	ax.plot(x_full, y_full, '--', color='white', lw=2)
	
def add_180_darkhole(ax, iwa, owa):
	theta = np.linspace(0, 2 * np.pi, 512)
	x_owa = owa * np.sin(theta)
	y_owa = owa * np.cos(theta)

	y_outer_arc = y_owa[x_owa > 0]
	x_outer_arc = x_owa[x_owa > 0]

	x_iwa = iwa * np.sin(theta)
	y_iwa = iwa * np.cos(theta)

	y_inner_arc = y_iwa[x_iwa > 0]
	x_inner_arc = x_iwa[x_iwa > 0]
	
	ax.plot(x_inner_arc, y_inner_arc, '--', color='white', lw=2)
	ax.plot(x_outer_arc, y_outer_arc, '--', color='white', lw=2)

	y_vertical = np.linspace(iwa, owa, 128)
	ax.plot(np.zeros_like(y_vertical), y_vertical, '--', color='white', lw=2)
	ax.plot(np.zeros_like(y_vertical), -y_vertical, '--', color='white', lw=2)

#path_to_datafiles = './Data/'

grid = make_pupil_grid(256, 1.1)
aperture = evaluate_supersampled(make_circular_aperture(1), grid, 4)

focal_grid = make_focal_grid(q=15, num_airy=15)
prop = FraunhoferPropagator(grid, focal_grid)
target_contrast = 1e-5

app = Apodizer(grid.ones() * (aperture>0))

# Make the wavefront
wf = Wavefront(aperture)
wf.total_power = 1
norm = prop(wf).power.max()

# The dark hole geometries
iwa = 2.0
owa = 10.
d_shaped_dark_hole_mask = make_circular_aperture(2 * owa)(focal_grid) - make_circular_aperture(2 *  iwa)(focal_grid)
d_shaped_dark_hole_mask *= focal_grid.x > iwa

iwa = 3.0
owa = 10.
annular_dark_hole_mask = make_circular_aperture(2 * owa)(focal_grid) - make_circular_aperture(2 *  iwa)(focal_grid)
annular_dark_hole_mask *= focal_grid.x > 0

fig = plt.figure(figsize=(12,4))
uses_phase = [False, True, True, True]
dark_hole_labels = ['circular', 'annular', 'dshaped', 'rectangular']
dark_hole_titles = ['360 degree', '180 degree', 'D shaped', 'Rectangular']
for di, (label,ti) in enumerate(zip(dark_hole_labels,dark_hole_titles)):
	if uses_phase[di]:
		dark_hole_phase = read_field(str(paths.data / 'app_phase_{:s}_example.fits'.format(label)))
		app.apodization = np.exp(1j * dark_hole_phase)
	else:
		dark_hole_amplitude = read_field(str(paths.data / 'spp_amplitude_example.fits'))
		app.apodization = dark_hole_amplitude

	wf_app = app(wf)
	wf_foc = prop(wf_app)
	corim = wf_foc.power / norm
	
	plt.subplot(1, 4, 1 + di)
	imshow_psf(corim, vmax=1, vmin=target_contrast / 10, colorbar=False, cmap='inferno')
	plt.title(ti)

	if label == 'rectangular':
		add_box(plt.gca(), [6, 0], [8, 20])
	elif label == 'circular':
		add_circular_darkhole(plt.gca(), 3.0, 10.0)
	elif label == 'dshaped':
		add_d_shaped_darkhole(plt.gca(), 2.0, 10.0)
	elif label == 'annular':
		add_180_darkhole(plt.gca(), 3.0, 10.0)

fig.supxlabel('$X [\lambda/D]$',fontsize=20)
fig.supylabel('$Y [\lambda/D]$',fontsize=20)
plt.tight_layout()
plt.savefig(paths.figures / 'dark_hole_geometries.pdf')
		
	
