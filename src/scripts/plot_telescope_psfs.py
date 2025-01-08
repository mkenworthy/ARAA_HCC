from hcipy import *
import numpy as np
from matplotlib import pyplot as plt
import paths


# import matplotlib as mpl
# mpl.use('MacOSX')
aperture_list = [
    make_elt_aperture,
    make_gmt_aperture,
    make_tmt_aperture,
    make_magellan_aperture,
    make_luvoir_a_aperture,
    make_luvoir_b_aperture,
#    make_hale_aperture,
    make_vlt_aperture,
#    make_habex_aperture,
#    make_hst_aperture,
    make_jwst_aperture,
    make_keck_aperture
]

telescope_names = [
    'ELT',
    'GMT',
    'TMT',
    'Magellan',
    'Luvoir - A',
    'Luvoir - B',
#    'Hale',
    'VLT',
#    'Habex',
#    'HST',
    'JWST',
    'Keck'
]



# We make the grids on which the simulation is defined.
pupil_grid = make_pupil_grid(128)

# We want to sample from -16 lam/D to 16 lam/D with 5 pixels per lam/D.
focal_grid = make_focal_grid(5, 16)

prop = FraunhoferPropagator(pupil_grid, focal_grid)


fig, axes = plt.subplots(3,6,figsize=(6,3))

aaa = np.ndarray.flatten(axes)

for i in range(len(aperture_list)):
  ap = aperture_list[i]
  label = telescope_names[i]

  telescope_aperture = evaluate_supersampled(ap(normalized=True), pupil_grid, 4)
  wf = Wavefront(telescope_aperture)
  psf = prop(wf).power
  psf /= psf.max()
  aaa[2*i].set_yticks([])
  aaa[2*i].set_xticks([])
  aaa[2*i+1].set_yticks([])
  aaa[2*i+1].set_xticks([])


  imshow_field(telescope_aperture, cmap='gray',ax=aaa[2*i])

  imshow_psf(psf, vmax=1, vmin=1e-6,ax=aaa[2*i+1],colorbar=False)
  aaa[2*i+1].text(-14,10,label,color='white',fontsize=8)

plt.tight_layout()
plt.draw()
#plt.show()
plt.savefig(paths.figures/'telescope_psfs.pdf')
