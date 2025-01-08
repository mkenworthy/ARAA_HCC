import hcipy
import numpy as np
from scipy import special

def jinc(scale):
	def func(grid):
		r = grid.as_('polar').r
		mask = r > 0
		airy = grid.ones()
		airy[mask] = (2 * special.j1(np.pi * scale * r[mask]) / (np.pi * r[mask] * scale))
		return (airy / airy.max())**2
	return func

def generate_psd_grid(grid):
	# Todo: replace this a better function.
	# creating a FFT object just to get the grid is a bit overkill.
	fft = hcipy.FastFourierTransform(grid, q=1)
	return fft.output_grid	