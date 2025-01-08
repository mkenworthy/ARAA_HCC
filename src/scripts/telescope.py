import hcipy
import numpy as np

class Telescope():
	def __init__(self, aperture, telescope_diameter):
		self._telescope_diameter = telescope_diameter
		self._aperture = aperture
		self._grid = aperture.grid
		
		# Generate a FFT object
		self._fft = hcipy.FastFourierTransform(aperture.grid)

	@property
	def grid(self):
		return self._grid

	def otf(self):
		# TODO: figure out normalization but for now this works.
#		if self._grid.x.ptp() / self._telescope_diameter < 2:
		if np.ptp(self._grid.x) / self._telescope_diameter < 2:
			psf = self.psf(q=2)
		else:
			psf = self.psf()

		otf = self._fft.backward(psf + 0j) * self._grid.weights
		return otf
	
	def psf(self, q=None):
		if q is not None:
#			current_sampling = (self._grid.x.ptp() / self._telescope_diameter)
			current_sampling = (np.ptp(self._grid.x) / self._telescope_diameter)
			oversampling = q / current_sampling
			if oversampling > 1:
				self._fft = hcipy.FastFourierTransform(self._grid, q=oversampling)
		
		# The 2pi normalization is the standard required normalization for the FFT!	
		psf = np.abs(self._fft.forward(self._aperture + 0j) / (2 * np.pi))**2 * self._fft.output_grid.weights
		return psf