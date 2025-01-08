import hcipy
import numpy as np

class AtmosphericTurbulence():

	def __init__(self, Cn2_profile, velocities, heights, r0=0.16, L0=50.0):
		
		self._Cn2_profile = Cn2_profile / np.sum(Cn2_profile)
		self._velocities = velocities
		self._heights = heights

		self.r0 = r0
		self.L0 = L0

	@property
	def r0(self):
		return self._r0
	
	@r0.setter
	def r0(self, new_r0):
		if new_r0 > 0:
			self._r0 = new_r0

	@property
	def L0(self):
		return self._L0
	
	@L0.setter
	def L0(self, new_L0):
		if new_L0 > 0:
			self._L0 = new_L0

	@property
	def heights(self):
		return self._heights
	
	@property
	def velocities(self):
		return self._velocities

	def layer_psds(self, wavelength):
		def func(grid):
			# Calcuate the correct amount of r0 per layer
			integrated_cn_squared = hcipy.Cn_squared_from_fried_parameter(self.r0, 550e-9)
			r0_at_wavelength = hcipy.fried_parameter_from_Cn_squared(integrated_cn_squared, wavelength)
			layered_r0 = r0_at_wavelength * (self._Cn2_profile)**(-3.0/5.0)
			# Add the PSDs from each layer
			layer_psds = hcipy.Field([hcipy.power_spectral_density_von_karman(r0_i, self.L0)(grid) for r0_i in layered_r0], grid)
			return layer_psds
		return func

	def integrated_psd(self, wavelength):
		def func(grid):
			return np.sum(self.layer_psds(wavelength)(grid), axis=0)
		return func
	

# r0=0.16, wavelength=550e-9
def make_armazones_atmospheric_layers(quantile='Q2'):

	heights = np.array([30, 90, 150, 200, 245, 300, 390, 600, 1130, 1880, 2630,
					3500, 4500, 5500, 6500, 7500, 8500, 9500, 10500, 11500,
					12500, 13500, 14500, 15500, 16500, 17500, 18500, 19500,
					20500, 21500, 22500, 23500, 24500, 25500, 26500])
	

	if quantile == 'Q1':
		r0 = 0.22
	elif quantile == 'Q2':
		r0 = 0.17
	else:
		r0 = 0.14

	integrated_cn_squared = hcipy.Cn_squared_from_fried_parameter(r0, wavelength=500e-9)

	Cn_squared = np.array([22.6, 11.2, 10.1, 6.4, 4.15, 4.15, 4.15, 4.15, 3.1, 2.26, 1.13,
			2.21, 1.33, 0.88, 1.47, 1.77, 0.59, 2.06, 1.92, 1.03, 2.3, 3.75,
			2.76, 1.43, 0.89, 0.58, 0.36, 0.31, 0.27, 0.2, 0.16, 0.09, 0.12,
			0.07, 0.06])
	Cn_squared *= integrated_cn_squared / np.sum(Cn_squared)

	velocities = 10.0 * np.ones_like(Cn_squared)
	velocities[0] = 6.4
	velocities[-3] = 25.0
	velocities[-2] = 40.0
	velocities[-1] = 20.0

	theta = np.deg2rad(80.0) + np.random.uniform(-np.pi/4, np.pi/4, size=velocities.size)
	velocities = np.array([np.cos(theta), np.sin(theta)]) * velocities

	return AtmosphericTurbulence(Cn_squared, velocities, heights)

def make_lco_atmospheric_layers(quantile='Q2'):
	'''Creates the different LCO atmospheric layers.

	Parameters 
	-----------

	Returns 
	-----------
	
	'''
	heights = np.array([250., 500., 1000., 2000., 4000., 8000., 16000.])
	velocities = np.array([10., 10., 20., 20., 25., 30., 25.])

	if quantile == 'Q1':
		r0 = 0.22
		velocities *= (9.4/18.7)**(3/5)
	elif quantile == 'Q2':
		r0 = 0.17
	else:
		r0 = 0.14
		velocities *= (23.4/18.7)**(3/5)

	integrated_cn_squared = hcipy.Cn_squared_from_fried_parameter(r0, wavelength=500e-9)
	Cn_squared = np.array([0.42, 0.03, 0.06, 0.16, 0.11, 0.10, 0.12]) * integrated_cn_squared

	theta = np.random.uniform(0, 2 * np.pi, size=velocities.size)
	velocities = np.array([np.cos(theta), np.sin(theta)]) * velocities

	return AtmosphericTurbulence(Cn_squared, velocities, heights, r0)

def make_simple_atmospheric_layers(quantile='Q2'):

	heights = np.array([30, 10500, 25500])
	Cn_squared = np.array([0.7, 0.2, 0.1])

	velocities = 10.0 * np.ones_like(Cn_squared)
	velocities[0] = 15.0
	velocities[1] = 40.0
	velocities[2] = 20.0

	theta = np.random.uniform(0, 2 * np.pi, size=velocities.size)
	velocities = np.array([np.cos(theta), np.sin(theta)]) * velocities

	return AtmosphericTurbulence(Cn_squared, velocities, heights)


def make_single_atmospheric_layers(r0, velocity):
	velocities = np.array([[velocity, 0.0],])
	return AtmosphericTurbulence(np.array([1.0,]), velocities, np.array([0.0,]), r0)