import hcipy
import numpy as np
from matplotlib import pyplot as plt

from noise_sources import open_loop_photon_noise, open_loop_read_noise

class HighContrastImager():

	def __init__(self, telescope, environment, adaptive_optics_system, science_instrument):
		self._telescope = telescope
		self._environment = environment
		self._ao_system = adaptive_optics_system

		self._science_instrument = science_instrument

	def vibration_OTF(self, vibration_rms, wavelength):
		''' A Gaussian blurring OTF
		'''
		grid = self._science_instrument._otf_grid
		lamD = wavelength / self._telescope._telescope_diameter

		r = grid.as_('polar').r
		r2 = r**2
		a = 0.5 * (2 * np.pi * lamD/vibration_rms)**2
		
		OTF = hcipy.Field(np.exp(-np.pi**2 * r2 / a), grid)
		OTF /= abs(OTF).max()

		return OTF

	def psf(self, wavelength, angle=np.array([0,0]), with_environment=True, jitter_rms=None):
		# Analytical model
		TelescopeOTF = self._telescope.otf()
		if with_environment:
			DisturbanceOTF = self._ao_system.otf(self._environment, wavelength)
			
			if jitter_rms is not None:
				DisturbanceOTF *= self.vibration_OTF(jitter_rms, wavelength)
		else:
			DisturbanceOTF = TelescopeOTF.grid.ones()

		return self._science_instrument.psf(TelescopeOTF, DisturbanceOTF, wavelength, angle)

class DeformableMirror():
	def __init__(self, num_actuators, pitch, use_circle=True):
		self._num_actuators = num_actuators
		self._pitch = pitch
		self._use_circle = use_circle

	def transfer_function(self, grid):
		if self._use_circle:
			return 1 - hcipy.evaluate_supersampled(hcipy.make_circular_aperture(2 * np.pi / self._pitch), grid, 4)
		else:
			return 1 - hcipy.evaluate_supersampled(hcipy.make_rectangular_aperture(2 * np.pi / self._pitch), grid, 4)

class WavefrontSensor():
	''' Highly idealized wavefront sensor.
	'''
	def __init__(self, integration_time, num_subapertures, detector_variance, photon_sensitivity=1, readnoise_sensitivity=1):
		self._integration_time = integration_time
		self._photon_sensitivity = photon_sensitivity
		self._readnoise_sensitivity = readnoise_sensitivity
		self._num_subaperture = num_subapertures
		self._detector_variance = detector_variance

		self._optical_gain = 1

	def optical_gain(self, new_optical_gain):
		self._optical_gain = new_optical_gain

	def open_loop_variance(self, photon_flux):

		variance = open_loop_photon_noise(self._photon_sensitivity, photon_flux)
		variance += open_loop_read_noise(self._num_subaperture, self._detector_variance, self._readnoise_sensitivity, photon_flux)
		
		return variance / self._optical_gain**2
	
class AdaptiveOptics():
	def __init__(self, otf_grid, delay, deformable_mirror, wavefront_sensor, controller):
		self._otf_grid = otf_grid
		self._fft_is_prepared = False
		self.prepare_fft(otf_grid)

		self._delay = delay
		self._deformable_mirror = deformable_mirror
		self._wavefront_sensor = wavefront_sensor
		self._controller = controller

		self._gains = 0

	def prepare_fft(self, otf_grid):
		if not self._fft_is_prepared:
			self._fft = hcipy.FastFourierTransform(otf_grid)
			self._spatial_frequencies = self._fft.output_grid
			self._fft_is_prepared = True

	def optimize(self, atmosphere, wavelength, photon_flux=1e7, filepath=None, gain_max=0.7):
		dm_transfer_function = self._deformable_mirror.transfer_function(self._spatial_frequencies)

		total_psd = self._spatial_frequencies.zeros()
		atmospheric_layers = atmosphere.layer_psds(wavelength)(self._spatial_frequencies)
		low_pass_filter = (1 - dm_transfer_function)
		low_pass_mask = low_pass_filter > 0

		subset_spatial_freqs = self._spatial_frequencies.subset(low_pass_mask)
		
		# Gain limits
		optimization_gains = np.linspace(0.0, gain_max, 11)
		gain_cube = np.outer(optimization_gains, subset_spatial_freqs.ones())
		aniso_error = self._spatial_frequencies.zeros((optimization_gains.size,))

		for i, (layer_psd, height, velocity) in enumerate(zip(atmospheric_layers, atmosphere.heights, atmosphere.velocities)):
			# The fitting error
			fitting_error = dm_transfer_function * layer_psd

			# Aniso-server error
			#aniso_transfer_function = self._controller.aniso_servo_transfer_function(gain_cube, velocity, height)(self._spatial_frequencies)
			#aniso_error = aniso_transfer_function * layer_psd
			aniso_transfer_function = self._controller.aniso_servo_transfer_function(gain_cube, velocity, height)(subset_spatial_freqs)
			aniso_error[:,low_pass_mask] = aniso_transfer_function * layer_psd[low_pass_mask]
			
			# The total error
			total_psd = total_psd + fitting_error + aniso_error

		ntfs = np.array([self._controller.noise_transfer_function(gain) for gain in optimization_gains])
		noise_psd = 2 * self._wavefront_sensor.open_loop_variance(photon_flux) * ntfs
		
		# A noise PSD for each gain
		total_psd = ((total_psd.T) + noise_psd).T
		
		if filepath is not None:
			hcipy.write_field(total_psd, filepath)

		mindx = np.argmin(total_psd[:, low_pass_mask], axis=0)
		self._gains = self._spatial_frequencies.zeros()
		self._gains[low_pass_mask] = optimization_gains[mindx]

		self._noise_psd = self._spatial_frequencies.zeros()
		self._noise_psd[low_pass_mask] = noise_psd[mindx]

	def set_gains(self, gains):
		spatial_low_pass_filter = 1 - self._deformable_mirror.transfer_function(self._spatial_frequencies)
		self._gains = gains * spatial_low_pass_filter

	def residual(self, atmosphere, wavelength):
		dm_transfer_function = self._deformable_mirror.transfer_function(self._spatial_frequencies)

		total_psd = self._spatial_frequencies.zeros()
		atmospheric_layers = atmosphere.layer_psds(wavelength)(self._spatial_frequencies)
		low_pass_filter = (1 - dm_transfer_function)
		low_pass_mask = low_pass_filter > 0
		subset_spatial_freqs = self._spatial_frequencies.subset(low_pass_mask)

		aniso_error = self._spatial_frequencies.zeros()

		for i, (layer_psd, height, velocity) in enumerate(zip(atmospheric_layers, atmosphere.heights, atmosphere.velocities)):
			# The fitting error
			fitting_error = dm_transfer_function * layer_psd

			# Aniso-server error
			aniso_transfer_function = self._controller.aniso_servo_transfer_function(self._gains[low_pass_mask], velocity, height)(subset_spatial_freqs)
			aniso_error[low_pass_mask] = aniso_transfer_function * layer_psd[low_pass_mask]
			
			# The total error
			total_psd += fitting_error + aniso_error
		
		total_psd += self._noise_psd 
		
		#print("Total PSD: ", 0.02 * np.sum(total_psd * self._spatial_frequencies.weights))
		
		return total_psd

	def otf(self, atmosphere, wavelength):
		''' input_grid is defined in OTF space.
		'''	
		psd_out = self.residual(atmosphere, wavelength)

		Bg = self._fft.backward(psd_out + 0j)
		index = self._otf_grid.closest_to([0,0])
		Dphi = np.real(2 * (Bg[index] - Bg) )

		# I still need to figure out the correct normalization factors.
		return hcipy.Field(np.exp(-Dphi / 2), self._otf_grid)
	
class PerfectAdaptiveOptics(AdaptiveOptics):
	def __init__(self, otf_grid, delay, deformable_mirror, wavefront_sensor, controller):
		super().__init__(otf_grid, delay, deformable_mirror, wavefront_sensor, controller)

	def optimize(self, atmosphere, wavelength, photon_flux=1e7, filepath=None, gain_max=0.7):
		self._gains = 1
		self._noise_psd = 0

	def residual(self, atmosphere, wavelength):
		dm_transfer_function = self._deformable_mirror.transfer_function(self._spatial_frequencies)

		total_psd = self._spatial_frequencies.zeros()
		atmospheric_layers = atmosphere.layer_psds(wavelength)(self._spatial_frequencies)
		
		for i, (layer_psd, height, velocity) in enumerate(zip(atmospheric_layers, atmosphere.heights, atmosphere.velocities)):
			# The fitting error
			fitting_error = dm_transfer_function * layer_psd		

			# The total error
			total_psd += fitting_error

		return total_psd
	
class Imager():
	def __init__(self, num_pixels, field_of_view, telescope_diameter, otf_input_grid):
		self._focal_grid = hcipy.make_pupil_grid(num_pixels, field_of_view)
		self._otf_grid = otf_input_grid
		self._telescope_diameter = telescope_diameter
		self._wavelengths = np.array([])
		self._propagators = []
	
	def get_monochromatic_propagator(self, wavelength):
		'''
		'''
		if self._wavelengths.size > 0:
			# First check if propagator already exists
			wavelength_distance = abs(self._wavelengths - wavelength) / wavelength
			wave_min_indx = np.argmin(wavelength_distance)
			if wavelength_distance[wave_min_indx] < 1e-6:
				return self._propagators[wave_min_indx]		
			else:
				chromatic_focal_grid = self._focal_grid.scaled(1 / wavelength)
				monochromatic_propagator = hcipy.make_fourier_transform(self._otf_grid, chromatic_focal_grid)

				self._wavelengths = np.append(self._wavelengths, wavelength)
				self._propagators.append(monochromatic_propagator)
				return self._propagators[-1]
		else:
			# Convert from radians to cycles per pupil.
			chromatic_focal_grid = self._focal_grid.scaled(2 * np.pi * 1 / wavelength)
			monochromatic_propagator = hcipy.make_fourier_transform(self._otf_grid, chromatic_focal_grid)

			self._wavelengths = np.array([wavelength,])
			self._propagators.append(monochromatic_propagator)
			return self._propagators[-1]

	def psf(self, telescope_OTF, atmospheric_OTF, wavelength):
		'''
		'''
		mono_propagator = self.get_monochromatic_propagator(wavelength)
		seeing_psf = abs(mono_propagator.forward(atmospheric_OTF * telescope_OTF + 0j).real)
		norm = abs(mono_propagator.forward(telescope_OTF + 0j).real).max()
		return hcipy.Field(seeing_psf / norm, self._focal_grid)

class CoronagraphicImager():
	def __init__(self, focal_grid, otf_grid, coronagraph_grid, switch_radius, aperture, coronagraph, science_propagator):
		self._focal_grid = focal_grid
		
		# coronagraph properties
		self._coronagraph_aperture = aperture
		self._coronagraph_grid = coronagraph_grid
		self._coronagraph = coronagraph
		self._science_propagator = science_propagator
		self._switch_radius = switch_radius

		# These are the propagators for the Fourier kernels
		self._otf_grid = otf_grid
		self._fft = hcipy.FastFourierTransform(self._otf_grid)
		self._freq_grid = self._fft.output_grid

		self._wavelengths = np.array([])
		self._propagators = []
	
	def get_monochromatic_propagator(self, wavelength):
		'''
		'''
		if self._wavelengths.size > 0:
			# First check if propagator already exists
			wavelength_distance = abs(self._wavelengths - wavelength) / wavelength
			wave_min_indx = np.argmin(wavelength_distance)
			if wavelength_distance[wave_min_indx] < 1e-6:
				return self._propagators[wave_min_indx]		
			else:
				chromatic_focal_grid = self._focal_grid.scaled(1 / wavelength)
				monochromatic_propagator = hcipy.make_fourier_transform(self._otf_grid, chromatic_focal_grid)

				self._wavelengths = np.append(self._wavelengths, wavelength)
				self._propagators.append(monochromatic_propagator)
				return self._propagators[-1]
		else:
			# Convert from radians to cycles per pupil.
			chromatic_focal_grid = self._focal_grid.scaled(2 * np.pi * 1 / wavelength)
			monochromatic_propagator = hcipy.make_fourier_transform(self._otf_grid, chromatic_focal_grid)

			self._wavelengths = np.array([wavelength,])
			self._propagators.append(monochromatic_propagator)
			return self._propagators[-1]

	def psf(self, telescope_OTF, atmospheric_OTF, wavelength, angle):
		'''
		'''
		# Calculate the power in all spatial frequenceis
		OTF_angle_tilt = np.exp(1j * 2 * np.pi / wavelength * (angle[0] * self._fft.input_grid.x + angle[1] * self._fft.input_grid.y) )
		wave_power = abs(np.real(self._fft.forward(atmospheric_OTF * OTF_angle_tilt)))

		# Define the low-pass and high-pass content
		self._low_pass_filter = hcipy.make_circular_aperture(2 * np.pi * 2 * self._switch_radius / wavelength)(self._freq_grid)
		self._high_pass_filter = 1 - self._low_pass_filter
		
		# Subselect the grid and points we want to fully propagate end-to-end
		new_grid = self._freq_grid.subset(self._low_pass_filter>0)
		wave_power_lp = wave_power[self._low_pass_filter>0] / np.sum(np.sum(wave_power))
	
		# Get the propagator
		low_pass_coronagraph_residuals = 0
		for i, p in enumerate(new_grid.points):
			phase_tilt = (self._coronagraph_grid.x * p[0] + self._coronagraph_grid.y * p[1])
			wf = hcipy.Wavefront(self._coronagraph_aperture * np.exp(1j * phase_tilt), wavelength)
			wf.total_power = wave_power_lp[i]
			low_pass_coronagraph_residuals += self._coronagraph(wf).power
		
		# Now let's get the high-pass filtered part
		mono_propagator = self.get_monochromatic_propagator(wavelength)
		high_pass_coronagraph_residuals = np.real( mono_propagator.forward(telescope_OTF * self._fft.backward(self._high_pass_filter * wave_power) ) )
		
		# And now let's measure the normalization constants
		# First let's estimate the high-pass normalization constant
		psf = np.real( mono_propagator.forward(self._fft.backward(self._fft.forward(telescope_OTF))) )
		high_pass_norm = psf.max()

		# And now the low-pass normalization constant
		wf = hcipy.Wavefront(self._coronagraph_aperture, wavelength)
		wf.total_power = 1
		ref_psf = self._science_propagator(wf).power
		low_pass_norm = ref_psf.max()
		
		# And now combine and return
		total_residual = hcipy.Field(low_pass_coronagraph_residuals / low_pass_norm + high_pass_coronagraph_residuals / high_pass_norm, ref_psf.grid)
		
		return low_pass_coronagraph_residuals / low_pass_norm, high_pass_coronagraph_residuals / high_pass_norm, total_residual
