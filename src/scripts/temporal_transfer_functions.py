import hcipy
import numpy as np

def window_function(sample_time):
	
	def func(frequencies):
		return np.sinc(sample_time * frequencies / np.pi)
	return func

def delay_transfer_function(delay):
	
	def func(frequencies):
		return np.exp(2 * np.pi * 1j * delay * frequencies)
	return func

def noise_rejection_transfer_function(modal_gain, integration_time, delay):
	
	def func(frequencies):

		s = 2 * np.pi * integration_time * frequencies
		system_delay = delay_transfer_function(-delay)(frequencies)
		integration_delay = delay_transfer_function(-integration_time)(frequencies)

		Hnom = -modal_gain * system_delay * s
		Hdenom = modal_gain * system_delay * (1 - integration_delay) - s**2
		
		H = np.zeros_like(Hnom, dtype=complex)
		H[s>0] = Hnom[s>0] / Hdenom[s>0]
		if modal_gain > 0:
			H[0] = 1.0
		else:
			H[0] = 0.0

		return H

	return func

def integrated_ntf(modal_gain, integration_time, delay):
	def func(num_samples):
		#np.linspace(0, integration_time /2 )
		pass
	return func


class IntegralController():
	''' The aniso_servo transfer functions depend on the particular controller!
	so I should make controllers. These will determine the actuall performance.
	'''
	def __init__(self, integration_time, delay):
		self._integration_time = integration_time
		self._delay = delay
			
		# The sample and hold delay from the WFS integration
		self._sh_delay_tf = delay_transfer_function((self._integration_time / 2 + self._delay))

		# Computational delay + DM reponse
		self._delay_tf = delay_transfer_function(self._integration_time)
		self._integration_tf = window_function(self._integration_time)

	def noise_transfer_function(self, gain):
		# What is a good amount of sampling? 2 samples per Hz?
		temporal_frequencies = np.linspace(0, 1/(2 * self._integration_time), 2 * 8196)
		Hn = noise_rejection_transfer_function(gain, self._integration_time, self._delay)(temporal_frequencies)
		return np.trapz(abs(Hn)**2, temporal_frequencies)

	def aniso_servo_transfer_function(self, gains, velocity=np.array([10., 0]), layer_height=0, offaxis_angle=np.array([0.0,0.0])):
		'''
		'''	
		
		def func(grid):
			# This might need to change
			modal_gains = gains * grid.ones()

			# Determine the spatial and temporal frequencies
			offaxis_frequency = grid.x * offaxis_angle[0] + grid.y * offaxis_angle[1]
			frequencies = (grid.x * velocity[0] + grid.y * velocity[1]) / (2 * np.pi)

			# Calculate the on-axis transfer functions
			sh_delay_tf = self._sh_delay_tf(frequencies)
			integration_tf = self._integration_tf(frequencies)
			delay_tf = self._delay_tf(frequencies)

			# Determine the off-axis transfer function effects
			offaxis_delay = delay_transfer_function(layer_height)(offaxis_frequency)
			
			Fas = hcipy.Field(np.ones_like(modal_gains, dtype=complex), grid)
			Fas *= offaxis_delay

			gain_mask = abs(modal_gains) > 1e-5
			Fas[~gain_mask] = 1.0
			Fas[gain_mask] -= (modal_gains * integration_tf * sh_delay_tf)[gain_mask] / (1 + (modal_gains * integration_tf * sh_delay_tf - delay_tf)[gain_mask])
			
			transfer_function = np.nan_to_num(abs(Fas)**2)

			return hcipy.Field(transfer_function, grid)
		
		return func

	def optimize_gains(self, environment):
		# TODO: add gain stability check!
		# For now we us a simple upper-bound on the gain.
		modal_gains = np.linspace(0.0, 0.6, 151)

