import numpy as np

def open_loop_read_noise(num_subapertures, detector_variance, read_noise_sensitivity, photon_flux):
	return num_subapertures * detector_variance / (read_noise_sensitivity**2 * photon_flux**2)

def open_loop_photon_noise(photon_noise_sensitivity, photon_flux):
	return  1 / (photon_noise_sensitivity**2 * photon_flux)