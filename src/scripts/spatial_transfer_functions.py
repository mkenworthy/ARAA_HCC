import hcipy
import numpy as np

from utils import jinc

def ideal_dm_transfer_function(number_of_actuators, telescope_diameter):
	def func(grid):
		return 1 - hcipy.make_rectangular_aperture(2 * np.pi * number_of_actuators / telescope_diameter)(grid)
	return func

def piston_subtraction_transfer_function(telescope_diameter):
	def func(grid):
		return 1 - jinc(telescope_diameter / (2 * np.pi))(grid)
	return func