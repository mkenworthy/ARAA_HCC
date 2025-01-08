from hcipy import *
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import paths

# import matplotlib as mpl
# mpl.use('MacOSX')

# Input parameters
pupil_diameter = 7e-3 # m
wavelength = 700e-9 # m
focal_length = 500e-3 # m

num_actuators_across = 32
actuator_spacing = 1.05 / 32 * pupil_diameter
aberration_ptv = 0.02 * wavelength # m

epsilon = 1e-9

spatial_resolution = focal_length * wavelength / pupil_diameter
iwa = 2 * spatial_resolution
owa = 12 * spatial_resolution
offset = 1 * spatial_resolution

efc_loop_gain = 0.5
num_iterations = 50

# Create grids
pupil_grid = make_pupil_grid(256, pupil_diameter * 1.1)
focal_grid = make_focal_grid(q=3, num_airy=16, spatial_resolution=spatial_resolution)
prop = FraunhoferPropagator(pupil_grid, focal_grid, focal_length)

# Create aperture and dark zone
aperture = Field(np.exp(-(pupil_grid.as_('polar').r / (0.5 * pupil_diameter))**30), pupil_grid)

dark_zone = make_circular_aperture(2 * owa)(focal_grid)
dark_zone -= make_circular_aperture(2 * iwa)(focal_grid)
dark_zone *= focal_grid.x > offset
dark_zone = dark_zone.astype(bool)

# Create optical elements
coronagraph = PerfectCoronagraph(aperture, order=6)

tip_tilt = make_zernike_basis(3, pupil_diameter, pupil_grid, starting_mode=2)
aberration = SurfaceAberration(pupil_grid, aberration_ptv, pupil_diameter, remove_modes=tip_tilt, exponent=-3)

aberration_distance = 100e-3 # m
aberration_at_distance = SurfaceAberrationAtDistance(aberration, aberration_distance)

influence_functions = make_xinetics_influence_functions(pupil_grid, num_actuators_across, actuator_spacing)
deformable_mirror = DeformableMirror(influence_functions)

#
def get_image(actuators=None, include_aberration=True):
    if actuators is not None:
        deformable_mirror.actuators = actuators

    wf = Wavefront(aperture, wavelength)
    if include_aberration:
        wf = aberration_at_distance(wf)

    img = prop(coronagraph(deformable_mirror(wf)))

    return img

img_ref = prop(Wavefront(aperture, wavelength)).power

def get_jacobian_matrix(get_image, dark_zone, num_modes):
    responses = []
    amps = np.linspace(-epsilon, epsilon, 2)

    for i, mode in enumerate(np.eye(num_modes)):
        response = 0

        for amp in amps:
            response += amp * get_image(mode * amp, include_aberration=False).electric_field

        response /= np.var(amps)
        response = response[dark_zone]

        responses.append(np.concatenate((response.real, response.imag)))

    jacobian = np.array(responses).T
    return jacobian

jacobian = get_jacobian_matrix(get_image, dark_zone, len(influence_functions))

def run_efc(get_image, dark_zone, num_modes, jacobian, rcond=1e-2):
    # Calculate EFC matrix
    efc_matrix = inverse_tikhonov(jacobian, rcond)

    # Run EFC loop
    current_actuators = np.zeros(num_modes)

    actuators = []
    electric_fields = []
    images = []

    for i in range(num_iterations):
        img = get_image(current_actuators)

        electric_field = img.electric_field
        image = img.power

        actuators.append(current_actuators.copy())
        electric_fields.append(electric_field)
        images.append(image)

        x = np.concatenate((electric_field[dark_zone].real, electric_field[dark_zone].imag))
        y = efc_matrix.dot(x)

        current_actuators -= efc_loop_gain * y

    return actuators, electric_fields, images

actuators, electric_fields, images = run_efc(get_image, dark_zone, len(influence_functions), jacobian)

def make_animation_1dm(actuators, electric_fields, images, dark_zone):
    anim = FFMpegWriter('efc_closed_loop_video.mp4', framerate=5)

    average_contrast = [np.mean(image[dark_zone] / img_ref.max()) for image in images]

    for i in range(num_iterations):
        plt.clf()

        plt.subplot(1, 2, 1)
        plt.title('Intensity image')
        imshow_psf(images[i] / img_ref.max(), spatial_resolution=spatial_resolution, cmap='inferno', vmin=1e-10, vmax=1e-5)
        contour_field(dark_zone, grid_units=spatial_resolution, levels=[0.5], colors='white')
        plt.xlabel(r'x ($\lambda / D$)')
        plt.ylabel(r'y ($\lambda / D$)')

        plt.subplot(1, 2, 2)
        plt.title('Average contrast')
        plt.plot(range(i), average_contrast[:i], 'o-')
        plt.xlim(0, num_iterations)
        plt.yscale('log')
        plt.ylim(1e-11, 1e-5)
        plt.grid(color='0.5')
        plt.xlabel('iterations')

        anim.add_frame()

    plt.close()
    anim.close()

    return anim

#plt.figure(figsize=(11,4))
#make_animation_1dm(actuators, electric_fields, images, dark_zone)

average_contrast = [np.mean(image[dark_zone] / img_ref.max()) for image in images]

plt.figure(figsize=(13,4))
plt.subplot(1, 3, 1)
deformable_mirror.actuators = actuators[-1]
plt.title('DM surface in nm')
imshow_field(deformable_mirror.surface * 1e9, grid_units=pupil_diameter, mask=aperture, cmap='RdBu', vmin=-8, vmax=8)
plt.colorbar()

plt.subplot(1, 3, 2)
plt.title('Intensity image')
imshow_psf(images[-1] / img_ref.max(), spatial_resolution=spatial_resolution, cmap='inferno', vmin=1e-10, vmax=1e-5)
contour_field(dark_zone, grid_units=spatial_resolution, levels=[0.5], colors='white')
plt.xlabel(r'x ($\lambda / D$)')
plt.ylabel(r'y ($\lambda / D$)')

plt.subplot(1, 3, 3)
plt.title('Average contrast')
plt.plot(range(num_iterations), average_contrast, 'o-')
plt.xlim(0, num_iterations)
plt.yscale('log')
plt.ylim(1e-11, 1e-5)
plt.grid(color='0.5')
plt.xlabel('iterations')

plt.subplots_adjust(left=0.05, right=0.95, top=0.85, bottom=0.15, wspace=0.2)
plt.savefig(paths.figures/'efc_closed_loop_still_frame.pdf')
