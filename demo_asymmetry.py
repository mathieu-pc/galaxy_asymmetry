import numpy as np
from matplotlib import pyplot as plt
from astropy.io import fits
from astropy import units
from scipy.ndimage import gaussian_filter
#======================
import galaxy_asymmetry
#======================
SIGMA_TO_FWHM = 2 * np.sqrt(2 * np.log(2))
FWHM_TO_SIGMA = 1 / SIGMA_TO_FWHM


source  = fits.open("demo_asymmetry_source.fits")[0].data
beam_FWHM = np.array([5, 5]) * units.pixel #beam FWHM specified in pixel units

#The source is noiseless, so we make our own noise cube
def make_noise_cube(noise_RMS, psf_FWHM, shape):
	psf_sigma = np.zeros(len(shape))
	psf_sigma[1:] = FWHM_TO_SIGMA * (psf_FWHM / units.pixel).decompose().value
	noise_cube = gaussian_filter(np.random.normal(size = shape), sigma = psf_sigma, mode = 'wrap')
	noise_cube *= noise_RMS * np.sqrt(noise_cube.size) / np.linalg.norm(noise_cube)
	return noise_cube

#For example, we set the noise as 1/10 the maximum signal of the source
noise_RMS = np.max(source) / 10 
source += make_noise_cube(noise_RMS, beam_FWHM, source.shape)

#Decide on a center.
#For an example, the center is arbitrary so we just choose the middle of the cube
#Direct computation from data assumes an integer center
center = (np.array(source.shape)/2).astype(int) 

#Make a mask for the data
mask = source > noise_RMS * 3

#Show moment 0 of the source
plt.imshow(np.sum(source * mask, axis = 0))
plt.show()
plt.close()

#Compute using from_data API calls
A3D = galaxy_asymmetry.compute_asymmetry_from_data(source, mask, noise_RMS, center)
p_a = galaxy_asymmetry.compute_asymmetry_probability_from_data(
	source,
	mask,
	noise_RMS,
	center,
	beam_FWHM
)
a0, sigmas = galaxy_asymmetry.get_mean_standard_deviations(galaxy_asymmetry.ASYMMETRY_RANGE, p_a)
print("Measure:", a0,"+",sigmas[1],"-", sigmas[0])

#Compare the results between frequentist (vertical line) and Bayesian (distribution)
plt.plot(galaxy_asymmetry.ASYMMETRY_RANGE, p_a, 'k-', label = "Bayesian")
plt.axvline(A3D, color = 'r', label = "Frequentist")
plt.legend(labelspacing = 0)
plt.show()