"""
This file handles the interface of asymmetry between input data and the main computation.
"""
import numpy as np
from astropy import units
from astropy.units import quantity_input
from .array_utils import fill_around
from .asymmetry_computation import compute_asymmetry_probability


ASYMMETRY_RESOLUTION = 100
ASYMMETRY_RANGE = np.arange(0, 1 + 1 / ASYMMETRY_RESOLUTION, 1 / ASYMMETRY_RESOLUTION)


@quantity_input(beam_FWHM = units.pixel)
def _get_beam_factor(beam_FWHM):
	beam_f = (np.copy(beam_FWHM) / units.pixel).decompose().value
	beam_f[beam_f < 1] = 1
	beam_f = np.prod(beam_f)
	return beam_f


def _make_expanded_data(data, mask, center):
	expanded_data = fill_around(data, center)
	expanded_mask = fill_around(mask, center)
	expanded_mask = expanded_mask | np.flip(expanded_mask) #mask is symmeterized
	anti_symmetric_data = (expanded_data - np.flip(expanded_data))[expanded_mask]
	even_symmetric_data = (expanded_data + np.flip(expanded_data))[expanded_mask]
	return even_symmetric_data, anti_symmetric_data


@quantity_input(beam_FWHM = units.pixel)
def compute_reduced_magnitude(data, noise_RMS, beam_FWHM, counting = 2):
	"""
	Returns the reduced magnitude of data according to statistical significance
	
	Parameters
	----------
	data: array
		data to compute the reduced magnitude of
	noise_RMS: float
		gaussian noise root mean square of data
		same units as data
	beam_FWHM: array
		beam FWHM in pixels
		dimensions not convolved should receive 1 as input or be omitted
	Returns
	-------
	reduced_magnitude: float
		reduced magnitude of data
	"""
	reduced_magnitude = np.linalg.norm(data) / noise_RMS / np.sqrt(_get_beam_factor(beam_FWHM) * counting)
	return reduced_magnitude


def compute_asymmetry_from_data(data, mask, noise_RMS, center):
	"""
	Compute the frequentist estimate asymmetry (A3D, A2D, A1D)
	
	Parameters
	----------
	data: array
		data to compute the reduced magnitude of
	mask: array
		binary mask of data
		mask will be symmeterized
	noise_RMS: float
		gaussian noise root mean square of data
		same units as data
	Returns
	-------
	A: float
		asymmetry best estimate
	"""
	even_symmetric_data, anti_symmetric_data = _make_expanded_data(data / noise_RMS, mask, center)
	N = len(even_symmetric_data) / 2
	X2 = np.sum(even_symmetric_data ** 2) / 2
	Y2 = np.sum(anti_symmetric_data ** 2) / 2
	A2 = (Y2 - N) / (X2 - N)
	A = np.sqrt(A2)
	return A


@quantity_input(beam_FWHM = units.pixel)
def compute_asymmetry_probability_from_data(data, mask, noise_RMS, center, beam_FWHM, **kwargs):
	"""
	Compute the asymmetry probability for input data
	
	Parameters
	----------
	data: array
		data to compute the reduced magnitude of
	mask: array
		binary mask of data
		mask will be symmeterized
	noise_RMS: float
		gaussian noise root mean square of data
		same units as data
	beam_FWHM: array
		beam FWHM in pixels for each dimension
		dimensions not convolved should receive 1 as input or be omitted
	Returns
	-------
	p_a: array
		probability distribution of underlying asymmetry for input data	
	"""
	assert mask.dtype == bool
	even_symmetric_data, anti_symmetric_data = _make_expanded_data(data, mask, center)
	X = compute_reduced_magnitude(even_symmetric_data, noise_RMS, beam_FWHM)
	Y = compute_reduced_magnitude(anti_symmetric_data, noise_RMS, beam_FWHM)
	nu = len(even_symmetric_data) / _get_beam_factor(beam_FWHM) / 2
	p_a = compute_asymmetry_probability(ASYMMETRY_RANGE, X, Y, nu, **kwargs)
	return p_a