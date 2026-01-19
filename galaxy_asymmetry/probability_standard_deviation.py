import numpy as np


def get_mode_standard_deviations(x, p_x):
	return _get_standard_deviations_from_centering(lambda x, p_x: x[np.argmax(p_x)], x, p_x)


def get_mean_standard_deviations(x, p_x):
	return _get_standard_deviations_from_centering(lambda x, p_x: np.sum(x * p_x), x, p_x)


def _get_standard_deviations_from_centering(centering, x, p_x):
	assert (np.abs(np.sum(p_x)- 1) < 1e-5)
	x0 = centering(x, p_x)
	return x0, _get_standard_deviations(x, p_x, x0)


def _get_standard_deviations(x, p_x, x_centre):
	centre_index = np.argmin(np.abs(x - x_centre))
	if centre_index == 0:
		low_sigma = 0
	else:
		low_sigma = _get_standard_deviation(x[:centre_index], x_centre, p_x[:centre_index])
	if centre_index == (len(x)-1):
		top_sigma = 0
	else:
		top_sigma = _get_standard_deviation(x[centre_index:], x_centre, p_x[centre_index:])
	return (low_sigma, top_sigma)


def _get_standard_deviation(x, x0, p_x):
	return np.sqrt(np.sum( (x-x0) ** 2 * p_x) / np.sum(p_x))


if __name__ == "__main__":
	res = 10001
	x = np.linspace(0, 1, res)
	p = x * (1-x) * np.exp( - 2 * (np.sqrt(10 * x) - 1) ** 2 )
	p /= np.sum(p)
	x0, s = get_mode_standard_deviations(x, p)
	print(s)
	from matplotlib import pyplot as plt
	plt.plot(x, p)
	plt.axvline(x0 - s[0])
	plt.axvline(x0)
	plt.axvline(x0 + s[1])
	plt.show()