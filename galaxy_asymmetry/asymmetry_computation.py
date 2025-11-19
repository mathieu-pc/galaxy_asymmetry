"""
Asymmetry Bayesian Calculator
"""
import numpy as np
if __name__ == "__main__":
	from probability_magnitude import noncentral
else:
	from .probability_magnitude import noncentral


def _calculate_chi_prod(x, X, N):
	pXx = np.ones(len(x))
	for i in range(len(X)):
		pXx *= noncentral(x, X[i], N[i])
	return pXx


def compute_asymmetry_probability_multi(a, X, Y, N, n_m = 4000, step = 0.01):
	n_a = len(a)
	xCenter = np.sqrt(X ** 2 - N)
	xBounds = (np.min(xCenter), np.max(xCenter))
	x = np.arange(max(xBounds[0] - n_m * step, 0), xBounds[1] + n_m * step, step = step)
	nMeasure = len(X)
	pXx = _calculate_chi_prod(x,X,N)
	mask = (pXx != 0)
	print("np:", np.sum(mask))
	x = x[mask]
	pXx = pXx[mask]
	index = np.argwhere(np.isnan(pXx))
	PA = np.empty(n_a)
	for k in range(n_a):
		pYay = calculate_chi_prod(x * a[k], Y, N)
		integrand = pXx * pYay
		PA1 = np.nansum(integrand)#noncentral chi square started sometimes giving nan instead of 0
		#PA2 = np.nansum(integrand[::2]) * 2
		PA[k] = PA1 #(4 * PA2 - PA1) / 3
	return PA/np.sum(PA)

def compute_asymmetry_probability(a, X, Y, N, n_m = 4000, step = 0.01, *, voice = True):
	"""
	Calculate asymmetry probability distribution.

	Parameters
	----------
	a: array
		asymmetry values to evaluate p(a|X,Y) for
	X:
		denominator defined as sqrt(Q / sigma^2)
	Y:
		numerator defined as sqrt(P / sigma^2)
	N:
		number of points
	n_m:
		maximum number of steps on which asymmetry integral is computed
	step:
		step size for integration
	tol:
		log difference tolerated between maximum and current value before integral is truncated

	Returns
	-------
	PA: array
		probability distribution function evaluated at a for domain a
	"""

	n_a = len(a)
	xCenter = np.sqrt(max(X ** 2 - N, 0))
	x = np.arange(max(xCenter - n_m * step, 0), xCenter + n_m * step, step = step)
	n_x = len(x)
	pXx = noncentral(x, X, N)
	mask = ~((pXx == 0) & np.isnan(pXx))
	ax_all = np.tile(x, n_a) * np.repeat(a, n_x)
	mask_all = np.tile(mask, n_a)
	integrand = np.zeros(mask_all.shape)
	integrand[mask_all] = noncentral(ax_all[mask_all], Y, N)
	integrand = integrand.reshape(n_a, n_x) * pXx
	mask_inf = np.isinf(integrand)
	integrand[mask_inf] = 0 #there is an issue where 0 values sometimes give inf (stems from ncx2 from what I can tell)
	PA1 = np.nansum(integrand, axis = 1)
	PA2 = np.nansum(integrand[:, ::2], axis = 1) * 2
	PA = (PA1 - PA2) / 3 + PA1
	norm = np.nansum(PA)
	if norm != 0:
		return PA/norm
	return PA
	A2 = (Y**2-N) / (X**2-N)
	if voice:
		print("Failed to recover from chi square.")
		print(X, Y, N, A2)
	if np.sum(pXx) == 0:
		if voice:
			print("Will use Lorentzian approximation for high N")
		PA = 1/np.sqrt(a ** 2 + 1 / A2)
		return PA/np.sum(PA)
	if voice:
		print("Will use high N approximation for Y")
	for k in range(n_a):
		pYay = np.exp( - (x * a[k]) ** 2 * (1-Y**2/N) / 2)
		integrand = pXx * pYay
		PA1 = np.nansum(integrand)
		PA2 = np.nansum(integrand[::2]) * 2
		PA[k] = PA1 #(PA1 - PA2) / 3 + PA1 #formula is (4x PA1 - PA2) / 3, but doing thing this order may avoid overflows (Richardson extrap)
	return PA/np.sum(PA)

if __name__ == "__main__":
	a = np.arange(0,1,0.001)
	N = 10000
	Y = np.sqrt(N * 1.03)
	X = 1.05 * Y
	pa = compute_asymmetry_probability(a, X, Y, N)
	#quit()
	from matplotlib import pyplot as plt
	plt.plot(a, pa, 'k-', linewidth = 5)
	plt.xlabel(r"Asymmetry $A$")
	plt.ylabel(r"$p\left(A|X,Y\right)$")
	plt.xlim(0,1)
	plt.axvline( np.sqrt((Y**2 - N)/(X**2 - N)), linewidth = 5)
	plt.tight_layout()
	plt.show()
