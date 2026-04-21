"""
Compute the noncentral chi square distribution
This is essentially a wrapper for scipy.stats.ncx2
"""
from scipy.stats import ncx2

def noncentral(x, X, degrees_of_freedom):
	return 2 * X * np.exp(ncx2.logpdf(X ** 2, nc = x ** 2, df = degrees_of_freedom))


if __name__ == "__main__":
	import numpy as np
	from matplotlib import pyplot as plt
	#manual integration is a little tricky because the exponentials tend to blow up
	#strategy is evaluate multiplicative constant near and return log values
	def inta(x, N, step = 1/1000):
		angles = np.arange(0,np.pi, step)[:,None]
		args = x * np.cos(angles)
		dif = np.average(args)
		args -= dif
		result = np.sum(np.exp(args) * (np.sin(angles) ** N) * step, axis = 0)
		return np.log(result) + dif
	def manualEvaluation(x,X, N):
		logInt = inta(x * X, N-2)
		args = logInt - (x ** 2 + X ** 2)/2
		result = np.exp(args)
		result /= np.sum(result)
		return result
	x = np.arange(0, 20, 0.1)
	par = 10
	N = 10
	mean = np.sqrt(par ** 2 - N)
	eval = noncentral(x, par, N)
	eval /= np.sum(eval)
	plt.plot(x, eval, 'k-', label = "Scipy ncx2")
	plt.plot(x, manualEvaluation(x, par, N), 'g--', label = "Manually integrated")
	plt.axvline(mean, linestyle = '--', color = 'r', label = "Mean approx")
	plt.legend(labelspacing = 0)
	plt.show()
