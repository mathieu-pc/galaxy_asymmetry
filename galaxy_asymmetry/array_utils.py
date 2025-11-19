import numpy as np

def fill_around(array, point):
	over_reach=array.shape-np.ones(array.ndim)-point	#Over reach of the cube
								#The under reach is given by the center itself due to array starting at zero
	pad_list=[0]*array.ndim
	for n in range(array.ndim):
		pad_list[n] = tuple((
			max(0, int(over_reach[n] - point[n] + 0.5)),
			max(0, int(point[n] - over_reach[n] + 0.5))
		))
	return np.pad(array, pad_list, mode='constant')