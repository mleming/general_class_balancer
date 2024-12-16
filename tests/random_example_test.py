import numpy as np
from scipy.stats import mannwhitneyu
from scipy import stats
import random
from numpy.random import normal  as norm
import matplotlib.pyplot as plt

from general_class_balancer import *

# Sample script showing how this balances on simulated, random data.

show_balanced = True
N = 12000 # Number of datapoints
confounds = np.random.rand(2, N)

# Unfortunately, numpy arrays don't support mixed types, so in order to mix
# floats (i.e., continuous covariates) and strings (i.e., discrete, or labeled,
# covariates), it is necessary to cast them all as objects. Slows things down,
# but if covariates are all of one type, they can be cast as one or the other.
c1 = np.array([norm(1, 2, int(N/3)),norm(0.5, 2, int(N/3)),norm(2, 1, int(N/3))],dtype=object)
c2 = np.array([norm(3, 2, int(N/3)),norm(0.5, 2, int(N/3)),norm(3.1, 0.25, int(N/3))],dtype=object)
c3 = np.array([norm(2, 1, int(N/3)),norm(0.5, 2, int(N/3)),norm(2.9, 1, int(N/3))],dtype=object)
confounds = np.concatenate((c1,c2,c3),axis=1)
c4 = np.array([[np.random.choice(["a","b","c"]) for x in range(N)]],dtype=object)
confounds = np.concatenate((confounds,c4),axis=0)

classes = np.array([0 for x in range(int(N/2))] + [1 for x in range(int(N/2))])
classes = np.array([0 for x in range(int(N/3))] + [1 for x in range(int(N/3))] + [2 for x in range(int(N/3))])

selection = class_balance(classes,confounds,plim=0.25)
print(np.sum(selection))
print(test_all(classes[selection],confounds[:,selection]))

cnum = 0
no_bins = 50
#bins = (np.array(range(no_bins)) - (no_bins/2))/(float(no_bins)/4)
bins = 'auto'
if show_balanced:
	plt.hist(confounds[cnum,np.logical_and(selection,classes == 0)], bins= bins, fc=(0, 0, 1, 0.5))
	plt.hist(confounds[cnum,np.logical_and(selection,classes == 1)], bins= bins, fc=(0, 1, 0, 0.5))
	plt.hist(confounds[cnum,np.logical_and(selection,classes == 2)], bins= bins, fc=(1, 0, 0, 0.5))
else:
	plt.hist(confounds[cnum,classes == 0], bins= bins)
	plt.hist(confounds[cnum,classes == 1], bins= bins)
	plt.hist(confounds[cnum,classes == 2], bins= bins)


plt.show()
