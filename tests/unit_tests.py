#!/usr/bin/python
import unittest
import os,sys,time

import os,sys,time
import numpy as np
import shutil
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu
from scipy import stats
import random
import pandas as pd
from numpy.random import normal  as norm

wd = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

sys.path.insert(0,wd)
sys.path.insert(0,os.path.join(wd,'src'))
sys.path.insert(0,os.path.join(wd,'src','general_class_balancer'))

from src.general_class_balancer.general_class_balancer import *

class TestSimple(unittest.TestCase):
	def test_one(self):
		return
		show_balanced = True
		N = 12000 # Number of datapoints
		#confounds = np.random.rand(2, N)

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
		print(confounds.shape)
		selection = class_balance(classes,confounds,plim=0.1)
		print(len(selection))
		print(np.sum(selection))
		print(test_all(classes[selection],confounds[:,selection]))

		cnum = 2
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

	def test_pandas(self):
		show_balanced = True
		N = 12000 # Number of datapoints
		#confounds = np.random.rand(2, N)

		# Unfortunately, numpy arrays don't support mixed types, so in order to mix
		# floats (i.e., continuous covariates) and strings (i.e., discrete, or labeled,
		# covariates), it is necessary to cast them all as objects. Slows things down,
		# but if covariates are all of one type, they can be cast as one or the other.
		c1 = np.array([norm(1, 2, int(N/3)),norm(0.5, 2, int(N/3)),norm(2, 1, int(N/3))],dtype=object).flatten()
		c2 = np.array([norm(3, 2, int(N/3)),norm(0.5, 2, int(N/3)),norm(3.1, 0.25, int(N/3))],dtype=object).flatten()
		c3 = np.array([norm(2, 1, int(N/3)),norm(0.5, 2, int(N/3)),norm(2.9, 1, int(N/3))],dtype=object).flatten()
		
		#confounds = np.concatenate((c1,c2,c3),axis=1)
		c4 = np.array([[np.random.choice(["a","b","c"]) for x in range(N)]],dtype=object).flatten()
		#confounds = np.concatenate((confounds,c4),axis=0)

		classes = np.array([0 for x in range(int(N/2))] + [1 for x in range(int(N/2))])
		classes = np.array([0 for x in range(int(N/3))] + [1 for x in range(int(N/3))] + [2 for x in range(int(N/3))])
		print(classes.shape)
		print(c1.shape)
		print(c2.shape)
		print(c3.shape)
		print(c4.shape)
		df = pd.DataFrame({"class":classes,"c1":c1,"c2":c2,"c3":c3,"c4":c4})
		selection = class_balance(df,class_col="class",confounds=["c1"],plim=0.1)
		#print(test_all(classes[selection],confounds[:,selection]))

		cnum = 2
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

	

if __name__ == "__main__":
	#clear_files()
	# Runs the tests twice, once with cached files and once without
	#unittest.main()
	unittest.main()
	
	
