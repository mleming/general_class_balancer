import numpy as np
from scipy.stats import mannwhitneyu
from scipy import stats
import random

def prime(i, primes):
	for prime in primes:
		if not (i == prime or i % prime):
			return False
	primes.append(i)
	return i

def get_first_n_primes(n):
	primes = []
	i, p = 2, 0
	while True:
		if prime(i, primes):
			p += 1
			if p == n:
				return primes
		i += 1

def discretize_value(v,buckets):
	if isinstance(v,str):
		for i in range(len(buckets)):
			if buckets[i] == v:
				return i
	else:
		return np.searchsorted(buckets,v)
	assert(False)

# This method uses prime numbers to speed up datapoint matching. Each bucket
# gets a prime number, and each datapoint is assigned a product of these primes.
# These are then matched with one another.
def get_prime_form(confounds,no_buckets,sorted_confounds = None):
	if sorted_confounds is None:
		sorted_confounds = np.sort(confounds,axis=0)
	n_primes = get_first_n_primes(np.sum(no_buckets) + 1)
	discretized_confounds = np.zeros(confounds.shape)
	for i in range(confounds.shape[0]):
		if isinstance(confounds[i,0],str):
			buckets = np.unique(confounds[i,:])
		else:
			buckets_s = []
			for kk in range(0,sorted_confounds.shape[1],int(np.ceil(sorted_confounds.shape[1]/float(no_buckets[i])))):
				buckets_s.append(sorted_confounds[i,kk])
			buckets_s.append(sorted_confounds[i,-1])
			buckets_s = np.array(buckets_s)
			min_conf = sorted_confounds[i,0]
			max_conf = sorted_confounds[i,-1]
			buckets_v = (np.array(range(no_buckets[i] + 1))/float(no_buckets[i])) * (max_conf - min_conf) + min_conf
			sv_ratio = 1.0
			buckets = (sv_ratio) * buckets_s + (1.0 - sv_ratio) * buckets_v
		for j in range(confounds.shape[1]):
			d = discretize_value(confounds[i,j],buckets)
			d = n_primes[int(np.sum(no_buckets[:i])) + d]
			discretized_confounds[i,j] = d
	return discretized_confounds

# Given buckets, selects values that fall into each one
def get_class_selection(classes,primed):
	assert(len(classes) == len(primed))
	num_classes = len(np.unique(classes))
	selection = np.zeros(classes.shape,dtype=bool)
	hasher = {}
	rr = list(range(len(classes)))
	random.shuffle(rr)
	for i in rr:
		p = primed[i]
		if p not in hasher:
			hasher[p] = [[] for x in range(num_classes)]
		hasher[p][classes[i]].append(i)
	for key in hasher:
		value = hasher[key]
		admitted_values = min(map(lambda k:len(k),value))
		for arr in value:
			for i in range(admitted_values):
				selection[arr[i]] = True
	return selection

def multi_mannwhitneyu(arr):
	max_p = -np.Inf
	min_p = np.Inf
	for i in range(len(arr)):
		for j in range(i+1,len(arr)):
			try:
				s,p = stats.ttest_ind(arr[i],arr[j])
			except:
				p = 1
			if p > max_p:
				max_p = p
			if p < min_p:
				min_p = p
	return min_p,max_p

def test_all(classes,confounds):
	unique_classes = np.unique(classes)
	all_min_p = np.Inf
	for i in range(confounds.shape[0]):
		if not isinstance(confounds[i,0],str):
			ts = [confounds[i,classes == j] for j in unique_classes]
			min_p,max_p = multi_mannwhitneyu(ts)
			if min_p < all_min_p:
				all_min_p = min_p
	return all_min_p

def integrate_arrs(S1,S2):
	assert(len(S1) >= len(S2))
	assert(np.sum(~S1) == len(S2))
	if len(S1) == len(S2):
		return S2
	i = 0
	i2 = 0
	output = np.zeros(S1.shape,dtype=bool)
	while i < len(S1):
		if ~S1[i]:
			output[i] = S2[i2]
			i2 += 1
		i += 1
	assert(np.sum(output) == np.sum(S2))
	return output
	
# Main function. Takes as input classes (as integers starting from 0 in a 1D
# numpy array) and confounds (as floats and strings, or just objects, in a
# 2D numpy array). plim is the max p-value, in a nonparametric statistical test,
# at which discretization stops and enough buckets have been reached. If recurse
# is set to True, this method calls itself recursively on excluded data, though
# this doesn't guarantee that the final p values for continuous covariates will
# be up to snuff.
# Method returns an array of logicals that selects a subset of the given data,
# also forcing equal ratios between each class.
def class_balance(classes,confounds,plim = 0.05,recurse=True):
	classes = np.array(classes)
	unique_classes = np.unique(classes)
	no_buckets = [1 for x in range(confounds.shape[0])]
	
	# Used for bucketing purposes
	sorted_confounds = np.sort(confounds,axis=1)
	# Automatically marks strings as discrete, giving each its own bucket
	string_mapper = {}
	unique_strs = []
	for i in range(confounds.shape[0]):
		if isinstance(confounds[i,0],str):
			u = np.unique(confounds[i,:])
			unique_strs.append(u)
			no_buckets[i] = len(u)
	p_vals = [0 for x in range(confounds.shape[0])]
	selection = np.ones(classes.shape,dtype=bool)
	while min(p_vals) < plim and np.sum(selection) > 0:
		primed = get_prime_form(confounds,no_buckets, sorted_confounds)
		primed = np.prod(primed,axis=0,dtype=int)
		selection = get_class_selection(classes,primed)
		rr = list(range(confounds.shape[0]))
		random.shuffle(rr)
		for i in rr:
			if not isinstance(confounds[i,0],str):
				ts = [confounds[i,np.logical_and(selection, classes == j)] for j in unique_classes]
				if np.any(list(map(lambda k: len(k) < 5, ts))):
					selection = np.zeros(classes.shape,dtype=bool)
					break
				min_p,max_p = multi_mannwhitneyu(ts)
				p_vals[i] = min_p
				if p_vals[i] < plim:
					no_buckets[i] += 1
					break
			else:
				p_vals[i] = 1
	if np.sum(selection) > 40:
		recurse_selection = integrate_arrs(selection, class_balance(classes[~selection],confounds[:,~selection],plim = plim))
		selection = np.logical_or(selection , recurse_selection)
	return selection