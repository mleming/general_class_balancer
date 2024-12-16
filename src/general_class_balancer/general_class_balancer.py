import numpy as np
from scipy.stats import mannwhitneyu
from scipy import stats
import random
import pandas as pd

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

def is_nan(k,inc_null_str=False):
	if k is None:
		return True
	if inc_null_str and isinstance(k,str):
		if k.lower() == "null" or k.lower() == "unknown":
			return True
	try:
		if np.isnan(k):
			return True
		else:
			return False
	except:
		if k == np.nan:
			return True
		else:
			return False

# This method uses prime numbers to speed up datapoint matching. Each bucket
# gets a prime number, and each datapoint is assigned a product of these primes.
# These are then matched with one another.
def get_prime_form(confounds,n_buckets,sorted_confounds = None):
	if sorted_confounds is None:
		sorted_confounds = np.sort(confounds,axis=0)
	n_primes = get_first_n_primes(np.sum(n_buckets) + 1)
	discretized_confounds = np.zeros(confounds.shape)
	for i in range(confounds.shape[0]):
		if isinstance(confounds[i,0],str):
			buckets = np.unique(confounds[i,:])
		else:
			buckets_s = []
			for kk in range(0,sorted_confounds.shape[1],int(np.ceil(sorted_confounds.shape[1]/float(n_buckets[i])))):
				buckets_s.append(sorted_confounds[i,kk])
			buckets_s.append(sorted_confounds[i,-1])
			buckets_s = np.array(buckets_s)
			min_conf = sorted_confounds[i,0]
			max_conf = sorted_confounds[i,-1]
			buckets_v = (np.array(range(n_buckets[i] + 1))/float(n_buckets[i])) * (max_conf - min_conf) + min_conf
			sv_ratio = 1.0
			buckets = (sv_ratio) * buckets_s + (1.0 - sv_ratio) * buckets_v
		for j in range(confounds.shape[1]):
			d = discretize_value(confounds[i,j],buckets)
			d = n_primes[int(np.sum(n_buckets[:i])) + d]
			discretized_confounds[i,j] = d
	return discretized_confounds

# Given buckets, selects values that fall into each one
def get_class_selection(classes,primed,unique_classes=None):
	assert(len(classes) == len(primed))
	if unique_classes is None:
		num_classes = len(np.unique(classes))
	else:
		num_classes = len(unique_classes)
	selection = np.zeros(classes.shape,dtype=bool)
	hasher = {}
	rr = list(range(len(classes)))
	random.shuffle(rr)

	for i in rr:
		if True:
			p = primed[i]
			if p not in hasher:
				hasher[p] = [[] for x in range(num_classes)]
			hasher[p][classes[i]].append(i)
		else:
			print("Hasher screw up")
			exit()
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
			s,p = mannwhitneyu(arr[i].astype(float),arr[j].astype(float))
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

def integrate_arrs_none(S1,S2):
	assert(len(S1) >= len(S2))
	assert(np.sum(S1) == len(S2))
	i = 0
	i2 = 0
	output = np.zeros(S1.shape,dtype=bool)
	while i < len(S1):
		if S1[i]:
			output[i] = S2[i2]
			i2 += 1
		i += 1
	return output

# Returns a boolean array that is true if either classes or confounds has a None
# or NaN value anywhere at the given index
def get_none_array(classes=None,confounds=None):
	if classes is not None and confounds is not None:
		assert(confounds.shape[1] == classes.shape[0])
	elif classes is None:
		classes = np.ones((confounds.shape[1],))
	elif confounds is None:
		confounds = np.ones((1,classes.shape[0]))
	else:
		raise Exception("Cannot have two null arrays input to get_none_array")
	has_none = np.zeros(classes.shape,dtype=bool)
	for i in range(confounds.shape[1]):
		if not has_none[i]: has_none[i] = is_nan(classes[i])
		for j in range(confounds.shape[0]):
			if not has_none[i]: has_none[i] = is_nan(confounds[j,i])
	return has_none



# Main function. Takes as input classes (as integers starting from 0 in a 1D
# numpy array) and confounds (as floats and strings, or just objects, in a
# 2D numpy array). plim is the max p-value, in a nonparametric statistical test,
# at which discretization stops and enough buckets have been reached. If recurse
# is set to True, this method calls itself recursively on excluded data, though
# this doesn't guarantee that the final p values for continuous covariates will
# be up to snuff.
# Method returns an array of logicals that selects a subset of the given data,
# also forcing equal ratios between each class.

def class_balance(classes,
		confounds,
		class_col = None,
		plim = 0.05,
		recurse=True,
		exclude_none=True,
		unique_classes = None):

	if isinstance(classes,np.ndarray):
		classes = np.array(classes).astype(object)
		assert isinstance(confounds,np.ndarray) or isinstance(confounds,list)
		confounds = np.array(confounds)
	elif isinstance(classes,pd.DataFrame):
		if class_col is None:
			raise Exception("Set class col is input is dataframe")
		else:
			confounds = classes[confounds].to_numpy()
			classes = classes[class_col].to_numpy()
			confounds = np.swapaxes(confounds,0,1)
			#classes = np.expand_dims(classes,axis=0)
			#print(confounds.shape)
			#print(classes.shape)
	confounds = np.array(confounds)
	if len(confounds) == 0:
		confounds = np.ones((1,len(classes)),dtype=object)
	ff = {}
	if exclude_none:
		has_none = get_none_array(classes,confounds)
		confounds = confounds[:,~has_none]
		classes = classes[~has_none]
		assert (all([not is_nan(_) for _ in classes]))
		for c in range(confounds.shape[0]):
			assert all([not is_nan(_) for _ in confounds[c,:]])
	else:
		has_none = get_none_array(classes=None,confounds=confounds)
		confounds[:,has_none] = "none"
		has_none = get_none_array(classes=classes,confounds=None)
		classes[has_none] = "none"
	assert(np.all([not is_nan(_) for _ in classes]))
	classes = np.array(classes)
	if unique_classes is None:
		try:
			unique_classes = np.unique(classes)
		except:
			print(classes)
			exit()
	elif isinstance(unique_classes,list):
		unique_classes = np.unique(unique_classes)
	if not np.all(sorted(unique_classes) == list(range(len(unique_classes)))):
		for i in range(len(classes)):
			for j in range(len(unique_classes)):
				if classes[i] == unique_classes[j]:
					classes[i] = j
					break
	n_buckets = [1 for x in range(confounds.shape[0])]
	# Used for bucketing purposes
	sorted_confounds = np.sort(confounds,axis=1)
	# Automatically marks strings as discrete, giving each its own bucket
	string_mapper = {}
	unique_strs = []
	for i in range(confounds.shape[0]):
		if len(confounds.shape) > 1 and confounds.shape[1] > 0 and isinstance(confounds[i,0],str):
			u = np.unique(confounds[i,:])
			unique_strs.append(u)
			n_buckets[i] = len(u)
	p_vals = [0 for x in range(confounds.shape[0])]
	selection = np.ones(classes.shape,dtype=bool)
	while min(p_vals) < plim and np.sum(selection) > 0:
		primed = get_prime_form(confounds,n_buckets, sorted_confounds)
		primed = np.prod(primed,axis=0,dtype=int)
		selection = get_class_selection(classes,
										primed,
										unique_classes=unique_classes)
		rr = list(range(confounds.shape[0]))
		random.shuffle(rr)
		for i in rr:
			if not isinstance(confounds[i,0],str):
				ts = [confounds[i,np.logical_and(selection, classes == j)] \
					for j in range(len(unique_classes))]
				#print("ts/ts.shape")
				#print(ts)
				#print(len(ts))
				#print(np.array(ts).shape)
				#print(unique_classes)
				#print(classes)
				# Makes sure there are at least five instances of 
				# each class remaining
				if np.any(list(map(lambda k: len(k) < 5, ts))):
					print("BREAKING")
					#print(list(map(lambda k: len(k) < 5, ts)))
					selection = np.zeros(classes.shape,dtype=bool)
					break				
				min_p,max_p = multi_mannwhitneyu(ts)
				p_vals[i] = min_p
				if p_vals[i] < plim:
					n_buckets[i] += 1
					break
			else:
				p_vals[i] = 1
	if np.sum(selection) > 40 and confounds[:,~selection].shape[1] > 0:
		recurse_selection = integrate_arrs(selection, class_balance(classes[~selection],confounds[:,~selection],plim = plim,exclude_none=False,unique_classes=unique_classes))
		selection = np.logical_or(selection , recurse_selection)
	if exclude_none:
		selection = integrate_arrs_none(~has_none,selection)
		assert(len(selection) == len(has_none))
		assert(np.sum(~has_none) == len(classes))
	return selection

def separate_set(selections,set_divisions = [0.5,0.5],IDs=None):
	assert(isinstance(set_divisions,list))
	set_divisions = [i/np.sum(set_divisions) for i in set_divisions]
	rr = list(range(len(selections)))
	random.shuffle(rr)
	if IDs is None:
		IDs = np.array(list(range(len(selections))))
	selections_ids = np.zeros(selections.shape,dtype=int)
	totals = list(range(len(set_divisions)))
	prime_hasher = {}
	for i in rr:
		if not selections[i]:
			continue
		is_none = IDs[i] == None or IDs[i] == "NULL"
		if not is_none and IDs[i] in prime_hasher:
			selections_ids[i] = prime_hasher[IDs[i]]
			totals[selections_ids[i] - 1] += 1
			continue
		for j in range(len(set_divisions)):
			if np.sum(totals) == 0 or \
				totals[j] / np.sum(totals) < set_divisions[j]:
				break
		selections_ids[i] = j+1
		totals[j] += 1
		if not is_none and IDs[i] not in prime_hasher:
			prime_hasher[IDs[i]] = j + 1
	return selections_ids
