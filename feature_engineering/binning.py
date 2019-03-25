# Built-In Libraries
import math

# Third-Party Libraries
import seaborn as sns
import pandas as pd
import numpy as np

# =============================================================================
# This module performs binning using equal interval, equal total, equal 
# positive or equal negative binning on a numerical variable.
# =============================================================================

class TigressBin:
	"""Binning using equal intervial, total, positve and negative"""

	def __init__(self, method='interval', percentile=0.05, binsize=None):
		"""Bins numerical variable 

		Parameters
        ----------
        method (Str, default='interval'): method must be 'interval', 'total',
        								  'positive', or 'negative'

        percetnile (Float, default=0.05): binning thresholds 

		binsize (Int): If percentile is not specified, use binsize instead
					 	to determine the number of bins. 
		"""
	
		if not percentile:
			percentile = math.floor( 1 / binsize )

		assert method in ['interval', 'total', 'positive', 'negative'], \
								'method must be ("interval", "total", "positive", "negative")'

		assert percentile >= 0.01 and percentile <= 0.2, 'percentile' + \
								' is not within range [0.01, 0.2], inclusive.' 

		self.method = method
		self.percentile = percentile

	@staticmethod
	def equal_interval_binning(X, p):
		# Width of the interval is equal
		_min = X.min()
		_max = X.max()
		interval = math.floor((_max - _min) * p)
		q = np.arange(_min, _max + interval, interval)
		return q
		
	@staticmethod
	def total_frequency_binning(X, p):
		# Count of observations per interval is interval
		q = X.quantile(np.arange(0,1+p,p)).values
		return q

	@staticmethod
	def table(X, y=None):
		N = X.shape[0]

		if not isinstance(y, (pd.Series)): 
			y = pd.Series([1]*N)		

		# Conpute distributions 
		freq = pd.crosstab(X, y)
		freq_all = freq.apply(lambda x: x.sum(), axis=1)
		prop = freq.apply(lambda x: x / X.shape[0], axis=1)
		prop_all = prop.apply(lambda x: x.sum(), axis=1)

		# Combine tables
		dist = pd.concat([freq, freq_all, prop, prop_all], axis=1)
		dist.columns = (['count_0','count_1','count_all',
						 'prop_0','prop_1','prop_all'])
		return dist

	def replace(self, X):
		"""Replaces X with binned intervals

		Parameters
        ----------
        X (Pandas Series): Numerical column to bin

		Returns
        -------        
        X (Pandas Series): Series with intervals 
		"""

		# Cut based on bin fit on X
		
		# Replace min and max limits with -inf and +inf
		_bins = self._bins
		binned_X = (pd.cut(X, _bins)
					  .cat.add_categories(['MISSING'])
					  .fillna('MISSING'))
		return binned_X				

	def bin(self, X, y, _return=False):
		"""Bins numerical variable 

		Parameters
        ----------
        X (Pandas DataFrame): Numerical column to bin

        y (Pandas DataFrame): Binary target column

        _return (Boolean, default=False): returns binned X column 
		"""

		if self.method == 'interval': 
			_bins = self.equal_interval_binning(X, self.percentile)
		elif self.method == 'total':
			_bins = self.total_frequency_binning(X, self.percentile)
		elif self.method == 'positive':
			_bins = self.total_frequency_binning(X[y == 1], self.percentile)
		else: # self.method == 'negative':
			_bins = self.total_frequency_binning(X[y != 1], self.percentile)

		# Replace lower and upper bounds with - and + inf
		_bins[0] = -np.inf
		_bins[-1] = np.inf 	

		# Save bins
		self._bins = _bins

		# Apply replace with intervals 
		binned_X = self.replace(X)
		temp = (pd.cut(X, _bins)
					  .cat.add_categories(['MISSING'])
					  .fillna('MISSING'))

		self.dist = self.table(binned_X, y)

		if _return:
			return binned_X