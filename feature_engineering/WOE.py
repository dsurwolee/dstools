# Built-in Libraries
import math

# Third-Party Libraries
import pandas as pd
import numpy as np

from IPython.display import display
from pandas.api.types import is_numeric_dtype

from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

# =============================================================================
# This module performs weight-of-evidence and information value computation for 
# categorical and continuous variables
# =============================================================================


class TigressWOE:
	"""Computes weight-of-evidence and information"""

	def __init__(self, vtype, min_perc_total=0.05, min_perc_class=0.001, limit=5):

		assert type(vtype) is dict, "vtype is not a dictionary: %r" % vtype 
		assert min_perc_total >= 0.0001 and min_perc_total <= 0.2, \
				"min_perc_total not in range [0.0001, 0.2]. User Input: %.2f" % min_perc_total
		assert min_perc_class >= 0 and min_perc_class <= 0.2, \
				"min_perc_total not in range [0.0, 0.2]: User Input: %.2f" % min_perc_class

		self.group_size = math.floor( 1 / min_perc_total )
		self.min_perc_total = min_perc_total        
		self.min_perc_class = min_perc_class
		self.limit = limit
		self.vtype = vtype
		self.btype = {} # Bin Type
		self.bins = {}
		self.tables = {}
		self.woe_map = {} 
		self.iv = {} 
		self.rare_group = {}

	@staticmethod
	def compute_woe(x, y, limit):
		"""Cretes weight-of-evidence table

		Parameters
        ----------
        x (Pandas Series): Array of explanatory variable 
            
        y (Pandas Series): Array of target values

		limit (Int): Truncation value applied on + INF and -INF woe.

		Returns
        -------        
        woe_table (Pandas DataFrame): Pandas DataFrame    
		"""

		# Compute distributions
		dist = pd.crosstab(x, y, margins=True)[:-1]
		pdist =  dist.apply(lambda x: x / x.sum(), axis=0)
		woe = pdist.apply(lambda x: np.log(x[0] / x[1]), axis=1) \
                    .clip(-limit, limit)

		# Combine results
		woe_table = pd.concat([dist, pdist, woe], axis=1)
		woe_table.columns = ['n_goods','n_bads','n_all','p_goods',
							 'p_bads','p_all','woe']
		return woe_table


	@staticmethod
	def compute_iv(table):
		"""Creates information value tavble

		Parameters
        ----------
        table (Pandas DataFrame): DataFrame with WOE values


		Returns
        -------        
        iv_table (Pandas DataFrame): DataFrame with IV values   
		"""

		def calculate_iv(x):
			if x['p_goods'] != x['p_bads']:
				return (x['p_goods'] - x['p_bads']) * x['woe']
			return 0 

		# Computes informaiton value 
		ivs = table.apply(lambda x: calculate_iv(x), axis=1)
		return ivs.sum()


	def quantile_binning(self, x, c): 
		q = x.quantile(np.arange(0,1+self.min_perc_total,
                         self.min_perc_total)).values
		q[0], q[-1] = -np.inf, np.inf
		q = np.unique(q)
		self.bins[c] = q
		x = pd.cut(x, bins=q)
		return x  


	def tree_binning(self, x, y, params, cv, v):
		clf = DecisionTreeClassifier(criterion='gini', 
									   min_samples_split=30, 
									   min_samples_leaf=30) # Instantiate decision tree 
		tree = GridSearchCV(clf, param_grid=params, cv=cv)
		tree.fit(x, y)

		if v == 'continuous':
			t = tree.best_estimator_.tree_.threshold
			bins = sorted([v for v in t if v >= 0] + [-np.inf, np.inf])
			name = x.columns[0]
			self.bins[name] = bins
			x_bin = pd.cut(x[name], bins=bins)
		else:	
			x_bin = pd.Series(tree.predict_proba(x)[:, 1].round(4))
			
		return x_bin.astype(str), tree


	def sparse_cleaning(self, x, y, v):
		stat = pd.crosstab(x, y, normalize=True, margins=True)
		rare = (stat[(stat[1] < self.min_perc_class) | 
					 (stat[1] > 1 - self.min_perc_class)].index.tolist())

		if v != 'continuous':
			rare_grp = stat[stat['All'] < self.min_perc_total].index.tolist()
			rare = list(set(rare).union(rare_grp)) 
			# Store rare conditions
			if len(rare) > 1:
				x = x.replace(rare, value='RARE')
				self.rare_group[x.name] = rare
		return x


	def fit(self, X, y):
		"""Computes WOE and IV and stores the results in a dictionary

		Parameters
        ----------
        X (Pandas DataFrame or Series): DataFrame with explanatory variables to store

        y (Series): Target column with labels 1's and 0's 
		"""

		# If X is not a frame, convert it to frame
		if not isinstance(X, (pd.DataFrame)):
			X = X.to_frame()

		# Match fileds in vtype dict and dataframe input X
		cols = list(set(X.columns).intersection(self.vtype))

		# Check that target is numeric
		assert is_numeric_dtype(y), "target column is non-numeric."

		for c in cols:
			v = self.vtype[c]
			x = X[c]
			self.btype[c] = 'reg_fit'
			
			# If variable is numeric, bin based on equal percentiles
			if v == 'continuous':
				x = self.quantile_binning(x, c)
				x = x.cat.add_categories(['MISSING','RARE'])

			# Pre-process - replace NaN
			x = x.fillna('MISSING').astype(str)

			# Pre-process - remove sparsity
			x = self.sparse_cleaning(x, y, v)

			# Calculate WOE
			woe = self.compute_woe(x, y, self.limit)
			self.tables[c] = woe

			# Revise Mapping for RARE values 
			woe_map = woe['woe'].to_dict()

			if 'RARE' in woe_map:
				for e in self.rare_group[c]:
					woe_map[e] = woe_map['RARE']

			self.woe_map = woe_map

	def tree_fit(self, X, y, num_impute='mean', params={'max_depth': [3,4,5]}, cv=5):
		"""Computes WOE and IV and stores the results in a dictionary

		Parameters
        ----------
        X (Pandas DataFrame or Series): Dataframe with explanatory variables to store

        y (Series): Target column with labels 1's and 0's 

        num_impute (String, default='mean'): ['mean','median'] imputation for continuous vars

        params (Dict, default={'max_depth': [3,4,5]}): grid-search parameter

        cv (Int, default=5): cross-validation for grid-search
		"""

		# If X is not a frame, convert it to frame
		if not isinstance(X, (pd.DataFrame)):
			X = X.to_frame()

		# Match fileds in vtype dict and dataframe input X
		cols = list(set(X.columns).intersection(self.vtype))

		# Check that target is numeric
		assert is_numeric_dtype(y), "target column is non-numeric."

		self.tree_num_replace_trees = {}
		self.tree_num_replace_with_v = {}
		self.woe_map = {}

		for c in cols:
			v = self.vtype[c]
			x = X[c]
			self.btype[c] = 'tree_fit'

			# If categorical, convert to WOE
			if v == 'categorical':
				x = x.fillna('MISSING')
				x = self.sparse_cleaning(x, y, v)
				woe = self.compute_woe(x, y, self.limit)
				woe = woe['woe'].to_dict()
				x = x.map(lambda k: woe[k] if woe[k] else k)
			else:
				imp_value = x.agg(num_impute)
				x = x.fillna(imp_value) 

			# Apply tree binning
			x, tree = self.tree_binning(x.to_frame(), y, params, cv, v)
			x = x.rename(c)

			# Compute WOE
			woe = self.compute_woe(x, y, self.limit)
			woe_values = woe['woe'].to_dict()

			# Save properties for replacing
			if v == 'continuous':
				self.tree_num_replace_trees[c] = tree
				self.tree_num_replace_with_v[c] = num_impute
				self.woe_map[c] = woe_values
				self.tables[c] = woe
			else: 
				raw_to_prob_map = (pd.concat([x, X[c].rename('raw')],axis=1)
										.fillna('MISSING')
										.drop_duplicates()
										.set_index('raw')
										.to_dict()[c])

				raw_to_woe_map = {k: woe_values[v] for k, v in raw_to_prob_map.items()}

				prob_to_raw_map = {}
				for raw, prob in raw_to_prob_map.items():
					if prob not in prob_to_raw_map:
						prob_to_raw_map[prob] = [raw]
					else:
						prob_to_raw_map[prob] += [raw]

				woe.index = [str(prob_to_raw_map[i]) for i in woe.index]
				woe.index.name = c

				self.woe_map[c] = raw_to_woe_map
				self.tables[c] = woe


	def replace(self, X, append=True, col_suffix='_woe'):
		"""Replaces X with woe value

		Parameters
        ----------
        X (Pandas DataFrame): DataFrame with columns to replace with woe values

        append (Dict, default=True): Returns both original and woe columns if True.
        							 Else, return just woe columns

		col_suffix (Str, default='_woe'): WOE column name suffix

		Returns
        -------        
        X or X[cols] (Pandas DataFrame): DataFrame with woe values
		"""
        
		if isinstance(X, (pd.Series)):
			X = X.to_frame()

        # Find common fields
		cols = list(set(X.columns).intersection(self.vtype))
		X = X[cols].copy()

		def replace_with_woe(k, wmap):
			# Logic to handle missingness and rare cases
			if k in wmap:
				return wmap[k]
			if k == 'RARE' and 'RARE' in wmap: 
				return wmap['RARE']
			if k == 'MISSING' and 'MISSING' in wmap: 	
				return wmap['MISSING']
			if k not in wmap: 
				if 'RARE' in wmap:
					return wmap['RARE']
				if 'MISSING' in wmap:
					return wmap['MISSING']	
				return 0

		# Replace fields fitted already 
		for c in cols:
			x = X[c]
			woe_map = self.woe_map[c]   

			if self.btype[c] == 'reg_fit':
				if self.vtype[c] == 'continuous':
					q = self.bins[c]
					x = pd.cut(x, bins=q)
					x = x.cat.add_categories(['MISSING'])
				x = x.fillna('MISSING').astype(str)
			else:
				if self.vtype[c] == 'continuous':
					num_impute = self.tree_num_replace_with_v[c]
					tree = self.tree_num_replace_trees[c]
					x = x.fillna(num_impute)
					x = pd.cut(x, self.bins[c]).astype(str)
			X[c+col_suffix] = x.map(lambda k: replace_with_woe(k, woe_map))
			if not append:
				X = X.drop(c, axis=1)        
		return X


	def information_value(self):
		"""Returns information value"""

		ivs = {}
		for c, stat in self.tables.items():
			ivs[c] = self.compute_iv(stat)

		return pd.Series(ivs)
	

	def weight_of_evidence(self, returns='All'):
		"""Returns weight of evidence"""

		def reformat_woe_table(name, t):
			rare_in_table = False
			if 'RARE' in t.index:
				rare_in_table = True

			t = t.reset_index().reset_index()
			t = t.rename(columns={'index': 'groups', name: 'values'})

			if rare_in_table:
				values = t['values'].tolist()
				rare_i = values.index('RARE')
				rares = self.rare_group[name]

				reformatted_values = [rares if v == 'RARE' else [str(v)] for v in values]
				t.loc[:,'values'] = reformatted_values
				t.loc[rare_i,'groups'] = 'RARE'
			return t

		reformatted_tables = {}
		for name, t in self.tables.items():
			reformatted_tables[name] = reformat_woe_table(name, t)

		if returns == 'All':
			return reformatted_tables

		return reformatted_tables[name]
