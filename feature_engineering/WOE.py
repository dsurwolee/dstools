"""

"""

from __future__ import print_function
from __future__ import division

# Author: Daniel Lee <dlee8@paypal.com>

import pandas as pd
import numpy as np

""" Notes

Binning
	> Numeric
	> Categorical
	> Tree-based binning

Edge Cases:
	> Rare cases
	> +Inf/-Info cases

Save and loading:
	> Saves mapping of weight of evidence to each value/interval of 
	  values. This can be loaded and re-implemented in a new instance
	  of WOE. 

Plotting and Tables:
	> Horizontal bar plot with information values
	> Weight-of-evidence plot per each feature
	> Weight-of-evidence table per each feature


What it must have:
	> Min_Percentage per Bin - If # of observations within each bin falls within the group, perform merge
	> Merging criteria - How far the absolute deviation from each bin should be for the merging to perform 

https://cran.r-project.org/web/packages/woeBinning/woeBinning.pdf
https://cran.r-project.org/web/packages/woeR/woeR.pdf
""" 

class WOE(object):
	"""<Purpose>

	<Description>

	Parameters
	----------

	Examples
	--------
	>>>
	>>>
	...
	...

	"""

	def __init__(self):
		return
	
	