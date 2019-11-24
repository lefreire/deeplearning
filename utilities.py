import numpy as np
import pandas as pd
from sklearn import preprocessing


class Utilities: 

	def __init__(self):
		pass

	def normalize(self, data):
	  norms = np.abs(data.sum(axis=1))
	  norms[norms==0] = 1
	  return data/norms[:, None]


	def reshape_to_conv1d(self, data):
	  return np.expand_dims(data, axis=2)