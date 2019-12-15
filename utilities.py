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
        data = np.array([data])
        return np.transpose(data, [1,2,0])

    def return_all_file(self, file_name):
        return dict(np.load(file_name))

    def read_file(self, file_name):
        raw = dict(np.load(file_name))
        data = raw['data'][:,1:101]
        data = self.reshape_to_conv1d(self.normalize(data))
        target = raw['target']
        del raw
        return data, target
