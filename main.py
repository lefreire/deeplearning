## para ajudar: https://towardsdatascience.com/building-a-convolutional-neural-network-cnn-in-keras-329fbbadc5f5

from utilities import *
from model import *

utils = Utilities()
mod = Model()

data=dict(np.load('data17_13TeV.AllPeriods.sgn.probes_lhmedium_EGAM2.bkg.VProbes_EGAM7.GRL_v97/data17_13TeV.AllPeriods.sgn.probes_lhmedium_EGAM2.bkg.VProbes_EGAM7.GRL_v97_et2_eta0.npz')) 

data_pd = pd.DataFrame.from_dict(data['data'])
target_pd = data['target']
data_pd = data_pd.iloc[:,1:101]

new_data = utils.reshape_to_conv1d(utils.normalize(data_pd))


test_pd = new_data[0:1000] 
test_y = target_pd[0:1000]    
train_pd = new_data[1000:] 
train_y = target_pd[1000:] 

tt = mod.all_model(train_pd, train_y, test_pd, test_y)




