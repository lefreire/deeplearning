## para ajudar: https://towardsdatascience.com/building-a-convolutional-neural-network-cnn-in-keras-329fbbadc5f5

from utilities import *
from model import *

utils = Utilities()
mod = ModelK()

file_name = '../et1520/data17_13TeV.AllPeriods.sgn.probes_lhmedium_EGAM1.bkg.VProbes_EGAM7.GRL_v97_et0_eta0.npz'
data, target = utils.read_file(file_name)
raw = utils.return_all_file(file_name)

# get pd and fa
ref_key = 'T0HLTElectronT2CaloTight'
d = raw['data'][:,np.where(raw['features'] == ref_key)[0][0]-1]
d_s = d[target==1]
d_b = d[target!=1]

# Get the detection probability as reference
pd = sum(d_s)/len(d_s) # passed==1 / total
# Get the fake probability as reference
fa = sum(d_b)/len(d_b) # passed==1 / total

del raw

print(data.shape)

#treino primeiro modelo > 15 GeV
mod.construct_model()
kf = StratifiedKFold(n_splits=10, random_state=512, shuffle=True)
# kf = StratifiedKFold(n_splits=2, random_state=512, shuffle=True)
splits = [(train_index, val_index) for train_index, val_index in kf.split(data, target)]
mod.training(mod.model, data[splits[0][0]], target[splits[0][0]],
								 data[splits[0][1]], target[splits[0][1]])

#fine_tuning
#treinando com et2_eta0
et2_et0_file = '../others/17_13/data17_13TeV.AllPeriods.sgn.probes_lhmedium_EGAM2.bkg.VProbes_EGAM7.GRL_v97/data17_13TeV.AllPeriods.sgn.probes_lhmedium_EGAM2.bkg.VProbes_EGAM7.GRL_v97_et2_eta0.npz'
data, target = utils.read_file(et2_et0_file)
mod.sp_obj = sp(patience=25, verbose=True, save_the_best=True)
new_model = mod.model_fine_tuning(mod.model)
mod.training_all_data(new_model, data, target)

et2_et0_file_predict = '../others/18_13/data18_13TeV.AllPeriods.sgn.probes_lhmedium_EGAM2.bkg.VProbes_EGAM7.GRL_v97/data18_13TeV.AllPeriods.sgn.probes_lhmedium_EGAM2.bkg.VProbes_EGAM7.GRL_v97_et2_eta0.npz'
data_test, target_test = utils.read_file(et2_et0_file_predict)

target_pred = mod.predict(new_model, data_test)


#treinando com et1_eta0
et2_et0_file = '../others/17_13/data17_13TeV.AllPeriods.sgn.probes_lhmedium_EGAM2.bkg.VProbes_EGAM7.GRL_v97/data17_13TeV.AllPeriods.sgn.probes_lhmedium_EGAM2.bkg.VProbes_EGAM7.GRL_v97_et1_eta0.npz'
data, target = utils.read_file(et2_et0_file)
mod.sp_obj = sp(patience=25, verbose=True, save_the_best=True)

head_model = Reshape((32, 1, )) (new_model.layers[-3].output)
head_model = Conv1D(16, kernel_size=2, activation='relu', input_shape=(100,1))(head_model)  
head_model = Conv1D(32, kernel_size=2, activation='relu' )(head_model)
head_model = Flatten()(head_model)    
head_model = Dense(32,  activation='relu')(head_model) 
head_model = Dense(1, activation='linear')(head_model) 
head_model = Activation('sigmoid')(head_model)   
for layer in new_model.layers:
			layer.trainable = False
new_model2 = Model(inputs=new_model.input, outputs=head_model)
mod.training_all_data(new_model, data, target)

et2_et0_file_predict = '../others/18_13/data18_13TeV.AllPeriods.sgn.probes_lhmedium_EGAM2.bkg.VProbes_EGAM7.GRL_v97/data18_13TeV.AllPeriods.sgn.probes_lhmedium_EGAM2.bkg.VProbes_EGAM7.GRL_v97_et1_eta0.npz'
data_test, target_test = utils.read_file(et2_et0_file_predict)

target_pred = mod.predict(new_model, data_test)

