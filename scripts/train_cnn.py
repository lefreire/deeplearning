

try:
  from tensorflow.compat.v1 import ConfigProto
  from tensorflow.compat.v1 import InteractiveSession

  config = ConfigProto()
  config.gpu_options.allow_growth = True
  session = InteractiveSession(config=config)
except Exception as e:
  print(e)
  print("Not possible to set gpu allow growth")


# import rp layer and sp metrics
from classifier_tutorial import sp, get_output_from

# import tensorflow/keras wrapper
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Conv1D, Flatten, Reshape
from keras.models import Model

# import numpy
import numpy as np

# import sklearn things
from sklearn.utils.class_weight import compute_class_weight



def norm1( data ):
  norms = np.abs( data.sum(axis=1) )
  norms[norms==0] = 1
  return data/norms[:,None]


def reshape_to_conv1d( data ):
  data = np.array([data])
  return np.transpose(data, [1,2,0])

# create the cv and split in train/validation samples just for sp validation
file = '../../et1520/data17_13TeV.AllPeriods.sgn.probes_lhmedium_EGAM1.bkg.VProbes_EGAM7.GRL_v97_et0_eta0.npz'
raw = dict(np.load(file))
data = raw['data'][:,1:101]
data = reshape_to_conv1d(norm1(data))
target = raw['target']

# get pd and fa
ref_key = 'T0HLTElectronT2CaloTight'
d = raw['data'][:,np.where(raw['features'] == ref_key)[0][0]-1]
d_s = d[target==1]
d_b = d[target!=1]

# Get the detection probability as reference
pd = sum(d_s)/len(d_s) # passed==1 / total
# Get the fake probability as reference
fa = sum(d_b)/len(d_b) # passed==1 / total

# release memory
del raw


print(data.shape)



# Create all necessary splits to separate the data in train and validation sets
# Here, we will use only the fist "sort" just for testing
from sklearn.model_selection import StratifiedKFold, KFold
kf = StratifiedKFold(n_splits=10, random_state=512, shuffle=True)
splits = [(train_index, val_index) for train_index, val_index in kf.split(data,target)]

# # shuffle, sort = 0
# x = data [ splits[0][0] ]
# y = target [ splits[0][0] ]
# x_val = data [ splits[0][1] ]
# y_val = target [ splits[0][1] ]





model = Sequential()
model.add(Conv1D(16, kernel_size=2, activation='relu', input_shape=(100,1) ))
model.add(Conv1D(32, kernel_size=2, activation='relu' ))
model.add(Flatten())
model.add(Dense(32,  activation='relu'))
model.add(Dense(1, activation='linear'))
model.add(Activation('sigmoid'))


# compile the model
model.compile( 'adam',
               loss ='binary_crossentropy',
               metrics = ['acc'],
              )


sp_obj = sp(patience=25, verbose=True, save_the_best=True)
# sp_obj.set_validation_data( (x_val, y_val) )


for i in splits:
  # shuffle, sort = 0
  x = data[i[0]]
  y = target[i[0]]
  x_val = data[i[1]]
  y_val = target[i[1]]

  sp_obj.set_validation_data( (x_val, y_val) )
  # train the model
  history = model.fit(x, y,
            epochs          = 10,
            batch_size      = 1024,
            verbose         = True,
            validation_data = (x_val,y_val),
            callbacks       = [sp_obj],
            class_weight    = compute_class_weight('balanced',np.unique(y),y),
            shuffle         = True)


model.pop() 
model.pop() 
head = model.output


head = Reshape((32, 1))(head)
head = Conv1D(16, kernel_size=2, activation='relu', input_shape=(100,1) )(head) 
head = Conv1D(32, kernel_size=2, activation='relu' )(head)
head = Flatten()(head) 
head = Dense(32,  activation='relu')(head)
head = Dense(1, activation='linear')(head)
head = Activation('sigmoid')(head) 
new_model = Model(inputs=model.input, outputs=head)


for layer in model.layers:
  layer.trainable = False

new_model.compile( 'adam',
               loss ='binary_crossentropy',
               metrics = ['acc'],
              )

# create the cv and split in train/validation samples just for sp validation
file = '../../others/17_13/data17_13TeV.AllPeriods.sgn.probes_lhmedium_EGAM2.bkg.VProbes_EGAM7.GRL_v97/data17_13TeV.AllPeriods.sgn.probes_lhmedium_EGAM2.bkg.VProbes_EGAM7.GRL_v97_et2_eta0.npz'
raw = dict(np.load(file))
data = raw['data'][:,1:101]
data = reshape_to_conv1d(norm1(data))
target = raw['target']

# get pd and fa
ref_key = 'T0HLTElectronT2CaloTight'
d = raw['data'][:,np.where(raw['features'] == ref_key)[0][0]-1]
d_s = d[target==1]
d_b = d[target!=1]

# Get the detection probability as reference
pd = sum(d_s)/len(d_s) # passed==1 / total
# Get the fake probability as reference
fa = sum(d_b)/len(d_b) # passed==1 / total

# release memory
del raw


print(data.shape)



# Create all necessary splits to separate the data in train and validation sets
# Here, we will use only the fist "sort" just for testing
from sklearn.model_selection import StratifiedKFold, KFold
kf = StratifiedKFold(n_splits=10, random_state=512, shuffle=True)
splits = [(train_index, val_index) for train_index, val_index in kf.split(data,target)]




for i in splits:
  # shuffle, sort = 0
  x = data[i[0]]
  y = target[i[0]]
  x_val = data[i[1]]
  y_val = target[i[1]]

  sp_obj = sp(patience=25, verbose=True, save_the_best=True)
  sp_obj.set_validation_data( (x_val, y_val) )


  # train the model
  history = new_model.fit(x, y,
            epochs          = 10,
            batch_size      = 1024,
            verbose         = True,
            validation_data = (x_val,y_val),
            callbacks       = [sp_obj],
            class_weight    = compute_class_weight('balanced',np.unique(y),y),
            shuffle         = True)





