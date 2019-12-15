from keras.models import Sequential 
from keras.layers import Dense, Dropout, Activation, Conv1D, Flatten, Reshape
from classifier_tutorial import sp, get_output_from
from sklearn.model_selection import StratifiedKFold, KFold
from keras.models import Model
from sklearn.utils.class_weight import compute_class_weight
import numpy as np


class ModelK:
	def __init__(self):
		self.model = Sequential()
		self.sp_obj = sp(patience=25, verbose=True, save_the_best=True)
		self.histories = []
		self.weights_model = []

	def construct_model(self, initializer):
		#add model layers
		self.model.add(Conv1D(16, kernel_size=2, activation='relu', input_shape=(100,1), kernel_initializer=initializer))
		self.model.add(Conv1D(32, kernel_size=2, activation='relu', kernel_initializer=initializer))
		self.model.add(Flatten())
		self.model.add(Dense(32,  activation='relu', kernel_initializer=initializer))
		self.model.add(Dense(1, activation='linear', kernel_initializer=initializer))
		self.model.add(Activation('sigmoid'))
		#compile model using accuracy to measure model performance
		self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

	def model_fine_tuning(self, model):
		model.pop()
		model.pop()
		head_model = model.output
		head_model = Reshape((32, 1, ))(head_model)
		head_model = Conv1D(16, kernel_size=2, activation='relu', input_shape=(100,1))(head_model) 
		head_model = Conv1D(32, kernel_size=2, activation='relu' )(head_model)
		head_model = Flatten()(head_model) 
		head_model = Dense(32,  activation='relu')(head_model)
		head_model = Dense(1, activation='linear')(head_model)
		head_model = Activation('sigmoid')(head_model) 
		new_model = Model(inputs=model.input, outputs=head_model)

		for layer in model.layers:
			layer.trainable = False

		new_model.compile( 'adam',
		               loss ='binary_crossentropy',
		               metrics = ['acc'],
		              )
		return new_model


	def training(self, model, x, y, x_val, y_val, patience_value=25):
		self.sp_obj = sp(patience=patience_value, verbose=True, save_the_best=True)
		return model.fit(x, y,
						            epochs          = 300,
						            batch_size      = 1024,
						            verbose         = False,
						            validation_data = (x_val,y_val),
						            callbacks       = [self.sp_obj],
						            class_weight    = compute_class_weight('balanced',np.unique(y),y),
						            shuffle         = True)

	def training_all_data(self, model, x, y, splits, patience_value=25):
		for i in splits:
			# shuffle, sort = 0
			x_train = x[i[0]]
			y_train = y[i[0]]
			x_val = x[i[1]]
			y_val = y[i[1]]
			self.sp_obj = sp(patience=patience_value, verbose=True, save_the_best=True)
			self.sp_obj.set_validation_data( (x_val, y_val))
			# print("FOLD: ", i)
			hist = self.training(model, x_train, y_train, x_val, y_val)
			histories_fold = {}
			histories_fold['val_loss'] = hist.history['val_loss'][-1]
			histories_fold['val_acc'] = hist.history['val_acc'][-1]
			histories_fold['loss'] = hist.history['loss'][-1]
			histories_fold['acc'] = hist.history['acc'][-1]
			histories_fold['max_sp_val'] = hist.history['max_sp_val'][-1]
			histories_fold['max_sp_fa_val'] = hist.history['max_sp_fa_val'][-1]
			histories_fold['max_sp_pd_val'] = hist.history['max_sp_pd_val'][-1]
			histories_fold['max_sp_best_epoch_val'] = hist.history['max_sp_best_epoch_val'][-1]
			self.histories.append(histories_fold)
			self.weights_model.append(model.get_weights())
		best_sp = []
		for i in self.histories:
			best_sp.append(i['max_sp_val'])
		model.set_weights(self.weights_model[np.argmax(best_sp)])
		return model


	def predict(self, model, x):
		return model.predict(x, batch_size=1024) 

	def all_model(self, X_train, y_train, X_test, y_test):
		print(X_train.shape)
		self.construct_model()
		return self.training(X_train, y_train, X_test, y_test)