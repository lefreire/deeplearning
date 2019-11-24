from keras.models import Sequential 
from keras.layers import Dense, Conv1D, Flatten

class Model:
	def __init__(self):
		self.model = Sequential()

	def construct_model(self):
		#add model layers
		self.model.add(Conv1D(64, kernel_size=2, activation='relu', input_shape=(100, 1)))
		self.model.add(Conv1D(32, kernel_size=2, activation='relu'))
		self.model.add(Flatten())
		self.model.add(Dense(1, activation='softmax'))
		#compile model using accuracy to measure model performance
		self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

	def training(self, X_train, y_train, X_test, y_test):
		return self.model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3)

	def all_model(self, X_train, y_train, X_test, y_test):
		print(X_train.shape)
		self.construct_model()
		return self.training(X_train, y_train, X_test, y_test)