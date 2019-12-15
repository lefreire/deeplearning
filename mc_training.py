from utilities import *
from model import *
from sklearn.model_selection import train_test_split
import keras
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt


class TrainFineTuning:

	def __init__(self, et_value, eta_value):
		self.et_value = et_value
		self.eta_value = eta_value
		self.mod = ModelK()
		self.utils = Utilities()

	def get_file(self):
		file_name = '../mc/mc15_13TeV.sgn.probes_lhmedium_Zee.bkg.Truth.JF17_et'+str(self.et_value)+'_eta'+str(self.eta_value)+'.npz'
		data, target = self.utils.read_file(file_name)
		return data, target

	def init_model(self):
		init = keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)        
		self.mod.construct_model(init)
		return init

	def train_first(self, data, target):
		kf = StratifiedKFold(n_splits=10, random_state=512, shuffle=True)
		splits = [(train_index, val_index) for train_index, val_index in kf.split(data, target)]
		self.mod.training(self.mod.model, data[splits[0][0]], target[splits[0][0]],
								 data[splits[0][1]], target[splits[0][1]], patience_value=25)

	def divide_file(self):
		file_name = '../17_EGAM1/data17_13TeV.AllPeriods.sgn.probes_lhmedium_EGAM1.bkg.VProbes_EGAM7.GRL_v97_et'+str(self.et_value)+'_eta'+str(self.eta_value)+'.npz'
		data, target = self.utils.read_file(file_name)
		X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(data, target, np.arange(len(data)), test_size=0.3, random_state=42)
		return X_train, y_train, X_test, y_test, id_train, id_test 

	def fine_tuning(self, data_train, target_train, splits):
		new_model = self.mod.model_fine_tuning(self.mod.model)
		new_model = self.mod.training_all_data(new_model, data_train, target_train, splits, patience_value=25)
		return new_model

	def predict_value(self, model, data_test):
		return self.mod.predict(model, data_test)

	def define_threshold(self, y_true, y_pred):
		fa, pd, thresholds = roc_curve(y_true, y_pred)
		sp = np.sqrt(  np.sqrt(pd*(1-fa)) * (0.5*(pd+(1-fa))))
		knee = np.argmax(sp)
		return thresholds[knee], fa[knee], pd[knee], sp[knee]

	def define_class(self, y_pred, threshold):
		y_pred_t = []
		for i in y_pred:
			if i >= threshold: y_pred_t.append(1)
			else: y_pred_t.append(0)
		return y_pred_t

	def acc_ratio_plot(self, y_true, y_pred, n_bins, id_test, name):
		raw = self.utils.return_all_file('../17_EGAM1/data17_13TeV.AllPeriods.sgn.probes_lhmedium_EGAM1.bkg.VProbes_EGAM7.GRL_v97_et'+str(self.et_value)+'_eta'+str(self.eta_value)+'.npz')
		ref_key = 'L2Calo_eratio'
		d = raw['data'][:,np.where(raw['features'] == ref_key)[0][0]-1] 
		acc_ratio = [] 
		false_ratio = []
		acc_pred_false_true = []
		false_pred_acc_true = []
		for i in range(0, len(y_true)):
			if y_true[i] == y_pred[i] == 1: acc_ratio.append(d[id_test[i]])
			if y_true[i] == y_pred[i] == 0: false_ratio.append(d[id_test[i]])
			if y_true[i] == 1 and y_pred[i] == 0: false_pred_acc_true.append(d[id_test[i]])
			if y_true[i] == 0 and y_pred[i] == 1: acc_pred_false_true.append(d[id_test[i]])
		# para os que aceitaram
		hist, bin_edges = np.histogram(acc_ratio, bins=200)
		normalized = (hist-min(hist))/(max(hist)-min(hist))  
		plt.plot(bin_edges[:200], normalized, 'o')
		plt.title('Eventos aceitos nos dois casos')
		plt.xlabel('Eratio')
		plt.ylabel('Frequência') 
		plt.savefig(str(name)+'all_aceito.png')
		plt.clf()

		# para os que não aceitaram
		hist, bin_edges = np.histogram(false_ratio, bins=200)
		normalized = (hist-min(hist))/(max(hist)-min(hist))  
		plt.plot(bin_edges[:200], normalized, 'o')
		plt.title('Eventos não aceitos nos dois casos')
		plt.xlabel('Eratio')
		plt.ylabel('Frequência') 
		plt.savefig(str(name)+'all_nao_aceito.png')
		plt.clf()

		# para os que foram aceitados mas preditos que não
		hist, bin_edges = np.histogram(false_pred_acc_true, bins=200)
		normalized = (hist-min(hist))/(max(hist)-min(hist))  
		plt.plot(bin_edges[:200], normalized, 'o')
		plt.title('Eventos aceitos')
		plt.xlabel('Eratio')
		plt.ylabel('Frequência') 
		plt.savefig(str(name)+'aceito_true_falso_pred.png')
		plt.clf()

		# para os que foram aceitados mas preditos que não
		hist, bin_edges = np.histogram(acc_pred_false_true, bins=200)
		normalized = (hist-min(hist))/(max(hist)-min(hist))  
		plt.plot(bin_edges[:200], normalized, 'o')
		plt.title('Eventos não aceitos')
		plt.xlabel('Eratio')
		plt.ylabel('Frequência') 
		plt.savefig(str(name)+'aceito_false_true_pred.png')
		plt.clf()


# tft = TrainFineTuning(2, 0)
# data_train, target_train = tft.get_file()
# x_train, y_train, x_test, y_test = tft.divide_file()
# kf = StratifiedKFold(n_splits=10, random_state=512, shuffle=True)
# splits = [(train_index, val_index) for train_index, val_index in kf.split(x_train, y_train)]
# all_weights = []
# all_acc = []
# all_weight_model = []
# for i in range(0, 5):
# 	print("ESTOU NA INICIALIZACAO ", i) 
# 	all_weights.append(tft.init_model())
# 	tft.train_first(data_train, target_train)
# 	model = tft.fine_tuning(x_train, y_train, splits)
# 	all_acc.append(tft.mod.histories)
# 	all_weight_model.append(tft.mod.weights_model)


