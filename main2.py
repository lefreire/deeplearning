from mc_training import *
from sklearn.metrics import accuracy_score 
from model import *

print("TESTE")
try:
  from tensorflow.compat.v1 import ConfigProto
  from tensorflow.compat.v1 import InteractiveSession

  config = ConfigProto()
  config.gpu_options.allow_growth = True
  session = InteractiveSession(config=config)
except Exception as e:
  print(e)
  print("Not possible to set gpu allow growth")


tft = TrainFineTuning(0, 0)
data_train, target_train = tft.get_file()
x_train, y_train, x_test, y_test, id_train, id_test = tft.divide_file()
kf = StratifiedKFold(n_splits=10, random_state=512, shuffle=True)
splits = [(train_index, val_index) for train_index, val_index in kf.split(x_train, y_train)]
all_weights = []
all_acc = []
all_weight_model = []



print("ESTOU NA INICIALIZACAO ") 
all_weights.append(tft.init_model())
tft.train_first(data_train, target_train)
model = tft.fine_tuning(x_train, y_train, splits)
all_acc.append(tft.mod.histories)
all_weight_model.append(tft.mod.weights_model)
y_predict = tft.predict_value(model, x_test)
threshold, fa, pd, sp = tft.define_threshold(y_test, y_predict)
y_pred_class = tft.define_class(y_predict, threshold)
tft.acc_ratio_plot(y_test, y_pred_class, 200, id_test, 'fine-tuning')

print("ACURACIA COM FINE TUNING: ")
print(all_acc)
# for i in all_acc:
# 	for j in i:
# 		print(j.history)
print(all_weight_model)
print("ACURACIA DO TESTE: ")
print(accuracy_score(y_test, y_pred_class), fa, pd, sp) 


all_acc_normal = []
all_weight_normal = []
print("CNN NORMAL")
mod = ModelK()
mod.construct_model(all_weights[0])
mod.training_all_data(mod.model, x_train, y_train, splits, patience_value=25)
all_acc_normal.append(mod.histories)
all_weight_normal.append(mod.weights_model)
y_predict = tft.predict_value(mod.model, x_test)
threshold, fa, pd, sp = tft.define_threshold(y_test, y_predict)
y_pred_class = tft.define_class(y_predict, threshold)
tft.acc_ratio_plot(y_test, y_pred_class, 200, id_test, 'normal')

print("ACURACIA NORMAL: ")
print(all_acc)
# for i in all_acc:
# 	for j in i:
# 		print(j.history)
print(all_weight_model)
print("ACURACIA DO TESTE SEM FINE TUNING: ")
print(accuracy_score(y_test, y_pred_class), fa, pd, sp) 

