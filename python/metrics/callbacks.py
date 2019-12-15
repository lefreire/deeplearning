
__all__ = ["sp", "monit"]

from tensorflow.keras.callbacks import Callback
from sklearn.metrics import roc_curve
import numpy as np


class sp(Callback):

  def __init__(self, verbose=False,save_the_best=False, patience=False, **kw):
    super(Callback, self).__init__()
    self.__verbose = verbose
    self.__patience = patience
    self.__ipatience = 0
    self.__best_sp = 0.0
    self.__save_the_best = save_the_best
    self.__best_weights = None
    self.__best_epoch = 0
    self._validation_data = None

  def set_validation_data( self, v ):
    self._validation_data = v



  def on_epoch_end(self, epoch, logs={}):
    if self._validation_data: # Tensorflow 2.0
      y_true = self._validation_data[1]
      y_pred = self.model.predict(self._validation_data[0],batch_size=1024).ravel()
    else:
      y_true = self.validation_data[1]
      y_pred = self.model.predict(self.validation_data[0],batch_size=1024).ravel()

    fa, pd, thresholds = roc_curve(y_true, y_pred)

    sp = np.sqrt(  np.sqrt(pd*(1-fa)) * (0.5*(pd+(1-fa)))  )

    knee = np.argmax(sp)
    logs['max_sp_val'] = sp[knee]
    logs['max_sp_fa_val'] = fa[knee]
    logs['max_sp_pd_val'] = pd[knee]
    if self.__verbose:
      print( (' - val_sp: %1.4f (fa:%1.4f,pd:%1.4f), patience: %d') % (sp[knee],fa[knee],pd[knee], self.__ipatience) )


    if sp[knee] > self.__best_sp:
      self.__best_sp = sp[knee]
      if self.__save_the_best:
        print('save the best configuration here...' )
        self.__best_weights =  self.model.get_weights()
        logs['max_sp_best_epoch_val'] = epoch
      self.__ipatience = 0
    else:
      self.__ipatience += 1

    if self.__ipatience > self.__patience:
      print('Stopping the Training by SP...')
      self.model.stop_training = True



  def on_train_end(self, logs={}):
    # Loading the best model
    if self.__save_the_best:
      print('Reload the best configuration into the current model...')
      try:
        self.model.set_weights( self.__best_weights )
      except:
        print( "Its not possible to set the weights. Maybe there is some" +
            "problem with the train split (check the quantity and kfold method.)")







class monit(Callback):

  def __init__(self, **kw):
    super(Callback, self).__init__()
    self.__weights = list()



  def on_epoch_end(self, epoch, logs={}):
    self.__weights.append(self.model.get_weights())


  def getWeights(self):
    return self.__weights








