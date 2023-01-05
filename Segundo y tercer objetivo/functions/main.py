import tempfile
import numpy as np
import pandas as pd
from scipy import stats
import tensorflow as tf
from sklearn import metrics
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator
from sklearn.metrics import confusion_matrix
from sklearn.metrics import pairwise_distances
from sklearn.cluster import SpectralClustering
from sklearn.model_selection import ShuffleSplit
from sklearn.utils.multiclass import unique_labels
from sklearn.cluster import KMeans, SpectralClustering
 

def conv_to_list(arr):
    """Ajustar el conjunto a forma Deep and Wide.

    Parameters
    ----------
    arr : {array-like, sparse matrix} de la forma (número de muestras, canales, tiempo, banda de frecuencia, metodo (csp o cwt) )
      Conjunto de  muestras.

    Returns
    -------
    X_pre : {array-like, sparse matrix} de la forma (número de muestras, canales, tiempo, banda de frecuencia, metodo (csp o cwt) )
      Conjunto de  muestras ordenado.
    """
    x_pre = []
    for i in range(arr.shape[-1]):
        x_pre.append(arr[:, :, :, :, i])
    return x_pre
   
def plot_model_(history_):
    pd.DataFrame(history_.history).plot(figsize=(5, 5))
    plt.grid(True)
    plt.show()
 
class CNNrITL(BaseEstimator):
  '''Red neuronal convolucional de tipo Deep and Wide ó solo Wide que incorpora cuya funcion de costo 
  utiliza herramientas de teoria de informacion, de manera que se tenga una funcion de costo regularizada que se inspira en 
  el principio de informacion revelante o PRI por sus siglas en ingles.

  Parámetros
  ----------

  d: float, default=2
    Dimensiones de la capa densa, sin embargo este valor se ve influenciado por el tamaño de la capa Flatten.
  
  sigma: float, default=None 
    Sigma para la funcion kernel de la capa de Random Fourier Features.

  k: int, default=2
    Numero de posibles salidas de la red neuronal.

  verbose: int, default=10
    Detalles de entrenamiento de la red neuronal.

  n_fill: int, default=1
    Cantidad de filtros para la capa convolucional.

  epochs: int, default=200
    Cantidad de epocas de entrenamiento para la red neuronal.

  batch_size: int, default=128
    Tamaño del lote para entrenamiento de la red neuronal.
  
  lr: float, default=1e-3
    Coeficiente de aprendizaje de la red neuronal.

  sl: str, default='ritl'
    Posibles funciones de costo de la red neuronal.
    'ce' para solo crossentropy.
    'MSE' para MeanSquaredError.
  
  lk: float, default=0.5
    Nivel de supervicion para la funcion de costo ritl, siendo 0 el menor valor y 1 el mayor.

  l1: float, default=1e-3
    Constante de regularizacion L1 y L2 para la funcion densa.
  
  wi: bool, default=True
    Modo Deep an Wide.

  plot_model: bool, default=True
    Mostrar graficas de rendimiento durante la fase de entrenamiento
  
  Atributos
  ---------
  model : Tensor
    Un tensor originado de una entrada o un conjunto de entradas (Deep and Wide o Wide). 

  labels_ : ndarray de forma (número de muestras)
    Etiquetas estimadas para cada una de las muestras.

  '''
  def __init__(self, d=2, sigma=None, k=2, verbose=10, n_fil=1,
                epochs=200, batch_size=128, lr=1e-3, sl='ritl',
                lk=0.5, l1=1e-3, wi=True, plot_model=True):
    self.verbose = verbose
    self.sigma = sigma
    self.d = d
    self.n_fil = n_fil
    self.k = k
    self.epochs = epochs
    self.batch_size = batch_size
    self.lr = lr
    self.sl = sl
    self.lk = lk
    self.l1 = l1
    self.wi = wi
    self.plot_model = plot_model
      

  def transform(self, X, *_):
    return self.model.predict(X)

  def fit_transform(self, X, y):
    self.fit(X, y)
    return self.transform(X)

  def fit(self, X, y=None):
    self.model = self.main(X, y)
    return self

  def main(self, x, y):

    # --------------------------------------------------
    # deep and wide mode for BCI topoplots
    if self.wi:
      batch0_ = [None] * x.shape[-1]
      input_ = [None] * x.shape[-1]
      conv_ = [None] * x.shape[-1]
      pool_ = [None] * x.shape[-1]
      batch1_ = [None] * x.shape[-1]
      for i in range(x.shape[-1]):
        input_[i] = keras.layers.Input(shape=x.shape[1:-1])
        batch0_[i] = keras.layers.BatchNormalization()(input_[i])
        # ------------------------------------------------------------------------------------------------------
        conv_[i] = keras.layers.Conv2D(filters=self.n_fil, kernel_size=5, strides=1, activation='relu',
                                        padding='SAME', input_shape=x.shape[1:-1],
                                        kernel_regularizer=tf.keras.regularizers.l1_l2(l1=self.l1, l2=self.l1),
                                        kernel_initializer="GlorotNormal")(batch0_[i])
        # ------------------------------------------------------------------------------------------------------
        pool_[i] = keras.layers.MaxPooling2D(pool_size=2)(conv_[i])
        # ------------------------------------------------------------------------------------------------------
        batch1_[i] = keras.layers.BatchNormalization()(pool_[i])

      concat = keras.layers.concatenate(batch1_)
      flat = keras.layers.Flatten()(concat)
      dim = np.round(self.d * batch1_[i].shape[1] * batch1_[i].shape[2])
    else:  # only deep mode
      input_ = keras.layers.Input(shape=x.shape[1:])
      # ----------------------------------------------------------------------------------------------------------
      conv_ = keras.layers.Conv2D(filters=self.n_fil, kernel_size=5, strides=1, activation='relu',
                                  padding='SAME', input_shape=x.shape[1:],
                                  kernel_regularizer=tf.keras.regularizers.l1_l2(l1=self.l1, l2=self.l1))(input_)
      # ----------------------------------------------------------------------------------------------------------
      pool_ = keras.layers.MaxPooling2D(pool_size=2, name='pool')(conv_)
      # ----------------------------------------------------------------------------------------------------------
      batch1_ = tf.keras.layers.BatchNormalization(name='batchF')(pool_)
      # ----------------------------------------------------------------------------------------------------------
      flat = keras.layers.Flatten()(batch1_)
      dim = np.round(self.d * flat.shape[1])
    # multilayer dense  perceptron with rff
    # -----------------------------------------------------------------------------------------
    flat_do = tf.keras.layers.Dropout(rate=0.25)(flat)
    d1 = tf.keras.layers.Dense(dim.astype('int'), activation='linear', kernel_initializer="GlorotNormal",
                                kernel_regularizer=tf.keras.regularizers.l1_l2(l1=self.l1, l2=self.l1))(flat_do)
    h1 = tf.keras.layers.experimental.RandomFourierFeatures(output_dim=dim.astype('int'),
                                                            scale=self.sigma, kernel_initializer='gaussian',
                                                            trainable=True, name='rbf_fourier')(d1)
    h1_bn = tf.keras.layers.BatchNormalization(name='brff')(h1)
    output_prob = tf.keras.layers.Dense(self.k, activation='softmax', name='out', kernel_initializer="GlorotNormal",
                                        kernel_regularizer=tf.keras.regularizers.l1_l2(l1=self.l1, l2=self.l1))(
        h1_bn)
    # define loss and compile
    opt = tf.keras.optimizers.Adam(learning_rate=self.lr)
    if self.sl == 'ritl':
      model = tf.keras.Model(inputs=input_, outputs=[output_prob, h1_bn])
      model_loss = [tf.keras.losses.CategoricalCrossentropy(), self.custom_ritl()]
      model.compile(loss=model_loss, loss_weights=[self.lk, 1 - self.lk], optimizer=opt,
                    metrics=['accuracy'])  # f1, precision, re
    elif self.sl == 'ce':
      model = tf.keras.Model(inputs=input_, outputs=output_prob)
      model_loss = [tf.keras.losses.CategoricalCrossentropy()]
      model.compile(loss=model_loss, optimizer=opt, metrics=['accuracy'])  # f1, precision, re
    else:
      model = tf.keras.Model(inputs=input_, outputs=output_prob)
      model_loss = [tf.keras.losses.MeanSquaredError()]
      model.compile(loss=model_loss, optimizer=opt, metrics=['accuracy'])

    rs = ShuffleSplit(n_splits=1, test_size=.1)

    # validation data
    for train_index, valid_index in rs.split(x):
      if self.wi:

          x_pre = conv_to_list(x[train_index])
          x_pre_v = conv_to_list(x[valid_index])

      else:
          x_pre = x[train_index]
          x_pre_v = x[valid_index]

    y_pre = keras.utils.to_categorical(y, self.k)  #

    if self.sl == 'ritl':
      history = model.fit(x_pre, [y_pre[train_index], y_pre[train_index]],
                          epochs=self.epochs,
                          batch_size=self.batch_size,
                          validation_data=(x_pre_v, [y_pre[valid_index], y_pre[valid_index]]),
                          verbose=self.verbose)
    else:
      history = model.fit(x_pre, y_pre[train_index],
                          epochs=self.epochs,
                          batch_size=self.batch_size,
                          validation_data=(x_pre_v, y_pre[valid_index]),
                          verbose=self.verbose)
    if self.plot_model:
      plot_model_(history)

    return model

  def predict(self, x_test):
    """Predice la pertenencia de nuevas muestras al grupo estimado mas cercano.

    Parameters
    ----------
    x_test : {array-like, sparse matrix} de la forma (número de muestras, número de características)
      Conjunto de nuevas muestras.

    Returns
    -------
    label_e : ndarray de la forma (número de muestras,)
      Etiqueta a la cual cada una de las muestras es asociada.
    """
    if self.wi:
      x_test_pre = conv_to_list(x_test)
    else:
      x_test_pre = x_test

    if self.sl == 'ritl':
      y_prob = np.stack(
          [self.model(x_test_pre, training=True)[0]
            for sample in range(100)])
    else:
      y_prob = np.stack(
          [self.model(x_test_pre, training=True)
            for sample in range(100)])

    y_prob = y_prob.mean(axis=0)
    label_e = np.argmax(y_prob, axis=1)

    return label_e

  def custom_ritl(self):
    def custom_kitl(y_true, y_pred):
      # -------------------------kernel----------------------------
      k = tf.matmul(y_pred, y_pred, transpose_b=True)
      # ----------------------center-------------------------------
      N = tf.cast(tf.shape(k)[0], dtype=tf.float32)
      # matrix for centered kernel
      h = tf.eye(N) - (1.0 / N) * tf.ones([N, 1]) * tf.ones([1, N])
      k = tf.matmul(tf.matmul(k, h), tf.matmul(k, h))
      k = tf.math.divide_no_nan(k, tf.linalg.trace(k))
      # ------------------------F_initial--------------------------
      f = -tf.math.log(tf.linalg.trace(tf.matmul(k, k) + 1E-5))
      return -f

    return custom_kitl

  def get_params(self, deep=True):

    return {'d': self.d, 'k': self.k, 'epochs': self.epochs, 'batch_size': self.batch_size,
            'lr': self.lr, 'sl': self.sl, 'lk': self.lk, 'l1': self.l1,
            'wi': self.wi,  'verbose': self.verbose, 'n_fil': self.n_fil,
            'plot_model': self.plot_model}

  def set_params(self, **parameters):
    for parameter, value in parameters.items():
      setattr(self, parameter, value)
    return self