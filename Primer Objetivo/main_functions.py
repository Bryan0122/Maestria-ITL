import numpy as np
import time as time
from scipy import stats
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.base import BaseEstimator, TransformerMixin, ClusterMixin


class PRI(BaseEstimator):

    '''Método del principio de información relevante o PRI por sus siglas en ingles con los algoritmos de punto fijo y gradiente descendiente. Adicional al método de representación se utilizan los algoritmos de K-means y SpectralClustering para la estimación de grupos. 
    Parámetros
    ----------

    lambda_ : float, default=2
      Constante de ponderación que regula el nivel preservación de la estructura de los datos originales.

    sigma_initial: float, default=30
      Constate que controla el ancho de la función kernel.

    yota: float, default=1
      Constante de decaimiento del parámetro sigma.

    max_iter: int, default=250
      Máximo número de iteraciones.

    reduction: String, default='Kmeans'
      Parámetro de selección de algoritmo de detección de grupos. 
      'SC' para SpectralClustering,
      'Kmeans' para K-Means.

    n_clusters: int, default=5
      Numero de grupos para los algoritmos de SpectralCLustering o K-means.

    nn: int, default=3
      Número de vecinos para predecir la pertenecía de una muestra nueva a uno de los grupos.

    ss: float, default=1
      Nivel de partición del conjunto de datos originales.

    method: string, default='FP'
      Algoritmo de minimización de la función de costo del PRI.
      'FP'= Punto fijo.
      'SGD'= Gradiente descendiente estocástico.

    optimization: string, default= None
      Funciones de optimización para el algoritmo de Gradiente descendiente estocastico 'Adam' o 'Nadam'.

    learning_rate: float, default=0.001
      Constante de aprendizaje para el algoritmo de Gradiente descendiente estocástico.

    prediction: string, default=None
      Método de predecir la pertenecía de una nueva etiqueta.
      'Gaussian': kernel 
      'Div': divergence 

    show_process: bool, default=False
      Muestra el desarrollo de la clase a lo largo de las iteraciones.

    
    Atributos
    ---------
    
    cluster_centers_ : ndarray de forma (número de grupos, número de caracteristicas)
    Coordinadas de los grupos estimados.

    labels_ : ndarray de forma (número de muestras)
    Etiquetas estimadas para cada una de las muestras.

    Adicionales
    -----------

    MiniBatchPRI : Una solución alternativa la cual utiliza el la función de costo del PRI y la combina con el algoritmo de gradiente por lotes, dicha función se puede encontrar en la clase con el mismo nombre

    
    Nota
    ----

    La función de costo del PRI
    
    J(X|Xo) = H2(X) + λDcs(X,Xo)
    Cuando λ --> ∞
    J(X|Xo) = -(1-λ)log(V(X))-2λlog(V(X,Xo))
    V(X) = 1/N^2 ∑_(i=1)^N ∑_(j=1)^N k(xi,xj)
    V(X,Xo) = 1/(NNo) ∑_(i=1)^N∑_(j=1)^No 

    '''

    def __init__(self, lambda_=2, sigma_initial=30, yota=1, max_iter=250,  reduction_=None, n_clusters=5, nn=5, ss=1, method='FP', optimization = None, learning_rate=0.001,  prediction=None, show_process=False, gamma_=1000):

        self.lambda_ = lambda_
        self.sigma_initial = sigma_initial
        self.yota = yota
        self.max_iter = max_iter
        self.reduction_ = reduction_
        self.n_clusters = n_clusters
        self.nn = nn
        if ss > 1:
            self.ss = 1
        else:
            self.ss = ss
        self.method = method
        self.optimization = optimization
        self.learning_rate = learning_rate
        self.prediction = prediction
        self.show_process = show_process
        self.gamma_ = gamma_

    def fit(self, X, y=None):
        
        """ Calcula la representación de los datos mediante el PRI.

        Parámetros
        ----------
        X : {ndarray, sparse matrix} de la forma (número de muestras, numero de características)
            Conjunto de datos de entrada.

        y : No es usado, se presenta para la siguiente clase por convenciones establecidas
        
        Retorno
        -------
        cluster_centers_ : ndarray de forma (número de grupos, numero de características)
        Coordinadas de los grupos estimados.

        labels_ : ndarray de forma (número de muestras)
        Etiquetas estimadas para cada una de las muestras.
        """


        self.y = y
        try:
            X[:, 0] = X[:, 0]
        except:
            X = np.array(X).astype(float)

        self.X = X
        self.cluster_centers_, self.labels_ = self.pri_fuction(self.X)
        return self

    def pri_fuction(self, Xo):
        """  Calcula las etiquetas y el conjunto de representación.

        Parámetros
        ----------
        X : {ndarray, sparse matrix} de la forma (número de muestras, número de características)
            Conjunto de datos de entrada.
        
        Retorno
        -------
        self : Objetos resultantes del estimador ajustado.
        """

        #################################################### Preparar los datos #######################################################
        X = np.zeros((np.round(Xo.shape[0] * self.ss), Xo.shape[1]))
        for i in range(Xo.ndim):

            X[:, i] = np.random.uniform(low=np.amin(
                Xo[:, i]) - abs(np.amin(Xo[:, i]) * 0.2), high=np.amax(Xo[:, i]) + abs(np.amin(Xo[:, i]) * 0.2), size=(np.round(Xo.shape[0] * self.ss),))

        ##################################################### MAIN #####################################################################

        NX = X.shape[0]
        NXo = Xo.shape[0]
        sigma = np.mean(pairwise_distances(Xo, X))
        sigmai = sigma
        Xr = Xo
        i = 0
        J = []
        D_ = []
        labels = np.zeros(Xo.shape[0])

        if self.optimization == 'Adam':
            optimization_model = Adam(self.learning_rate )
        elif self.optimization == 'Nadam':
            optimization_model = Nadam(self.learning_rate)
        else:
            optimization_model = Gd(self.learning_rate)

        while (i < self.max_iter ):

            K1, K2, K3, V1, _, V3 = Kernel_Estimation(
                sigma).fit(NX, NXo, X, Xo)
            if V1 == 0 or V3 == 0:

                break

            # Cost Function

            J.append(-(1 - self.lambda_) * np.log2(V1)
                     - 2 * self.lambda_ * np.log2(V3))


            if self.show_process == True:

                A = -np.log2(np.sum((1 / NX * NXo) * K1))
                B = -np.log2(np.sum((1 / NX**2) * np.ones((Xo.shape[0], X.shape[0])) @ K2))
                C = -np.log2(np.sum((1 / NXo**2) * K3 @ np.ones((Xo.shape[0], X.shape[0]))))
                D = 2 * A - B - C

                D_.append(D)
                plt.figure(figsize=(15,5))
                plt.ion()
                show_proc(Xo, X, i, J, D_)

            # Update Xk
            Xk = X
            if self.method == 'FP':

                if self.lambda_ != 0:

                    X = Fp(self.lambda_).step(NXo, NX, V1, V3, K1, K3, Xo, X)
                else:
                    num = K3 @ np.ones(Xo.shape)
                    X = (K3 @ Xk / num)

            elif self.method == 'SGD':

                # Calculate gradient

                FXk = 1 / (NX * sigma**2) * \
                    (K1 @ np.ones(X.shape) * Xk - K1 @ Xk)
                FXo = 1 / (NXo * sigma**2) * \
                    (K3 @ np.ones(Xo.shape) * Xk - K3 @ Xo)
                g = -2 * FXk/ V1 + 2 * FXo / V3

                # if you choose some method of optimization Adam or Nadam

                if self.optimization != None:
                    X = optimization_model.step(g, X)

                else:
                    X = optimization_model.step(g, X)

            # Update sigma

            sigma = (self.sigma_initial * sigmai) / (self.yota * i + 1) 
            i += 1
            Xf = X

        # if you want reduce the best dataset
        
        if (self.reduction_ is not None):
            # save sigma
            self.sigma = sigma

            if self.reduction_ == 'SC':

                # Sc
                sc = SpectralClustering(
                    n_clusters=self.n_clusters, gamma=self.gamma_, n_neighbors=5)
                labels_ = sc.fit(Xf).labels_
                D = self.Divergencecs(Xo, Xf)
                c = 1
                labels = np.zeros(D.shape[0])
                for i in range(D.shape[0]):
                    labels[i] = stats.mode(
                        labels_[np.argsort(c * D[i, :])[:self.nn]], keepdims=True)[0]

            elif self.reduction_ == 'Kmeans':
                # Kmeans
                Xr = KMeans(n_clusters=self.n_clusters).fit(Xf)
                Xr = Xr.cluster_centers_
                Xf = Xr
                DM = self.Divergencecs(Xo, Xf)
                labels = np.argmin(DM, axis=1)

        if self.show_process == True:
            plt.ioff()

        # organize labels respect to original labels

        if self.y is not None:
            labels = Lconvert().fit(labels, self.y)
        self.J = J

        return Xf, labels

    def Divergencecs(self, Xo, X):
        """  Calcula la divergencia de Cauchy-Schwarz.

        Parámetros
        ----------
        X : {ndarray, sparse matrix} de la forma (número de muestras, número de características)
            Conjunto de datos de entrada.
        Xo : {ndarray, sparse matrix} de la forma (número de muestras, número de características)
        Conjunto de datos procesados.
        
        Retorno
        -------
        D : Divergencia entre de Cachy-Schwarz entre X y Xo.

        """
        N = np.shape(X)[0]
        No = np.shape(Xo)[0]
        K3, K2, K1, _, _, _ = Kernel_Estimation(
            self.sigma).fit(N, No, Xo, X)

        A = -np.log((1 / N * No) * K1 + 1e-4)
        B = -np.log((1 / N**2) * np.ones((Xo.shape[0], X.shape[0])) @ K2)
        C = -np.log((1 / No**2) * K3 @ np.ones((Xo.shape[0], X.shape[0])))
        D = 2 * A - B - C

        return D

    def predict(self, X_test):
        """Predice la pertenencia de nuevas muestras al grupo estimado mas cercano.
        
        Parameters
        ----------
        X_test : {array-like, sparse matrix} de la forma (número de muestras, número de características)
            Conjunto de nuevas muestras.
        Returns
        -------
        labels : ndarray de la forma (número de muestras,)
            Etiqueta a la cual cada una de las muestras es asociada.
        """
        try:
            X_test[:, 0] = X_test[:, 0]
        except:
            X_test = np.array(X_test).astype(float)

        if self.prediction == 'Gaussian':
            Dx1 = pairwise_distances(X_test, self.X)
            D = np.exp(-Dx1**2 / (2 * self.sigma**2))
            c = -1
        else:
            D = self.Divergencecs(X_test, self.X)
            c = 1
        labels = np.zeros(D.shape[0])
        for i in range(D.shape[0]):
            labels[i] = stats.mode(
                self.labels_[np.argsort(c * D[i, :])[:self.nn]], keepdims=True)[0]

        return labels

    def get_params(self, deep=True):

        return {"lambda_": self.lambda_, "sigma_initial": self.sigma_initial, "yota": self.yota, 'max_iter': self.max_iter,  "reduction_": self.reduction_, 'n_clusters': self.n_clusters, 'nn': self.nn, 'ss': self.ss, "method": self.method, 'optimization': self.optimization, 'learning_rate': self.learning_rate,  'prediction': self.prediction, 'show_process': self.show_process, 'gamma_': self.gamma_}

    def set_params(self, **parameters):

        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

# %%
class SC(BaseEstimator, ClusterMixin, TransformerMixin):
    '''Clase basada en el algoritmo de SpectralClustering para las clases de PRI y PRI por lotes
    Parámetros
    ----------

    n_clusters : float, default=2
      Numero de grupos.
    
    gamma_ : float, default=2
      Constante gamma del algoritmo de spectral clustering.
    
    n_neighbors_ : float, default=2
      Numero de vecinos para el algoritmo de SpectralClustering.
    
    nn : float, default=2
      Numero de vecinos para el algoritmo del PRI o PRI por lotes.
    
    Atributos
    ---------
    labels_ : ndarray de forma (número de muestras)
    Etiquetas estimadas para cada una de las muestras.

    '''
    def __init__(self, n_clusters_=2, gamma_=1, n_neighbors_=10, nn=3):
        self.n_clusters_ = n_clusters_
        self.gamma_ = gamma_
        self.n_neighbors_ = n_neighbors_
        self.nn = nn

    def fit(self, X, y=None):
        self.X = X
        self.y = y
        m = SpectralClustering(
            gamma=self.gamma_, n_clusters = self.n_clusters_, n_neighbors=self.n_neighbors_)
        self.labels_ = m.fit(X).labels_
        self.labels_ = Lconvert().fit(self.labels_, self.y)

        return self

    def predict(self, X_test):
        sigma = np.mean(pairwise_distances(X_test, self.X))
        Dx1 = pairwise_distances(X_test, self.X)
        D = np.exp(-Dx1**2 / (2 * sigma**2))
        labels = np.zeros(D.shape[0])

        for i in range(D.shape[0]):
            labels[i] = stats.mode(
                self.labels_[np.argsort(-1 * D[i, :])[:self.nn]], keepdims=True)[0]

        return labels

    def get_params(self, deep=True):

        return {"n_clusters_": self.n_clusters_, "gamma_": self.gamma_, "n_neighbors_": self.n_neighbors_, "nn": self.nn}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():

            setattr(self, parameter, value)
        return self

# %%
class Kmeans(BaseEstimator, ClusterMixin, TransformerMixin):
    '''Clase basada en el algoritmo de K-means para las clases de PRI y PRI por lotes

    Parámetros
    ----------

    n_clusters : float, default=2
      Numero de grupos.
    
    nn : float, default=2
      Numero de vecinos para el algoritmo del PRI o PRI por lotes.
    
    Atributos
    ---------
    labels_ : ndarray de forma (número de muestras)
    Etiquetas estimadas para cada una de las muestras.

    '''
    def __init__(self, n_clusters_=2, nn=3):
        self.n_clusters_ = n_clusters_
        self.nn = nn

    def fit(self, X, y=None):
        self.X = X
        self.y = y
        kn = KMeans(n_clusters = self.n_clusters_)
        self.labels_ = kn.fit(X).labels_
        self.labels_ = Lconvert().fit(self.labels_, self.y)

        return self

    def predict(self, X_test):
        sigma = np.mean(pairwise_distances(X_test, self.X))
        Dx1 = pairwise_distances(X_test, self.X)
        D = np.exp(-Dx1**2 / (2 * sigma**2))
        labels = np.zeros(D.shape[0])

        for i in range(D.shape[0]):
            labels[i] = stats.mode(
                self.labels_[np.argsort(-1 * D[i, :])[:self.nn]], keepdims=True)[0]

        return labels

    def get_params(self, deep=True):

        return {"n_clusters_": self.n_clusters_, "nn": self.nn}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

# %%
class MiniBatchPRI(BaseEstimator, ClusterMixin, TransformerMixin):
    '''Método del principio de información relevante o PRI por sus siglas en ingles con el algoritmo de gradiente descendiente por lotes. Adicional al método de representación se utilizan los algoritmos de K-means y SpectralClustering para la estimación de grupos. 
    
    Parámetros
    ----------
    
    lambda_ : float, default=2
      Constante de ponderación que regula el nivel preservación de la estructura de los datos originales.

    sigma_initial: float, default=30
      Constate que controla el ancho de la función kernel.

    yota: float, default=1
      Constante de decaimiento del parámetro sigma.

    max_epochs: int, default=30
      Máximo número de epocas.

    batch_size: int, default=30
      Tamaño del lote.

    reduction: String, default='Kmeans'
      Parámetro de selección de algoritmo de detección de grupos. 
      'SC' para SpectralClustering,
      'Kmeans' para K-Means.

    n_clusters: int, default=5
      Numero de grupos para los algoritmos de SpectralCLustering o K-means.

    nn: int, default=3
      Número de vecinos para predecir la pertenecía de una muestra nueva a uno de los grupos.

    ss: float, default=1
      Nivel de partición del conjunto de datos originales.

    method: string, default='FP'
      Algoritmo de minimización de la función de costo del PRI.
      'FP'= Punto fijo.
      'SGD'= Gradiente descendiente estocástico.

    optimization: string, default= None
      Funciones de optimización para el algoritmo de Gradiente descendiente estocastico 'Adam' o 'Nadam'.

    learning_rate: float, default=0.001
      Constante de aprendizaje para el algoritmo de Gradiente descendiente estocástico.

    prediction: string, default=None
      Método de predecir la pertenecía de una nueva etiqueta.
      'Gaussian': kernel 
      'Div': divergence 

    show_process: bool, default=False
      Muestra el desarrollo de la clase a lo largo de las iteraciones.

    
    Atributos
    ---------
    
    cluster_centers_ : ndarray de forma (número de grupos, número de caracteristicas)
    Coordinadas de los grupos estimados.

    labels_ : ndarray de forma (número de muestras)
    Etiquetas estimadas para cada una de las muestras.

    Adicionales
    -----------

    PRI : Una solución alternativa la cual utiliza el la función de costo del PRI y la combina con los algoritmos de punto fijo y gradiente descendiente, dicha función se puede encontrar en la clase con el mismo nombre

    
    Nota
    ----

    La función de costo del PRI
    
    J(X|Xo) = H2(X) + λDcs(X,Xo)
    Cuando λ --> ∞
    J(X|Xo) = -(1-λ)log(V(X))-2λlog(V(X,Xo))
    V(X) = 1/N^2 ∑_(i=1)^N ∑_(j=1)^N k(xi,xj)
    V(X,Xo) = 1/(NNo) ∑_(i=1)^N∑_(j=1)^No 

    '''

    def __init__(self, lambda_=2, sigma_initial=30, yota=1, max_epochs=30, reduction_=False, n_clusters=3, nn=5, method='SGD', batch_size=30, optimization = None, learning_rate=0.001,   ss=1, prediction='Gaussian',  show_process=False, gamma_=1000):
        self.lambda_ = lambda_
        self.sigma_initial = sigma_initial
        self.yota = yota
        self.max_epochs = max_epochs
        self.reduction_ = reduction_
        self.n_clusters = n_clusters
        self.nn = nn
        self.method = method
        self.batch_size = batch_size
        self.optimization = optimization
        self.learning_rate = learning_rate
        if ss > 1:
            self.ss = 1
        else:
            self.ss = ss
        self.prediction = prediction
        self.show_process = show_process
        self.gamma_ = gamma_

    def fit(self, X, y=None):
        """ Calcula la representación de los datos mediante el PRI por lotes.

        Parámetros
        ----------
        X : {ndarray, sparse matrix} de la forma (número de muestras, numero de características)
            Conjunto de datos de entrada.

        y : No es usado, se presenta para la siguiente clase por convenciones establecidas
        
        Retorno
        -------
        cluster_centers_ : ndarray de forma (número de grupos, numero de características)
        Coordinadas de los grupos estimados.

        labels_ : ndarray de forma (número de muestras)
        Etiquetas estimadas para cada una de las muestras.
        """
        self.y = y
        try:
            X[:, 0] = X[:, 0]
        except:
            X = np.array(X).astype(float)
        self.X = X
        self.cluster_centers_, self.labels_ = self.pri_MiniBatch(X)
        return self

    def pri_MiniBatch(self, Xo):

        """  Calcula las etiquetas y el conjunto de representación.

        Parámetros
        ----------
        X : {ndarray, sparse matrix} de la forma (número de muestras, número de características)
            Conjunto de datos de entrada.
        
        Retorno
        -------
        self : Objetos resultantes del estimador ajustado.
        """

        #################################################### Preparar los datos #######################################################

        X = np.zeros((np.int(Xo.shape[0] * self.ss), Xo.shape[1]))
        for i in range(Xo.shape[1]):
            X[:, i] = np.random.uniform(low=np.amin(
                Xo[:, i]) - abs(np.amin(Xo[:, i]) * 0.2), high=np.amax(Xo[:, i]) + abs(np.amin(Xo[:, i]) * 0.2), size=(np.int(Xo.shape[0] * self.ss),))

        ##################################################### MAIN #####################################################################

        Xr = Xo
        J = []
        D_ = []
        labels = np.zeros(Xo.shape[0])
        if self.optimization == 'Adam':
            optimization_model = Adam(self.learning_rate)
        elif self.optimization == 'Nadam':
            optimization_model = Nadam(self.learning_rate)
        elif self.optimization == None:
            optimization_model = Gd(self.learning_rate)

        t = 0
        NX = X.shape[0]
        startt = time.time()
        sigma = np.mean(pairwise_distances(Xo, X))
        sigmai = sigma

        for ii in np.arange(0, self.max_epochs):

            sss = StratifiedShuffleSplit(n_splits=int(X.shape[0] / self.batch_size),
                                         train_size=self.batch_size, test_size=X.shape[0] - self.batch_size)

            for train_index, test_index in sss.split(Xo, labels):

                # Compute MiniBAtch Xo
                idxs = np.argsort(labels[train_index])
                Xoi = Xo[train_index[idxs], :]
                NXo = Xoi.shape[0]

                # Compute kernels and potential of information

                K1, K2, K3, V1, _, V3 = Kernel_Estimation(
                    sigma).fit(NX, NXo, X, Xoi)

                # Cost Function

                J.append(-(1 - self.lambda_) * np.log(V1)
                         - 2 * self.lambda_ * np.log(V3))

                Xk = X
                if self.show_process == True:
                    A = -np.log2(np.sum((1 / NX * NXo) * K1))
                    B = -np.log2(np.sum((1 / NX**2) * np.ones((X.shape[0], Xoi.shape[0])) @ K2))
                    C = -np.log2(np.sum((1 / NXo**2) * K3 @ np.ones((Xoi.shape[0], X.shape[0]))))
                    D = 2 * A - B - C

                    D_.append(D)
                    plt.figure(figsize=(15,5))
                    plt.ion()
                    show_proc(Xo, X, ii, J, D_)

                if self.method == 'FP':

                    X = Fp(self.lambda_).step(
                        NXo, NX, V1, V3, K1, K3, Xoi, X)

                elif self.method == 'SGD':

                    FXk = 1 / (NX * sigma**2) * \
                       (K1 @ np.ones(X.shape) * Xk - K1 @ Xk)
                    FXo = 1 / (NXo * sigma**2) * \
                       (K3 @ np.ones(Xoi.shape) * Xk - K3 @ Xoi)

                    g = ((1 - self.lambda_) * FXk
                         / V1 + (self.lambda_ * FXo) / V3)

                    if self.optimization != None:
                        X = optimization_model.step(g, X)
                    else:
                        X = optimization_model.step(g, X)

                # Update sigma

                sigma = (self.sigma_initial * sigmai) / (self.yota * t + 1)
                t += 1
                # Save results
                Xf = X

        if (self.reduction_ is not None):
            # save sigma
            self.sigma = sigma

            if self.reduction_ == 'SC':

                # Sc
                sc = SpectralClustering(
                    n_clusters=self.n_clusters, gamma=self.gamma_, n_neighbors=5)
                labels_ = sc.fit(Xf).labels_
                D = self.Divergencecs(Xo, Xf)
                c = 1
                labels = np.zeros(D.shape[0])
                for i in range(D.shape[0]):
                    labels[i] = stats.mode(
                        labels_[np.argsort(c * D[i, :])[:self.nn]], keepdims=True)[0]

            elif self.reduction_ == 'Kmeans':
                # Kmeans
                Xr = KMeans(n_clusters=self.n_clusters).fit(Xf)
                Xr = Xr.cluster_centers_
                Xf = Xr
                DM = self.Divergencecs(Xo, Xf)
                labels = np.argmin(DM, axis=1)

        if self.show_process == True:
            plt.ioff()

        # organize labels respect to original labels

        if self.y is not None:
            labels = Lconvert().fit(labels, self.y)

  
        self.J = J

        return Xf, labels

    def Divergencecs(self, Xo, X):
        """  Calcula la divergencia de Cauchy-Schwarz.

        Parámetros
        ----------
        X : {ndarray, sparse matrix} de la forma (número de muestras, número de características)
            Conjunto de datos de entrada.
        Xo : {ndarray, sparse matrix} de la forma (número de muestras, número de características)
        Conjunto de datos procesados.
        
        Retorno
        -------
        D : Divergencia entre de Cachy-Schwarz entre X y Xo.

        """
        N = np.shape(X)[0]
        No = np.shape(Xo)[0]
        K3, K2, K1, _, _, _ = Kernel_Estimation(
            self.sigma).fit(N, No, Xo, X)

        A = -np.log((1 / N * No) * K1 + 1e-4)
        B = -np.log((1 / N**2) * np.ones((Xo.shape[0], X.shape[0])) @ K2)
        C = -np.log((1 / No**2) * K3 @ np.ones((Xo.shape[0], X.shape[0])))
        D = 2 * A - B - C

        return D

    def predict(self, X_test):
        """Predice la pertenencia de nuevas muestras al grupo estimado mas cercano.
        
        Parameters
        ----------
        X_test : {array-like, sparse matrix} de la forma (número de muestras, número de características)
            Conjunto de nuevas muestras.
        Returns
        -------
        labels : ndarray de la forma (número de muestras,)
            Etiqueta a la cual cada una de las muestras es asociada.
        """
        try:
            X_test[:, 0] = X_test[:, 0]
        except:
            X_test = np.array(X_test).astype(float)

        if self.prediction == 'Gaussian':
            Dx1 = pairwise_distances(X_test, self.X)
            D = np.exp(-Dx1**2 / (2 * self.sigma**2))
            c = -1
        else:
            D = self.Divergencecs(X_test, self.X)
            c = 1
        labels = np.zeros(D.shape[0])
        for i in range(D.shape[0]):
            labels[i] = stats.mode(
                self.labels_[np.argsort(c * D[i, :])[:self.nn]], keepdims=True )[0]

        return labels

    def get_params(self, deep=True):

        return {"lambda_": self.lambda_, "sigma_initial": self.sigma_initial, "yota": self.yota, 'max_epochs': self.max_epochs,  'reduction_': self.reduction_, 'n_clusters': self.n_clusters, 'nn': self.nn, 'method': self.method, 'batch_size': self.batch_size, 'optimization': self.optimization, 'learning_rate': self.learning_rate,  'ss': self.ss, 'prediction': self.prediction, 'show_process': self.show_process, 'gamma_': self.gamma_}

    def set_params(self, **parameters):

        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

# %%
class Adam(BaseEstimator, TransformerMixin):
    def __init__(self, learning_rate=0.01, min_lr=0.00001,
                 beta_1=0.9, beta_2=0.999, epsilon=1e-7):

        self._lr = learning_rate
        self._beta1 = beta_1
        self._beta2 = beta_2
        self._epsilon = epsilon
        self._min_lr = min_lr
        self.iteration = 0
        self.m = None
        self.v = None

    def step(self, gradient, theta):
        if self.iteration == 0:
            self.iteration = self.iteration + 1
            m = np.zeros(theta.shape)
            self.m = m
            v = np.zeros(theta.shape)
            self.v = v

        m = self._beta1 * self.m + (1 - self._beta1) * gradient
        v = self._beta2 * self.v + (1 - self._beta2) * np.power(gradient, 2)
        m_hat = m / (1 - np.power(self._beta1, self.iteration))
        v_hat = v / (1 - np.power(self._beta2, self.iteration))
        self.step_ = self._lr * m_hat / (np.sqrt(v_hat) + self._epsilon)
        theta = theta - self.step_
        self.m = m
        self.v = v
        self.iteration = self.iteration + 1
        return theta

# %%
class Nadam(BaseEstimator, TransformerMixin):
    def __init__(self, learning_rate=0.01, min_lr=0.00001,
                 beta_1=0.9, beta_2=0.999, epsilon=1e-7):

        self._lr = learning_rate
        self._beta1 = beta_1
        self._beta2 = beta_2
        self._epsilon = epsilon
        self._min_lr = min_lr
        self.iteration = 0
        self.m = None
        self.v = None

    def step(self, gradient, theta):
        if self.iteration == 0:
            self.iteration = self.iteration + 1
            m = np.zeros(theta.shape)
            self.m = m
            v = np.zeros(theta.shape)
            self.v = v
        m = self._beta1 * self.m + (1 - self._beta1) * gradient
        v = self._beta2 * self.v + (1 - self._beta2) * np.power(gradient, 2)
        m_hat = m / (1 - np.power(self._beta1, self.iteration)) + (1
                                                                   - self._beta1) * gradient / (1 - np.power(self._beta1, self.iteration))
        v_hat = v / (1 - np.power(self._beta2, self.iteration))
        self.step_ = self._lr * m_hat / (np.sqrt(v_hat) + self._epsilon)
        theta = theta - self.step_
        self.m = m
        self.v = v
        self.iteration = self.iteration + 1
        return theta

# %%
class Gd(BaseEstimator, TransformerMixin):

    def __init__(self, learning_rate=0.02, decay=0.0001):
        self.learning_rate = learning_rate
        self.decay = decay
        self.iteration = 0
        self.error_pocket = 0

    def step(self, gradient, theta):
        self.learning_rate *= (1.0 / (1.0 + self.decay * self.iteration))
        self.step_ = self.learning_rate * gradient
        theta_t = theta - self.step_
        self.iteration = self.iteration + 1
        return theta_t

# %%
class Fp(BaseEstimator, TransformerMixin):
    def __init__(self, lambda_=15):
        self.lambda_ = lambda_

    def step(self, NXo, NX, V1, V3, K1, K3, Xo, X):

        Xk = X
        c = (V3 / V1) * (NXo / NX)
        eta = (1 - self.lambda_) / self.lambda_
        num = K3 @ np.ones(Xo.shape)
        X = c * eta * (K1 @ Xk / num) + K3 @ Xo / num - c * \
            eta * (K1 @ np.ones(X.shape) / num) * Xk
        return X

# %%
class Lconvert(BaseEstimator, TransformerMixin):
    def __init__(self, a=0):
        self.a = a

    def fit(self, labels_pre, labels_true):
        labels_conv = np.zeros(labels_pre.shape)
        u = np.unique(labels_pre)
        for i in u:
            labels_conv[labels_pre == i] = stats.mode(
                labels_true[labels_pre == i], self.a, keepdims=True)[0]
        return labels_conv

# %%
class Kernel_Estimation(BaseEstimator, TransformerMixin):
    def __init__(self, sigma=70):
        self.sigma = sigma

    def fit(self, NX, NXo, X, Xo):

        # Distances

        Dx1 = pairwise_distances(X, X)
        Dx2 = pairwise_distances(Xo, Xo)
        Dx3 = pairwise_distances(X, Xo)

        # Kernels

        #K1 = 1 / np.sqrt(2 * np.pi * self.sigma**2) * \
        #    np.exp(-Dx1**2 / (2 * self.sigma**2))
        #K2 = 1 / np.sqrt(2 * np.pi * self.sigma**2) * \
        #    np.exp(-Dx2**2 / (2 * self.sigma**2))
        #K3 = 1 / np.sqrt(2 * np.pi * self.sigma**2) * \
        #    np.exp(-Dx3**2 / (2 * self.sigma**2))

        K1 = np.exp(-Dx1**2 / (2 * self.sigma**2))
        K2 = np.exp(-Dx2**2 / (2 * self.sigma**2))
        K3 = np.exp(-Dx3**2 / (2 * self.sigma**2))

        # Potential of information

        V1 = 1 / (NX**2) * np.sum(K1)
        V2 = 1 / (NXo**2) * np.sum(K2)
        V3 = 1 / (NX * NXo) * np.sum(K3)

        return K1, K2, K3, V1, V2, V3

# %%
def show_proc(Xo, X, i, J, D):

    plt.clf()
    plt.subplot(131)
    plt.scatter(Xo[:, 0], Xo[:, 1], marker='.')
    plt.scatter(X[:, 0], X[:, 1], marker='.')
    plt.title('iteracion: ' + str(i))
    plt.axis('off')
    plt.subplot(132)
    plt.plot(J)
    plt.title('Funcion de costo')
    plt.subplot(133)
    plt.plot(D)
    plt.title('Divergencia Cauchy-Schwartz')
    plt.pause(0.01)
    plt.show()

# %%
def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)
