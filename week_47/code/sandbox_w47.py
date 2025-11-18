#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 13:42:21 2024

@author: lau
"""

#%% example code
# https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier

from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
X, y = make_classification(n_samples=100, random_state=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,
                                                    random_state=1)
clf = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)
clf.predict_proba(X_test[:1])
clf.predict(X_test[:5, :])
clf.score(X_test, y_test)

#%% COV

import numpy as np
import matplotlib.pyplot as plt

cov = np.cov(X.T)
plt.figure()
plt.imshow(cov, vmin=-1.75, vmax=1.75)
plt.show()


#%% DEFAULT PLOTTING

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams.update(mpl.rcParamsDefault)
    
mpl.rcParams['font.size'] = 14
mpl.rcParams['font.weight'] = 'bold'
mpl.rcParams['lines.linewidth'] = 3.0
plt.ion()

#%% PCA

from sklearn.decomposition import PCA

pca = PCA(X.shape[1])
X_train_pca = pca.fit_transform(X_train)

cov_pca = np.cov(X_train_pca.T)
plt.figure(figsize=(16, 9))
plt.subplot(121)
plt.plot(np.diag(cov_pca))
plt.xlabel('# PCA')
plt.xticks(range(1, 21, 5))
plt.ylabel('Variance')
plt.title('Variance explained by PCA')
plt.subplot(122)
plt.plot(np.log(np.diag(cov_pca)))
plt.xlabel('# PCA')
plt.xticks(range(1, 21, 5))
plt.ylabel('Variance (log)')
plt.title('Log-Variance explained by PCA')
plt.show()

#%% GET MNIST

from sklearn.datasets import load_digits

X, y = load_digits(return_X_y=True)

## plot first digit
first_image = np.reshape(X[0, :], (8, 8))
plt.figure()
plt.imshow(first_image, cmap='gray')
plt.xlabel('# pixel')
plt.ylabel('# pixel')
plt.title('True value: ' + str(y[0]))
plt.show()

## Raschka
from neuralnet import NeuralNetMLP
## from sklearn.neural_network import MLPClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=7)

nn = NeuralNetMLP(
                    n_output=10,
                    n_features=X_train.shape[1],
                    n_hidden=50,
                    l2=0,
                    l1=0,
                    epochs=1000,
                    eta=1e-3,
                    alpha=1e-3,
                    decrease_const=1e-5,
                    shuffle=True,
                    minibatches=1,
                    random_state=7
                    )

nn.fit(X_train, y_train, print_progress=True)
#%% plotting
plt.figure()
plt.plot(range(nn.epochs), nn.cost_)
plt.xlabel('Epochs')
plt.ylabel('Cost')
plt.title('Convergence plot')
plt.show()


y_train_pred = nn.predict(X_train)
acc_train = np.sum(y_train == y_train_pred, axis=0) / X_train.shape[0]
print('Training accuracy: %.2f%%' % (acc_train * 100))

y_test_pred = nn.predict(X_test)
acc_test = np.sum(y_test == y_test_pred, axis=0) / X_test.shape[0]
print('Testing accuracy: %.2f%%' % (acc_test * 100))