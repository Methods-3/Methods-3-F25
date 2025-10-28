#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 13:19:45 2024

@author: lau
"""

#%% early imports

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
sns.set(style='whitegrid', context='notebook')

def plot_decision_regions(X, y, classifier, resolution=0.02):
    # setup marker generator and color map
    plt.figure()
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
    np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
        alpha=0.8, c=cmap(idx),
        marker=markers[idx], label=cl)
    plt.show()

#%% LOGISTIC REGRESSION

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

X, y = load_iris(return_X_y=True)
X = X[:, 0:3:2]  ## petal length and sepal length

logreg = LogisticRegression(penalty=None) ## no regularisation
logreg.fit(X, y)
print(logreg.coef_.shape)

#%% plot logreg

plot_decision_regions(X, y, logreg)

#%% chatgpt

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss


# Generate a synthetic binary classification dataset
X, y = load_iris(return_X_y=True)
X = X[:, 0:3:2]  ## petal length and sepal length
# Split into train and test sets

# Initialize lists to track loss
loss = []

# Create SGDClassifier for logistic regression
sgd_clf = SGDClassifier(loss='log_loss', max_iter=1, tol=1e-4, warm_start=True,
                        penalty=None)

# Perform multiple iterations manually
for i in range(100):  # Number of iterations
    sgd_clf.fit(X, y)
    
    # Predict probabilities (for loss computation)
    y_prob = sgd_clf.predict_proba(X)
    
    # Calculate log-loss
    loss.append(log_loss(y, y_prob))


# Plot the loss over iterations
plt.figure()
plt.plot(loss, label='Loss')
plt.xlabel('Iteration')
plt.ylabel('Log Loss')
plt.legend()
plt.show()

#%% 


