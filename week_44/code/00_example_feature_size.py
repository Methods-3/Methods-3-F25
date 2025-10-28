#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 16:33:24 2021

@author: lau
"""

#%% IMPORT AND READ DATASET
from sklearn import datasets
iris = datasets.load_iris()
X = iris.data[:, 0:1] ## just picking one feature
y = iris.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)


#%% PLOT STANDARDISED VERSUS ORIGINAL DATA

import matplotlib.pyplot as plt

plt.figure()
plt.scatter(X_train, y_train)
plt.xlabel(iris.feature_names[0])
plt.ylabel('Flower type')
plt.show()

plt.figure()
plt.scatter(X_train_std, y_train)
plt.xlabel('Standardised: ' + iris.feature_names[0])
plt.ylabel('Flower type')
plt.show()


#%% CORRELATION MATRIX

import numpy as np
import seaborn as sns


plt.close('all')
cm = np.corrcoef(iris.data.T)
sns.set(font_scale=0.75)
hm = sns.heatmap(cm,
                 cbar=True,
                 annot=True,
                 square=True,
                 fmt='.2f',
                 annot_kws={'size': 15},
                 yticklabels=iris.feature_names,
                 xticklabels=iris.feature_names,
                 vmin=-1.0, vmax=1.0)

plt.show()

#%% LOGISTIC REGRESSION

from sklearn.linear_model import LogisticRegression
logR = LogisticRegression(penalty=None) # no regularisation
logR.fit(X_train_std, y_train)
print(logR.score(X_test_std, y_test))

#%% NOW WITH ALL FEATURES

features = (0, 4)

X = iris.data[:, features[0]:features[1]]
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=0)

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

plt.figure(figsize=(12, 9))
for i in range(features[0], features[1]):
    plt.subplot(int('22' + str(i+1)))
    plt.scatter(iris.data[:, i], y)
    plt.xlabel(iris.feature_names[i])
    plt.ylabel('Flower type')
    plt.show()
    
logR = LogisticRegression(penalty=None) # no regularisation
logR.fit(X_train_std, y_train)
print(logR.score(X_test_std, y_test))

#%% WITH CROSS-VALIDATION

from sklearn.model_selection import cross_val_score, StratifiedKFold
# Generate test sets such that all contain the same distribution of classes,
# or as close as possible.
cv = StratifiedKFold()

X = iris.data[:, 0:4]
y = iris.target

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X)
X_std = sc.transform(X)

scores = cross_val_score(logR, X_std, y, cv=cv)
print(np.mean(scores))

logR_l2 = LogisticRegression(penalty='l2', C=10)
scores_l2 = cross_val_score(logR_l2, X_std, y, cv=cv)
print(np.mean(scores_l2))


#%% SVM

cv = StratifiedKFold()

X = iris.data[:, 0:4]
y = iris.target

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X)
X_std = sc.transform(X)
from sklearn.svm import SVC

svm = SVC(kernel='linear', C=1, random_state=7, gamma=1e1) ## try to set it low

scores_svm = cross_val_score(svm, X_std, y, cv=cv)
print(np.mean(scores_svm))

#%% COEFFICIENTS

logR.fit(X_train_std, y_train)
svm.fit(X_train_std, y_train)

print(logR.coef_) # log odds
print(svm.coef_) # weights that maximise the margin?


#%% kernel

plt.close('all')

def kernel(x1, x2, sd):
    gamma = 1 / sd**2
    return np.exp(-gamma * np.abs(x1 - x2)**2)

x1 = X_std[:, 0:1]
x2 = X_std[:, 1:2]

sds = [100, 10, 1, 0.1, 0.01]
for sd in sds:
    k = kernel(x1, x2, sd)
    plt.figure()
    plt.plot(k)
    plt.xticks(ticks=range(len(y)), labels=y)
    plt.title('Similarity between features x1 and x2 with sd. ' + str(sd))
    plt.ylim(-0.1, 1.1)
    plt.show()
