#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 21:48:43 2024

@author: lau
"""

#%% TERMINAL COMMAND

## pip install ucimlrepo

#%% LOAD WINE QUALITY

from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
wine_quality = fetch_ucirepo(id=186) 
  
# data (as pandas dataframes) 
X = wine_quality.data.features 
y = wine_quality.data.targets 
  
# metadata 
print(wine_quality.metadata) 
  
# variable information 
print(wine_quality.variables) 

#%% NORMALISED COVARIANCE MATRIX

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

plt.close('all')
cm = np.corrcoef(X.T)
sns.set(font_scale=0.75)
hm = sns.heatmap(cm,
                 cbar=True,
                   annot=True,
                 square=True,
                 fmt='.2f',
                 annot_kws={'size': 15},
                 # cbar_kws={'fontsize': 15},
                  yticklabels=wine_quality.data.headers[1:-1],
                  xticklabels=wine_quality.data.headers[1:-1],
                 vmin=-1.0, vmax=1.0)

# Increase colorbar font size
cbar = plt.gcf().axes[-1]  # Get colorbar axis
cbar.tick_params(labelsize=18)  # Change tick font size

plt.xticks(fontsize=11)  # Set x-axis tick label font size
plt.yticks(fontsize=11)  # Set y-axis tick label font size

plt.show()

#%% Exercise in class - overall aim: get the best classification,
# in terms of accuracy, on a test set using logistic regression
# There is a prize for the best classification found
# work in pairs or threes
# OPTIONAL: can you make an even better score with SVM?

## you must start with the following split

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.10, random_state=7)
# Thus the final test will be on y_test as defined here
# NB! This is different from how I did it


## suggested operations in no particular order

#   a. test penalisers L1, L2 or elastic net
#   b. test different number of features in PCA space
#   c. scale variables
#   d. try different solvers (check the solver argument of LogisticRegression)
#   e. try multinomial or OvR classification
#   f. make sure to use the Pipeline function
#   g. also show your confusion matrix

#%% CONFUSION MATRIX
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)
logreg = LogisticRegression(C=0.1)
logreg.fit(X_train_std, y_train)
y_hat = logreg.predict(X_test_std)

from sklearn.metrics import confusion_matrix
confmat = confusion_matrix(y_test, y_hat)

fig, ax = plt.subplots()
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center',
                fontdict=dict(fontsize=20))
plt.xlabel('predicted quality', fontdict=dict(fontsize=20))
plt.ylabel('true quality', fontdict=dict(fontsize=20))
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xticks(ticks=range(7), labels=range(3, 10))
plt.yticks(ticks=range(7), labels=range(3, 10))

plt.show()

#%% PLOT Y VALUES

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams.update(mpl.rcParamsDefault)
    
mpl.rcParams['font.size'] = 20
mpl.rcParams['font.weight'] = 'bold'
mpl.rcParams['lines.linewidth'] = 3.0
plt.ion()

plt.figure()
plt.hist(y, range(np.min(y), np.max(y) + 1))
plt.xlabel('Wine quality')
plt.ylabel('Number of occurrences')
plt.title('Distribution of Wine Quality')
plt.show()