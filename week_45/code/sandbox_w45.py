#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 11:27:33 2024

@author: lau
"""

#%% TERMINAL COMMAND

## pip install ucimlrepo

#%% CODE IMPORT

from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17) 
  
# data (as pandas dataframes) 
X_df = breast_cancer_wisconsin_diagnostic.data.features 
y_df = breast_cancer_wisconsin_diagnostic.data.targets 
  
# metadata 
print(breast_cancer_wisconsin_diagnostic.metadata) 
  
# variable information 
print(breast_cancer_wisconsin_diagnostic.variables) 

#%% P. 170, chapter 6

import numpy as np
from sklearn.preprocessing import LabelEncoder
X = X_df.values
y = y_df.values
print(np.unique(y))
print(y.shape)
y = np.squeeze(y) ## removing the singleton dimension
print(y.shape)
le = LabelEncoder()
y = le.fit_transform(y)
print(np.unique(y))



#%% SPLIT DATA

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.20, random_state=1)
    
print(y_train.shape)
print(y_test.shape)


#% DOING IT MANUALLY

#%% scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
## why transform below and not fit_transform??
X_test_std  = sc.transform(X_test)

#%%# dimesionality reduction
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_train_std_pca = pca.fit_transform(X_train_std)
## why transform below and not fit_transform??
X_test_std_pca = pca.transform(X_test_std)

print(X_train_std_pca.shape)


#%%# logistic regression
from sklearn.linear_model  import LogisticRegression
logreg = LogisticRegression(random_state=1, penalty='l2', C=1.0, tol=1e-4)
logreg.fit(X_train_std_pca, y_train)

print('Training accuracy: %.3f' % logreg.score(X_train_std_pca, y_train))
print('Test accuracy: %.3f' % logreg.score(X_test_std_pca, y_test))

#%% DEFAULT PLOTTING

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams.update(mpl.rcParamsDefault)
    
mpl.rcParams['font.size'] = 14
mpl.rcParams['font.weight'] = 'bold'
mpl.rcParams['lines.linewidth'] = 3.0
plt.ion()

#%% PLOTTING - scaling

plt.figure(figsize=(16, 9))
## original scale
plt.subplot(121)
plt.plot(X_train)
plt.xlabel('Sample (#)')
plt.ylabel('Value')
plt.title('Original Scale')

## scaled
plt.subplot(122)
plt.plot(X_train_std)
plt.xlabel('Sample (#)')
plt.ylabel('Value')
plt.title('Standardised data')

## general
plt.suptitle('Each line is a feature')

plt.show()

#%% PLOTTING CORR COEF - std

import seaborn as sns

plt.close('all')
cm = np.corrcoef(X_train_std.T)
sns.set(font_scale=0.75)
hm = sns.heatmap(cm,
            cbar=True,
             # annot=True,
            square=True,
            fmt='.2f',
            annot_kws={'size': 15},
            # cbar_kws={'fontsize': 15},
             yticklabels=breast_cancer_wisconsin_diagnostic.data.headers[1:-1],
             xticklabels=breast_cancer_wisconsin_diagnostic.data.headers[1:-1],
            vmin=-1.0, vmax=1.0)

# Increase colorbar font size
cbar = plt.gcf().axes[-1]  # Get colorbar axis
cbar.tick_params(labelsize=18)  # Change tick font size

plt.xticks(fontsize=13)  # Set x-axis tick label font size
plt.yticks(fontsize=13)  # Set y-axis tick label font size

plt.show()

#%% PLOTTING - pca

plt.figure(figsize=(16, 9))
## original scale
plt.subplot(121)
plt.plot(X_train_std)
plt.xlabel('Sample (#)')
plt.ylabel('Value')
plt.title('Before PCA')

## scaled
plt.subplot(122)
plt.plot(X_train_std_pca)
plt.xlabel('Sample (#)')
plt.ylabel('Value')
plt.title('After PCA')

## general
plt.suptitle('Each line is a feature')

plt.show()

#%% PLOTTING CORR COEF - pca

import seaborn as sns

plt.close('all')
cm = np.corrcoef(X_train_std_pca.T)
sns.set(font_scale=0.75)
hm = sns.heatmap(cm,
        cbar=True,
          # annot=True,
        square=True,
        fmt='.2f',
        annot_kws={'size': 18},
        # cbar_kws={'fontsize': 15},
         # yticklabels=breast_cancer_wisconsin_diagnostic.data.headers[1:-1],
         # xticklabels=breast_cancer_wisconsin_diagnostic.data.headers[1:-1],
        vmin=-1.0, vmax=1.0)

# Increase colorbar font size
cbar = plt.gcf().axes[-1]  # Get colorbar axis
cbar.tick_params(labelsize=18)  # Change tick font size
# cbar.set_ylabel('Colorbar Label', fontsize=16)  # Change colorbar label font size

plt.show()


#%% PIPELINE WAY

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model  import LogisticRegression
from sklearn.pipeline      import Pipeline

pipe_lr = Pipeline(
    [
         ('scl', StandardScaler()),
         ('pca', PCA(n_components=2)),
         ('clf', LogisticRegression(random_state=1,
                                    penalty='l2',
                                    tol=1e-4,
                                    C=1.0))
    ]
                  )

pipe_lr.fit(X_train, y_train)
print('Training accuracy: %.3f' %  pipe_lr.score(X_train, y_train))
print('Test accuracy: %.3f'     %  pipe_lr.score(X_test, y_test))

#%% STRATIFIED K-FOLD - adapted from p. 177 - manual way of doing it

## note that code is different from when Raschka ran it

import numpy as np
from sklearn.model_selection import StratifiedKFold
kfold = StratifiedKFold(n_splits=10)
scores = list() ## create an empty list

## just checking what kfold.split does
for index, (train_indices, test_indices) in enumerate(kfold.split(X, y)):
    print(f"Fold {index}:")
    print(f"Train indices: indices={train_indices[:10]}")
    print(f"Test indices: indices={test_indices[:10]}")

## actually applying it
for index, (train_indices, test_indices) in enumerate(kfold.split(X, y)):
    pipe_lr.fit(X[train_indices], y[train_indices])
    score = pipe_lr.score(X[test_indices], y[test_indices])
    scores.append(score)
    print('Fold: %s, Class dist.: %s, Acc: %.3f' % (index,
          np.bincount(y[train_indices]), score))
    
print('Cross-validation accuracy: %.3f +/- %.3f' % (
    np.mean(scores), np.std(scores)))    

#%% IMPLEMENTING IT IN PIPELINE

from sklearn.model_selection import cross_val_score
scores = cross_val_score(estimator=pipe_lr, X=X, y=y, cv=10, n_jobs=1)

print('Cross-validation accuracy scores: %s' % scores)     
print('Cross-validation accuracy: %.3f +/- %.3f' % (
    np.mean(scores), np.std(scores)))    



# pipe_lr = Pipeline(
#     [
#          ('scl', StandardScaler()),
#          ('pca', PCA()),
#          ('clf', LogisticRegression(random_state=1,
#                                     penalty='l2',
#                                     tol=1e-4))
#     ]
#                   )
#%% GRID SEARCH CV - setup

from sklearn.model_selection import GridSearchCV
param_range_C   = [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4]
param_range_PCA = np.arange(1, 15) 
## Dictionary entry: name, e.g. 'clf' what it was called in the pipeline, 
## two underscores and then the name of the parameter, e.g. C or n_components
param_grid = [{'clf__C':            param_range_C, # you need two underscores
               'pca__n_components': param_range_PCA}]

grid_search = GridSearchCV(estimator=pipe_lr, param_grid=param_grid,
                           cv=10, n_jobs=-1) ## will use StratifiedKFold

#%% GRID SEARCH CV - fit

grid_search.fit(X, y)
print(grid_search.best_params_)

#%% LOOK AT CONFUSION MATRIX

opt_pipe_lr = Pipeline(
    [
         ('scl', StandardScaler()),
         ('pca', PCA(n_components=9)),
         ('clf', LogisticRegression(random_state=1,
                                    penalty='l2',
                                    tol=1e-4,
                                    C=1.0))
    ]
    )

opt_pipe_lr.fit(X_train, y_train)
from sklearn.metrics import confusion_matrix
y_pred = opt_pipe_lr.predict(X_test)
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
print(confmat)

#%% plot more nicely pp. 190-191

fig, ax = plt.subplots()
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center',
                fontdict=dict(fontsize=30))
plt.xlabel('predicted label', fontdict=dict(fontsize=30))
plt.ylabel('true label', fontdict=dict(fontsize=30))
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.show()