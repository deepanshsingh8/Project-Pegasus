# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 13:43:59 2019

@author: Deepansh
"""
#import Image
from Seqbacksel import SBS
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import StratifiedKFold
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV




def scoring(X_train,y_train,pipe_lr):
    kfold = StratifiedKFold(y=y_train, n_folds=10 , random_state=1)
    scores=[]
    for k, (train,test) in enumerate(kfold):
        pipe_lr.fit(X_train[train],y_train[train])
        score=pipe_lr.score(X_train[train],y_train[train])
        scores.append(score)
        print('Fold %s,class dist %s, acc %.3f' % (k+1,np.bincount(y_train[train]),score))

    print("CROSS VALIDATION ACCURACY: %.3f +/- %.3f" % (np.mean(scores),np.std(scores)))

data=pd.read_csv('testing.csv')

X_train,y_train = data.iloc[:,:-1].values, data.iloc[:,-1].values
#X_train, X_test, y_train, y_test =  train_test_split(X,Y, test_size=0.2, random_state=0)
#stdsc = StandardScaler()
#X_train_std = stdsc.fit_transform(X_train)
#X_test_std = stdsc.transform(X_test)

print('RANDOM FOREST\n')
pipe_lr= Pipeline([('scl', StandardScaler()), ('pca',PCA(n_components=20)),('clf',RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=1))])
scoring(X_train,y_train,pipe_lr)
print('KNN\n')
pipe_lr= Pipeline([('scl', StandardScaler()), ('pca',PCA(n_components=25)),('clf',KNeighborsClassifier(n_neighbors=4))])
scoring(X_train,y_train,pipe_lr)
print('LOGISTIC REGRESSION\n')
pipe_lr= Pipeline([('scl', StandardScaler()), ('pca',PCA(n_components=23)),('clf',LogisticRegression(random_state=1))])
scoring(X_train,y_train,pipe_lr)

print('SVC\n')
pipe_svc = Pipeline([('scl', StandardScaler()), ('pca',PCA(n_components=23)),('clf',SVC(random_state=1))])
scoring(X_train,y_train,pipe_svc)

#GRID SEARCH
'''
param_range = [0.0001,0.001,0.01,0.1,1.0,10.0,100.0,1000.0]
param_grid = [{'clf_C': param_range, 'clf_kernel':['linear']}, {'clf_C': param_range, 'clf_gamma': param_range, 'clf_kernel': ['rbf']}]

gs= GridSearchCV(estimator=pipe_svc, param_grid = param_grid, scoring='accuracy', cv=10 , n_jobs=-1)
gs = gs.fit(X_train,y_train)
print(gs.best_score_)
print(gs.best_params_)

#Feature selections:

knn = KNeighborsClassifier(n_neighbors=5)
LogReg = LogisticRegression(random_state = 0, solver = 'lbfgs', multi_class = 'multinomial')
Gauss = GaussianNB()

sbs = SBS(knn, k_features=1)
sbs.fit(X_train_std, y_train)
k_feat = [len(k) for k in sbs.subsets_]
plt.plot(k_feat,sbs.scores_,marker='o')
plt.ylim([0.7,1.1])

plt.grid()
plt.show()


sbs = SBS(LogReg, k_features=1)
sbs.fit(X_train_std, y_train)
k_feat = [len(k) for k in sbs.subsets_]
plt.plot(k_feat,sbs.scores_,marker='o')
plt.ylim([0.7,1.1])

plt.grid()
plt.show()



feat_labels=data.columns[:-1]
forest = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=1)
forest.fit(X_train_std, y_train)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(X_train_std.shape[1]):
    print("%2d) %-*s %f" % (f+1, 30 , feat_labels[f], importances[indices[f]]))
    
    
plt.title('Feature Importances')
plt.bar(range(X_train_std.shape[1]), importances[indices], color='blue', align='center')
plt.xticks(range(X_train_std.shape[1]), feat_labels, rotation=90)
#plt.xlim([-1, X_train_std.shape[1]])
#plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()
#Image.open("feature_importance.png").save('feature_importance.png','JPEG')



forest = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=1)
sbs = SBS(forest, k_features=1)
sbs.fit(X_train_std, y_train)
k_feat = [len(k) for k in sbs.subsets_]
plt.plot(k_feat,sbs.scores_,marker='o')
plt.ylim([0.7,1.1])

plt.grid()
plt.show()
'''