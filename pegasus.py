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
#from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier

#from feature_selector import selected_features
#from PCA import pca

#take the features into array


#Do PCA on everyhing and add it to the table
#Just randomly shuffle the whole dataset, 10 times using np.random.shuffle
def testing(model,X_train,y_train,X_test,y_test):
    model.fit(X_train,y_train)
    y_predict = model.predict(X_test)
    score = model.score(y_test,y_predict)
    return score

def scoring(X,y,pipe_lr):
    scores=[]
    skf = StratifiedKFold(n_splits=10, random_state=None, shuffle=True)
    k=0
    for train_index, test_index in skf.split(X,y):
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        pipe_lr.fit(X_train,y_train)
        score=pipe_lr.score(X_train,y_train)
        scores.append(score)
        k+=1
        print('Fold %s, acc %.3f' % (k,score))
        
    print("CROSS VALIDATION ACCURACY: %.3f +/- %.3f" % (np.mean(scores),np.std(scores)))
    

data=pd.read_csv('testing.csv')
data.drop_duplicates(keep=False,inplace=True)
print(data.shape)
data.to_csv('no_dups.csv')
#full dataset
X,Y = data.iloc[:,:-1].values, data.iloc[:,-1].values
X_train,y_train = X,Y
#X_train, X_test, y_train, y_test =  train_test_split(X,Y, test_size=0.1, random_state=0)
#stdsc = StandardScaler()
#X_train_std = stdsc.fit_transform(X_train)
#X_test_std = stdsc.transform(X_test)

'''
X_train_selected,X_test_selected=selected_features.sel(X_train_std,X_test_std,y_train,y_test,data)
knn=KNeighborsClassifier(n_neighbors=10)
X_train_pca, X_test_pca=pca.feat_sel()
testing(knn,X_train,y_train,X_test,y_test)
'''

print('\nRANDOM FOREST\n')
#random = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=1))
pipe_lr= Pipeline([('scl', StandardScaler()),('clf',RandomForestClassifier(n_estimators=500, random_state=1, n_jobs=1))])
scoring(X_train,y_train,pipe_lr)



print('\nKNN\n')
pipe_lr= Pipeline([('scl', StandardScaler()),('clf',KNeighborsClassifier(n_neighbors=4))])
scoring(X_train,y_train,pipe_lr)



print('\nLOGISTIC REGRESSION\n')
pipe_lr= Pipeline([('scl', StandardScaler()),('clf',LogisticRegression(solver = 'lbfgs', penalty='l2', C=0.1))])
scoring(X_train,y_train,pipe_lr)

print('\nSVC\n')
pipe_svc = Pipeline([('scl', StandardScaler()),('clf',SVC(gamma = 'auto'))])
scoring(X_train,y_train,pipe_svc)


print("\nNeural Nets - adam \n")
pipe_lr= Pipeline([('scl', StandardScaler()),('clf',MLPClassifier(solver='adam', alpha = 1e-5 ,max_iter=5000 , hidden_layer_sizes=(20,10), random_state=1))])
scoring(X_train,y_train,pipe_lr)


print("\nNeural Nets - sgd \n")
pipe_lr= Pipeline([('scl', StandardScaler()),('clf',MLPClassifier(solver='sgd', alpha = 1e-5, max_iter=5000 , hidden_layer_sizes=(20,10), random_state=1))])
scoring(X_train,y_train,pipe_lr)

print("\nNeural Nets - lbfgs\n")
pipe_lr= Pipeline([('scl', StandardScaler()),('clf',MLPClassifier(solver='lbfgs', alpha = 1e-5 , hidden_layer_sizes=(20,10), random_state=1))])
scoring(X_train,y_train,pipe_lr)
# ('pca',PCA(n_components=23))
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