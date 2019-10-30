from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from splitter import dbhandler
from Seqbacksel import SBS
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from feature_selector import selected_features
from PCA import pca
from sklearn.cross_validation import train_test_split

class KNneighbors:
        data=pd.read_csv('testing.csv')
        #df_stroke_sample= df_stroke.sample(n=10000)
        X,Y = data.iloc[:,:-1].values, data.iloc[:,-1].values

        X_train_std,X_test_std,y_train,y_test= train_test_split(X,Y, test_size=0.1, random_state=0)
        X_train_selected,X_test_selected=selected_features.sel(X_train_std,X_test_std,y_train,y_test,data)
        knn=KNeighborsClassifier(n_neighbors=10)
        X_train_pca, X_test_pca=pca.feat_sel()
        '''
        sbs=SBS(knn, k_features=1)
        sbs.fit(X_train_std, y_train)
        k_feat = [len(k) for k in sbs.subsets_]
        
        plt.plot(k_feat, sbs.scores_, marker='o')
        plt.ylim([0.6,1.1])
        plt.xlabel('Accuracy')
        plt.ylabel('Number of Features')
        plt.grid()
        plt.show()
        
        
        k7=list(sbs.subsets_[3]) #selecting 7 features for better accuracy
        print(df_stroke_sample.columns[1:][k7])
        '''
        
        print('TRAINING WITHOUT SELECTING FEATURES')
        knn.fit(X_train_std,y_train)
        print('Training accuracy:', knn.score(X_train_std, y_train))
        print('Test accuracy:', knn.score(X_test_std,y_test))
        
        '''
        print("AFTER SBS FEATURE SELECTION of KNN::")
        knn.fit(X_train_std[:,k7],y_train)
        print('Training accuracy:', knn.score(X_train_std[:,k7], y_train))
        print('Test accuracy:', knn.score(X_test_std[:,k7],y_test))
        '''
        
        print("AFTER FOREST FEATURE SELECTION of KNN::")
        knn.fit(X_train_selected,y_train)
        print('Training accuracy:', knn.score(X_train_selected, y_train))
        print('Test accuracy:', knn.score(X_test_selected,y_test))
        
        
        print("AFTER PCA:")
        knn.fit(X_train_pca, y_train)
        print('Training Accuracy:', knn.score(X_train_pca, y_train))
        print('Test Accuracy:', knn.score(X_test_pca, y_test))
        
        
        
