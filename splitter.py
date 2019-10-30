import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


class dbhandler():
    def split(n_size):
        df_stroke=pd.read_csv('STROKEDOC.csv',sep=',')
        df_stroke_sample= df_stroke.sample(n=n_size)
        X,y=df_stroke_sample.iloc[:,1:11].values, df_stroke_sample.iloc[:,11].values
        X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.3, random_state=0)
        stsdc=StandardScaler()
        X_train_std=stsdc.fit_transform(X_train)
        X_test_std=stsdc.transform(X_test)
        mmslr=MinMaxScaler()
        X_train_norm=mmslr.fit_transform(X_train)
        X_train_norm=mmslr.transform(X_test)
        
        return X_train_std, X_test_std, y_train, y_test
    
    
    
   
        
    
