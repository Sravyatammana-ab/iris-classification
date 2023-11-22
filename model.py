from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import pickle as pkl
from sklearn.ensemble import RandomForestClassifier
iris=load_iris()
x=iris.data
y=iris.target
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=8)
rf=RandomForestClassifier()
rf=rf.fit(x_train,y_train)
pkl.dump(rf,open('model.pkl','wb'))
