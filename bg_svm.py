import pickle
import gzip
import numpy as np
from sklearn.svm import SVC,SVR
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import fetch_mldata
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.model_selection import cross_validate,StratifiedShuffleSplit,GridSearchCV
np.random.seed(666)
filename = '../mnist.pkl.gz'
f = gzip.open(filename, 'rb')
training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
f.close()
train_data = training_data[0]
train_target = training_data[1]
val_data = validation_data[0]
val_target = validation_data[1]
test_target = test_data[1]
test_data = test_data[0]
classifier1 = SVC(kernel='rbf', C=2, gamma = 0.05,cache_size=7000,probability=True,verbose=True)
classifier1.fit(train_data,train_target)
from sklearn.externals import joblib
joblib.dump(classifier1,'./models/SVMSlideModelProbT.joblib',compress=True)
#baseline = SVC(kernel='rbf',gamma=1,cache_size=7024,verbose=True,probability=True)
#baseline
#classifier = baseline
#classifier.fit(train_data,train_target)
#joblib.dump(classifier,'./models/SVMpdfModel2.joblib',compress=True)
#baseline3 = SVC(kernel='rbf',cache_size=7000,verbose=True,probability=True)
#classifier = baseline3
#classifier
#classifier.fit(train_data,train_target)
#joblib.dump(classifier,'./models/SVMpdfModel3.joblib',compress=True)
