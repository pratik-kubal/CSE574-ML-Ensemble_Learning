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
classifier1 = SVC(kernel='rbf', C=4, gamma = 0.1,decision_function_shape='ovr',cache_size=7000,verbose=True)
classifier1.fit(train_data,train_target)
from sklearn.externals import joblib
joblib.dump(classifier1,'./models/SVMGridSearchedParamModelC_4.joblib',compress=True)
