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
scaler = StandardScaler()
scaler.fit(train_data)
processed_train_data = scaler.transform(train_data)
scaler.fit(val_data)
processed_val_data = scaler.transform(val_data)
scaler.fit(test_data)
processed_test_data = scaler.transform(test_data)
classifier1 = SVC(kernel='rbf', C=2, gamma = 0.05)
classifier1.fit(processed_train_data,train_target)
from sklearn.externals import joblib
joblib.dump(classifier1,'./models/SVMModel.joblib')
C_range = np.logspace(-2, 10, 13)
gamma_range = np.logspace(-9, 3, 13)
param_grid = dict(gamma=gamma_range, C=C_range)
cv = StratifiedShuffleSplit(n_splits=10, test_size=0.2)
grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
grid.fit(processed_train_data, train_target)

joblib.dump(grid,'./models/SVMGridSearch.joblib')
