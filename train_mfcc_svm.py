import os
import os.path
import librosa
import numpy as np
import keras
import scipy
import tqdm
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from keras.utils import to_categorical
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.layers import Conv1D,AveragePooling1D,MaxPooling1D,Dense,Dropout,Activation, Flatten,Reshape
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report
from numpy import array
from keras import optimizers
from sklearn.utils import shuffle
from sklearn.externals import joblib

n_mfcc=13 # mfcc=13
n_frames=431 # number of frames per audio 
path = "//local//scratch//rajshree//LangClassification//Dataset//trainMFCC13//"
file_list = os.listdir(path)
feature=[]
label=[]
count=0

for file_name in file_list:
    file_path=os.path.join(path,file_name)
    feature.append(np.load(file_path).T)
    label.append(file_name[:2])
    count=count+1

norm_feature = stats.zscore(feature,axis=0)

lb=LabelEncoder()
oneHotEncode=to_categorical(lb.fit_transform(label))

# now get test data as well

path1 = "//local//scratch//rajshree//LangClassification//Dataset//testMFCC13//"
file_list1 = os.listdir(path1)
feature1=[]
label1=[]
count1=0

for file_name1 in file_list1:
    file_path1=os.path.join(path1,file_name1)
    feature1.append(np.load(file_path1).T)
    label1.append(file_name1[:2])
    count1=count1+1

norm_feature1 = stats.zscore(feature1,axis=0)

lb1=LabelEncoder()
oneHotEncode1=to_categorical(lb1.fit_transform(label1))

#test data retrieved

#print(norm_feature.shape)
(n_sample,x,y)=norm_feature.shape
new_data=norm_feature.reshape(n_sample,x*y)
#print(new_data.shape)

(n_sample,x,y) = norm_feature1.shape
new_data1=norm_feature1.reshape(n_sample, x*y)

xTrain,xTest,yTrain,yTest = train_test_split(new_data,label,test_size=0.10)

svclassifier = SVC(kernel='rbf',verbose=True)

svclassifier.fit(xTrain, yTrain)
print("SVM model trained!"
y_test_pred = svclassifier.predict(xTest)
print("Training Accuracy=",metrics.accuracy_score(yTest, y_test_pred))
print("Testing SVM model on Test Dataset")
y_pred = svclassifier.predict(new_data1)
print("Confusion Matrix:")
print(confusion_matrix(label1,y_pred))
print("Classification Report:")
print(classification_report(label1,y_pred))

print("saving SVM model..")
joblib.dump(svclassifier,"svm-mfcc13-model_joblib.mdl")

print("SVM model saved")

