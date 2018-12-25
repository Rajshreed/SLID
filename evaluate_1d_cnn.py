import os
import os.path
import librosa
import numpy as np
import keras
import scipy
import tqdm
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.models import Sequential,load_model
from sklearn.model_selection import train_test_split
from keras.layers import Conv1D,GlobalAveragePooling1D,MaxPooling1D,Dense,Dropout,Activation, Flatten,Reshape
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report
from numpy import array
from keras import optimizers
from sklearn.utils import shuffle

#selection matrix defines the 4x3 array, containing saved_model,feature_name, feature_size per audio
selection_matrix=[['MFCC-13-1dCNNmodel.mdl','MFCC13',13*431],
        ['MFCC-40-1dCNNmodel.mdl','MFCC40',40*431],
        ['MFCC-delta-1dCNNmodel.mdl','MFCCdelta',39*431],
        ['RAW-std-1dCNNmodel.mdl','RAWstd',22050*10]]

#Change select value:: 0 => MFCC 13, 1 => MFCC-40, 2=> MFCC Delta, 3=> RAW std
select=3

#n_mfcc=40 # mfcc=13, delta=13, deltadelta=13

path = "//local//scratch//rajshree//LangClassification//Dataset//test"+selection_matrix[select][1]+"//"
file_list = os.listdir(path)
feature=[]
label=[]
count=0

for file_name in file_list:
    file_path=os.path.join(path,file_name)
    feature.append(np.load(file_path).T)
    label.append(file_name[:2])
    count=count+1
if select != 2 :
    print("select!=2")
    norm_feature = stats.zscore(feature,axis=0)

lb=LabelEncoder()
oneHotEncode=to_categorical(lb.fit_transform(label))

num_labels = oneHotEncode.shape[1]

print("All audio features extracted and normalized")

input_shape = int(selection_matrix[select][2]) #n_mfcc

#special handling for raw data as it is already standardized and saved DS

if select==2 :
    xa=array([*zip(*feature)])
else:
    print("select!=2")
    xa=array([*zip(*norm_feature.T)])

xa = np.reshape(xa,(input_shape, count))
print("xa.shape=",xa.shape)

model = load_model(selection_matrix[select][0])
y_pred = model.predict(xa.T)
print("Evaluation done for - ", selection_matrix[select][1])

score = model.evaluate(xa.T, oneHotEncode, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1],score[1]*100))
print(confusion_matrix(np.argmax(oneHotEncode,axis=1),np.argmax(y_pred,axis=1)))

print(classification_report(np.argmax(oneHotEncode,axis=1),np.argmax(y_pred,axis=1)))

