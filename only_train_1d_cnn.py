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
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.layers import Conv1D,GlobalAveragePooling1D,MaxPooling1D,Dense,Dropout,Activation, Flatten,Reshape
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn import metrics
from numpy import array
from keras import optimizers
from sklearn.utils import shuffle

#feature_name is name of the feature to be used for training, various values are specified below
#n_mfcc is the number of mfcc coefficients extracted for the feature used for training

n_mfcc=40 # mfcc13=13, mfcc40=40, mfcc-delta=39 (delta=13+ deltadelta=13)
feature_name="40"# 13, 40, delta
#Dataset full path - change dataset directory path as per required 
path = "//local//scratch//rajshree//LangClassification//Dataset//trainMFCC"+feature_name+"//"

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

num_labels = oneHotEncode.shape[1]

print("All audio features extracted and normalized")

# 431 is the time frames extracted from audio files by librosa library
TIME_PERIODS = 431
num_features = n_mfcc
input_shape = 431*n_mfcc

#build model
model = Sequential()

model.add(Reshape((TIME_PERIODS, num_features),input_shape=(input_shape,)))
model.add(Conv1D(100,10,activation='relu',input_shape=(TIME_PERIODS,num_features)))
model.add(Conv1D(100,10,activation='relu'))
model.add(MaxPooling1D(3))
model.add(Conv1D(160,10,activation='relu'))
model.add(Conv1D(160,10,activation='relu'))
model.add(GlobalAveragePooling1D())
model.add(Dropout(0.5))
model.add(Dense(num_labels,activation='softmax'))
print(model.summary())

#uncomment below if using callbacks
#callbacks_list = [
#    keras.callbacks.ModelCheckpoint(
#        filepath='best_model{epoch:02d}-{val_loss:.2f}.h5',
#        monitor='val_loss',save_best_only=True),
#    keras.callbacks.EarlyStopping(monitor='acc', patience=1)
#]

model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam')

BATCH_SIZE = 400
EPOCHS = 40

print("model built and compiled")

xa=array([*zip(*norm_feature.T)])
xa = np.reshape(xa,(input_shape, count))
#print("xa.shape=",xa.shape)


history = model.fit(xa.T,oneHotEncode,epochs=EPOCHS,verbose=1,batch_size=BATCH_SIZE,validation_split=0.2)#,callbacks=callbacks_list)
print("model trained and now saving the model..")

model.save("MFCC-"+feature_name+"-1dCNNmodel.mdl")

print("Model saved")

