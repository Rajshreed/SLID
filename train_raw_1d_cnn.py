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
from keras import regularizers
from keras.utils import np_utils
from sklearn import metrics
from numpy import array
from keras import optimizers
from sklearn.utils import shuffle
from my_first_data_generator_class import DataGenerator

#number of features per frame(per sec in this case) i.e. 1 value per frame and total 22050 frames
n_feature=10 
samples_per_sec=22050 
#specify the dimension of feature fed for training, batch_size for training
params = { 'dim': (samples_per_sec*n_feature),
            'batch_size':128,
            'n_classes':3,
            'n_channels':1,
            'shuffle':True}

#If required modify the training dataset path
path = "//local//scratch//rajshree//LangClassification//Dataset//trainRAWstd//"
file_list = os.listdir(path)
feature=[]
label=[]
count=0
train_ID_list=[]
for file_name in file_list:
    file_path=os.path.join(path,file_name)
    train_ID_list.append(str(file_path))
    label.append(file_name[:2])
    count=count+1
#If required modify the testing dataset path
path1 = "//local//scratch//rajshree//LangClassification//Dataset//testRAWstd//"
file_list1 = os.listdir(path1)
label1=[]
count1=0
test_ID_list=[]
for file_name1 in file_list1:
    file_path1=os.path.join(path1,file_name1)
    test_ID_list.append(str(file_path1))
    label1.append(file_name1[:2])
    count1=count1+1

lb=LabelEncoder()
oneHotEncode=to_categorical(lb.fit_transform(label))

lb1=LabelEncoder()
oneHotEncode1=to_categorical(lb1.fit_transform(label1))

training_generator= DataGenerator(train_ID_list,oneHotEncode,**params)
validation_generator=DataGenerator(test_ID_list,oneHotEncode1,**params)


num_labels = 3
print("All audio features extracted and normalized")

# sampling_rate is the time frames extracted from audio files by librosa library

TIME_PERIODS = samples_per_sec
num_sensors = n_feature 
input_shape = samples_per_sec*n_feature

#build model
model = Sequential()

model.add(Reshape((TIME_PERIODS, num_sensors),input_shape=(input_shape,)))
model.add(Conv1D(64,3,activation='relu',kernel_regularizer=regularizers.l2(0.0001),input_shape=(TIME_PERIODS,num_sensors)))
model.add(Conv1D(64,3,kernel_regularizer=regularizers.l2(0.0001),activation='relu'))
model.add(MaxPooling1D(3))
model.add(Dropout(0.5))
model.add(Conv1D(64,3,kernel_regularizer=regularizers.l2(0.0001),activation='relu'))
model.add(Conv1D(64,3,kernel_regularizer=regularizers.l2(0.0001),activation='relu'))

model.add(GlobalAveragePooling1D())
model.add(Dropout(0.5))

model.add(Dense(num_labels,activation='softmax'))
print(model.summary())

callbacks_list = [
    keras.callbacks.ModelCheckpoint(
        filepath='best_model{epoch:02d}-{val_loss:.2f}.h5',
        monitor='val_loss',save_best_only=True),
    keras.callbacks.EarlyStopping(monitor='acc', patience=1)
]

model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam')

print("model built and compiled")

model.fit_generator(generator=training_generator, validation_data=validation_generator, use_multiprocessing=True, workers=15,epochs=40) #,callbacks=callbacks_list,epochs=25)

print("model trained and now saving the model..")

model.save("RAW-std-1dCNNmodel.mdl")
print("model saved.")

