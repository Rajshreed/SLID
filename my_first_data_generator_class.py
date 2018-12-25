import numpy as np
import keras

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=32, dim=(16000), n_channels=1,
                 n_classes=3, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Find list of IDs
        X=np.empty([self.batch_size,self.dim])
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        i=0
        for l in list_IDs_temp:
            X[i] = np.load(l)[0:self.dim]
            i=i+1
        y=self.labels[indexes]
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

#    def __data_generation(self, list_IDs_temp):
#        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
#        print(self.dim, self.n_channels)
#        X = np.empty((64,16000))
#        y=np.empty((64),dtype=int)
#        # Generate data
#        for i, ID in enumerate(list_IDs_temp):
#            # Store sample
#            #print("i=",i," ID=",ID," X.shape",X.shape," np.load(ID).shape=",np.load(ID).shape)
#            X[i]=np.load(ID)
#
#            # Store class
#            ID.rsplit('/',1)[0-1]
#            y[i]=self.labels[i]
#
#        return X, y
