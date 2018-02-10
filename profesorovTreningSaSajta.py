from sklearn.datasets import fetch_mldata
import numpy as np
from keras.models import Sequential
from keras.layers.core import Activation, Dense
from keras.optimizers import SGD

mnist = fetch_mldata('MNIST original')

data   = mnist.data / 255.0
labels = mnist.target.astype('int')

train_rank = 5000
test_rank = 100
#------- MNIST subset --------------------------
train_subset = np.random.choice(data.shape[0], train_rank)
test_subset = np.random.choice(data.shape[0], test_rank)

# train dataset
train_data = data[train_subset]
train_labels = labels[train_subset]

# test dataset
test_data = data[test_subset]
test_labels = labels[test_subset]

def to_categorical(labels, n):
    retVal = np.zeros((len(labels), n), dtype='int')
    ll = np.array(list(enumerate(labels)))
    retVal[ll[:,0],ll[:,1]] = 1
    return retVal

# test = [3, 5, 9]
# print to_categorical(test, 10)

# train and test to categorical
train_out = to_categorical(train_labels, 10)
test_out = to_categorical(test_labels, 10)

# prepare model
model = Sequential()
model.add(Dense(70, input_dim=784))
model.add(Activation('relu'))
#model.add(Dense(50))
#model.add(Activation('tanh'))
model.add(Dense(10))
model.add(Activation('relu'))

# compile model with optimizer
sgd = SGD(lr=0.1, decay=0.001, momentum=0.7)
model.compile(loss='mean_squared_error', optimizer=sgd)

# training
#training = model.fit(train_data, train_out, nb_epoch=500, batch_size=400, verbose=0)
history = model.fit(train_data, train_out,
          batch_size=900, epochs=900,
          verbose=20
          )
#print training.history['loss'][-1]

model.save("profModel.h5")