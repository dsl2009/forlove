from keras.layers.core import Dense
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import math
import os
NUM_TIMESTEPS = 60


data = np.load('data.npy')
data = data.reshape(-1, 3)
scaler = MinMaxScaler(feature_range=(0, 1), copy=False)
data = scaler.fit_transform(data)
print(data.shape)
X = np.zeros((data.shape[0], NUM_TIMESTEPS, data.shape[1]))
Y = np.zeros((data.shape[0],data.shape[1]))

for i in range(len(data) - NUM_TIMESTEPS - 1):
    X[i] = data[i:i + NUM_TIMESTEPS]

    Y[i] = data[i + NUM_TIMESTEPS + 1]

#X = np.expand_dims(X, axis=2)

sp = int(0.8 * len(data))
Xtrain, Xtest, Ytrain, Ytest = X[0:sp], X[sp:], Y[0:sp], Y[sp:]

print(Xtrain.shape, Xtest.shape, Ytrain.shape, Ytest.shape)

HIDDEN_SIZE = 10
BATCH_SIZE = 1


model = Sequential()
model.add(LSTM(HIDDEN_SIZE, stateful=True,
    batch_input_shape=(BATCH_SIZE, NUM_TIMESTEPS, 3),
    return_sequences=False))
model.add(Dense(3))
model.compile(loss="mean_squared_error", optimizer="adam",
    metrics=["mean_squared_error"])


model.fit(Xtrain, Ytrain, epochs=10, batch_size=BATCH_SIZE,
    validation_data=(Xtest, Ytest),
    shuffle=False)
