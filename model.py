from keras.layers.core import Dense
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import math
import os
NUM_TIMESTEPS = 90


data = np.load('data.npy')
data = data.reshape(-1, 3)
scaler = MinMaxScaler(feature_range=(0, 1), copy=False)
data = scaler.fit_transform(data)
print(data.shape)
X = np.zeros((data.shape[0]-NUM_TIMESTEPS, NUM_TIMESTEPS, data.shape[1]))
Y = np.zeros((data.shape[0]-NUM_TIMESTEPS,data.shape[1]))



for i in range(len(data) - NUM_TIMESTEPS - 1):
    X[i] = data[i:i + NUM_TIMESTEPS]

    Y[i] = data[i + NUM_TIMESTEPS + 1]

#X = np.expand_dims(X, axis=2)

sp = int(0.9 * X.shape[0])
Xtrain, Xtest, Ytrain, Ytest = X[0:sp], X[sp:], Y[0:sp], Y[sp:]

print(Xtrain.shape, Xtest.shape, Ytrain.shape, Ytest.shape)

HIDDEN_SIZE = 20
BATCH_SIZE = 1


model = Sequential()
model.add(LSTM(HIDDEN_SIZE, stateful=True,
    batch_input_shape=(BATCH_SIZE, NUM_TIMESTEPS, 3),
    return_sequences=False))
model.add(Dense(3))
model.compile(loss="mean_squared_error", optimizer="adam",
    metrics=["mean_squared_error"])
model.summary()

model.fit(Xtrain, Ytrain, epochs=10, batch_size=BATCH_SIZE,
    validation_data=(Xtest, Ytest),
    shuffle=False)

x_test = data[-NUM_TIMESTEPS:]
result = []
for x in range(15):
    x_data = np.expand_dims(x_test[-NUM_TIMESTEPS:],0)
    resul = model.predict(x_data)
    print(resul)
    print(x_test[-15:])
    x_test = np.vstack((x_test, resul))

