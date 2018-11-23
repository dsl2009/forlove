from torch import nn
import torch.utils.data as Data
import torchvision
from sklearn.preprocessing import MinMaxScaler
import numpy as np

import torch
class RNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn=torch.nn.LSTM(
            input_size=3,
            hidden_size=20,
            num_layers=1,
            batch_first=True
        )
        self.liner = nn.Linear(20, 3)

    def forward(self,x):
        output,(h_n,c_n)=self.rnn(x)
        x = output[:,-15:,:]
        x = self.liner(x)
        return x


NUM_TIMESTEPS = 90
data = np.load('data.npy')
data = data.reshape(-1, 3)
scaler = MinMaxScaler(feature_range=(0, 1), copy=False)
data = scaler.fit_transform(data)
X = np.zeros((data.shape[0]-NUM_TIMESTEPS, NUM_TIMESTEPS, data.shape[1]))
Y = np.zeros((data.shape[0]-NUM_TIMESTEPS,data.shape[1]))



for i in range(len(data) - NUM_TIMESTEPS - 1):
    X[i] = data[i:i + NUM_TIMESTEPS]

    Y[i] = data[i + NUM_TIMESTEPS + 1]

sp = int(0.9 * X.shape[0])
Xtrain, Xtest, Ytrain, Ytest = X[0:sp], X[sp:], Y[0:sp], Y[sp:]