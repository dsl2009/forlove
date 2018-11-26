from torch import nn
import torch.utils.data as Data

import os
from torch.nn import functional as F
import numpy as np

import torch
class RNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn=torch.nn.LSTM(
            input_size=3,
            hidden_size=16,
            num_layers=1,
            batch_first=True
        )
        self.liner = nn.Linear(16, 3)

    def forward(self,x):
        output,(h_n,c_n)=self.rnn(x)
        x = output[:,-15:,:]
        x = self.liner(x)
        return x


NUM_TIMESTEPS = 90
data = np.load('traget.npy')
X = np.zeros((274-NUM_TIMESTEPS-15,NUM_TIMESTEPS,3))
Y = np.zeros((274-NUM_TIMESTEPS-15,15, 3))
print(data.shape)
for i in range(data.shape[1] - NUM_TIMESTEPS - 15):
    if data[10, i + NUM_TIMESTEPS :i + NUM_TIMESTEPS + 15].shape[0]==15:
        X[i] = data[10, i:i + NUM_TIMESTEPS]
        Y[i] = data[10, i + NUM_TIMESTEPS :i + NUM_TIMESTEPS + 15]
print(np.where(Y==0))
mean =[257.717463,    381.688098  , 381.688098]
std = [389.969604 ,696.205022 , 692.336528]
sp = int(0.9 * X.shape[0])
Xtrain, Xtest, Ytrain, Ytest = X[0:sp], X[sp:], Y[0:sp], Y[sp:]
print(Xtrain.shape)
print(Ytrain.shape)
mse = nn.MSELoss(size_average=False)
batch_size = 1
net = RNN()
net.cuda()
optimzer = torch.optim.Adam(net.parameters(), lr=0.001)


def run():
    state={'train_loss':100, 'test_loss':100, 'best_loss':100}
    def revert(dts):
        dts[:, 0] = dts[:, 0] * std[0] + mean[0]
        dts[:, 1] = dts[:, 1] * std[1] + mean[1]
        dts[:, 2] = dts[:, 2] * std[2] + mean[2]
        return dts


    def train():
        net.train()
        ls = []
        for k in range(Xtrain.shape[0]):
            ip = Xtrain[k:k + 1]
            target = Y[k:k + 1]
            ip = torch.from_numpy(ip).float()
            target = torch.from_numpy(target).float()

            img, target = torch.autograd.Variable(ip.cuda()), torch.autograd.Variable(target.cuda())
            out = net(img)
            optimzer.zero_grad()

            loss = mse(out, target)
            ls.append(loss.cpu().detach().numpy())

            loss.backward()
            optimzer.step()
            train_loss =  sum(ls) / Xtrain.shape[0]
            state['train_loss'] = train_loss

        print('train_loss', sum(ls) / Xtrain.shape[0])

    def test():
        with torch.no_grad():
            net.eval()
            ls = []
            for k in range(Xtest.shape[0]):
                ip = Xtrain[k:k + 1]
                target = Y[k:k + 1]
                ip = torch.from_numpy(ip).float()
                target = torch.from_numpy(target).float()
                img, target = torch.autograd.Variable(ip.cuda()), torch.autograd.Variable(target.cuda())
                out = net(img)
                loss = mse(out, target)
                ls.append(loss.cpu().detach().numpy())
                ave_val_loss = sum(ls) / Xtest.shape[0]
                state['test_loss'] = ave_val_loss
            print('val_loss', sum(ls) / Xtest.shape[0])

    def pred():
        with torch.no_grad():
            net.eval()
            ip = Xtrain[-1:]
            target = Y[-1:]
            ip = torch.from_numpy(ip).float()
            target = torch.from_numpy(target).float()
            img, target = torch.autograd.Variable(ip.cuda()), torch.autograd.Variable(target.cuda())
            out = net(img)
            print(out.size())
            print(out.cpu().detach().numpy()[0])
            print(target.cpu().detach().numpy()[0])
            loss = mse(out, target)

    min_val_loss = 100
    for step in range(100):
        train()
        test()
        pred()
        if min_val_loss>state['test_loss']:
            state['best_loss'] = state['test_loss']
            min_val_loss = state['test_loss']
            torch.save(net.state_dict(), os.path.join('./log', '1' + '.pth'))
        print(state)


run()






